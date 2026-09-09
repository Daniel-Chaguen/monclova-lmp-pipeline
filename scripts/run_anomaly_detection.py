"""
Deteccion de anomalias en el PML del MTR para los 9 nodos de Monclova.

Uso:
    python scripts/run_anomaly_detection.py
    python scripts/run_anomaly_detection.py --no-figures --verbose


--------
Único detector, M4 (residual de descomposicion MSTL), sobre la serie
del Mercado de Tiempo Real y emite alertas horarias para todo el periodo
disponible. 

El umbral se calibra por nodo con todo el historico disponible, al cuantil
correspondiente a la tasa de alerta objetivo del config (anomaly.alert_rate).


MSTL descompone la serie en tendencia, estacionalidad diaria (24h),
estacionalidad semanal (168h) y residual. El residual es lo que ninguna de esas
componentes explica, y es donde viven las anomalias.


Salidas
-------
  data/processed/alertas_m4_operativas_<run_id>.csv   alertas horarias
  outputs/figures/m4_alertas_trimestrales_<run_id>.png
  outputs/metrics/anomaly_predict_<run_id>.json
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from common import (PipelineError, ensure_dirs, load_config, print_summary,
                    run_id, save_metrics, setup_logging)


# --------------------------------------------------------------------------- #
# Escala robusta movil
# --------------------------------------------------------------------------- #
def causal_rolling_mad(x: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """
    MAD movil: la fila t usa [t-window, t-1].


    """
    prev = np.concatenate(([np.nan], x[:-1]))
    n = len(prev)
    mad = np.full(n, np.nan)
    if n >= window:
        w = sliding_window_view(prev, window)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = np.nanmedian(w, axis=1)
            d = np.nanmedian(np.abs(w - m[:, None]), axis=1)
        d[(~np.isnan(w)).sum(axis=1) < min_periods] = np.nan
        mad[window - 1:] = d
    return mad


def causal_scale(x: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Sigma equivalente a partir de la MAD."""
    s = causal_rolling_mad(x, window, min_periods) * 1.4826
    return np.where((s == 0) | np.isnan(s), np.nan, s)


# --------------------------------------------------------------------------- #
# Detectores
# --------------------------------------------------------------------------- #

def detector_m4(panel: pd.DataFrame, cfg_an: dict, nodes: list[str],
                fit_end: pd.Timestamp | None, logger) -> pd.DataFrame:
    """
    M4 — residual de descomposicion MSTL.
    """
    from statsmodels.tsa.seasonal import MSTL

    panel = panel.copy()
    panel["m4_resid"] = np.nan
    panel["m4_score"] = np.nan
    periods = tuple(cfg_an["mstl_periods"])

    for node in nodes:
        m = panel["node_id"] == node
        sub = panel.loc[m].sort_values("timestamp").copy()
        y = sub.set_index("timestamp")["pml_log"].asfreq("h").interpolate()

        y_fit = y[y.index < fit_end] if fit_end is not None else y
        if len(y_fit) < max(periods) * 3:
            raise PipelineError(
                f"Historia insuficiente para MSTL en {node}: {len(y_fit)} horas. "
                f"Se requieren al menos {max(periods) * 3}.")

        res = MSTL(y_fit, periods=periods,
                   stl_kwargs={"robust": cfg_an["mstl_robust"]}).fit()

        seas = np.asarray(res.seasonal)
        if seas.ndim == 1:
            seas = seas.reshape(-1, 1)
        s24 = pd.Series(seas[:, 0], index=y_fit.index)
        s168 = (pd.Series(seas[:, 1], index=y_fit.index) if seas.shape[1] > 1
                else pd.Series(0.0, index=y_fit.index))

        tpl24 = s24.groupby(s24.index.hour).median()
        tpl168 = s168.groupby([s168.index.dayofweek, s168.index.hour]).median()

        idx = y.index
        p24 = pd.Series(idx.hour, index=idx).map(tpl24).to_numpy()
        p168 = pd.Series(list(zip(idx.dayofweek, idx.hour)), index=idx).map(tpl168).to_numpy()
        trend = y.shift(1).rolling(max(periods), min_periods=48).median().to_numpy()

        resid = pd.Series(y.to_numpy() - trend - p24 - p168,
                          index=idx).reindex(sub["timestamp"]).to_numpy()

        panel.loc[sub.index, "m4_resid"] = resid
        panel.loc[sub.index, "m4_score"] = np.abs(resid / causal_scale(
            resid, cfg_an["scale_window_hours"], cfg_an["scale_min_periods"]))
        logger.info("[M4] %s ajustado sobre %d horas", node, len(y_fit))

    return panel



# --------------------------------------------------------------------------- #
# PRODUCTO: alertas de M4 sobre todo el periodo
#
# --------------------------------------------------------------------------- #
def export_alertas_operativas(panel, cfg, corrida, logger):
    cfg_an = cfg["anomaly"]
    nodes = sorted(panel["node_id"].unique())

    panel["m4_flag_op"] = 0
    umbrales = {}
    for node in nodes:
        m = panel["node_id"] == node
        ref = panel.loc[m, "m4_score"].dropna()
        thr = ref.quantile(1 - cfg_an["alert_rate"])
        umbrales[node] = float(thr)
        panel.loc[m, "m4_flag_op"] = (panel.loc[m, "m4_score"] > thr).fillna(False).astype(int)

    cols = ["node_id", "timestamp", "hora_dia", "dia_semana", "pml",
            "m4_resid", "m4_score"]
    alertas = (panel.loc[panel["m4_flag_op"] == 1, cols]
               .assign(umbral=lambda d: d["node_id"].map(umbrales),
                       exceso=lambda d: (d["m4_score"] / d["node_id"].map(umbrales)).round(2))
               .sort_values(["node_id", "timestamp"])
               .reset_index(drop=True))

    out = Path(cfg["paths"]["processed_dir"]) / f"alertas_m4_operativas_{corrida}.csv"
    alertas.to_csv(out, index=False)
    logger.info("Alertas operativas M4: %d filas -> %s", len(alertas), out)
    return alertas, umbrales, out


def fig_alertas_trimestrales(panel, alertas, cfg, corrida, ref_node):
    """Seis paneles trimestrales: 18 meses no caben legibles en un solo eje."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    g = panel[panel["node_id"] == ref_node].sort_values("timestamp")
    trimestres = sorted(g["timestamp"].dt.to_period("Q").unique())

    fig, axes = plt.subplots(len(trimestres), 1, figsize=(14, 2.3 * len(trimestres)))
    for ax, q in zip(np.atleast_1d(axes), trimestres):
        w = g[g["timestamp"].dt.to_period("Q") == q]
        a = w[w["m4_flag_op"] == 1]
        ax.plot(w["timestamp"], w["pml"], color="#2c3e50", lw=0.6)
        ax.scatter(a["timestamp"], a["pml"], s=26, color="#c0392b", zorder=5,
                   label=f"{len(a)} alertas" if len(a) else None)
        ax.set_yscale("log")
        ax.set_ylabel("$/MWh", fontsize=8)
        ax.set_title(f"{q}", fontsize=9, loc="left", fontweight="bold")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.tick_params(labelsize=7)
        if len(a):
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(f"Alertas del detector MSTL sobre el mercado de tiempo real — {ref_node}",
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    path = Path(cfg["paths"]["figures_dir"]) / f"m4_alertas_trimestrales_{corrida}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path.name

# --------------------------------------------------------------------------- #
def load_panel(cfg: dict, logger) -> pd.DataFrame:
    """
    Carga la serie del MTR. 
    """
    mtr_path = Path(cfg["paths"]["features_mtr"])
    if not mtr_path.exists():
        raise PipelineError(f"No existe {mtr_path}. Correr build_features.py primero.")

    panel = (pd.read_csv(mtr_path, parse_dates=["timestamp"])
             .sort_values(["node_id", "timestamp"]).reset_index(drop=True))
    logger.info("Panel: %d filas, %d nodos", len(panel), panel["node_id"].nunique())
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deteccion de anomalias en el MTR (detector MSTL)")
    parser.add_argument("--config", help="Ruta a config.yaml")
    parser.add_argument("--no-figures", action="store_true",
                        help="Omite la figura trimestral")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    logger = setup_logging(args.verbose)
    corrida = run_id()
    cfg_an = cfg["anomaly"]

    t0 = time.time()
    panel = load_panel(cfg, logger)
    nodes = sorted(panel["node_id"].unique())
    ref_node = "06MON-115" if "06MON-115" in nodes else nodes[0]

    # El detector se ajusta con toda la historia disponible
    panel = detector_m4(panel, cfg_an, nodes, fit_end=None, logger=logger)

    alertas, umbrales, path_alertas = export_alertas_operativas(
        panel, cfg, corrida, logger)
    if alertas.empty:
        raise PipelineError(
            "No se genero ninguna alerta. Revisar que la serie del MTR tenga "
            "suficiente historia para ajustar MSTL.")

    figura = None
    if not args.no_figures:
        figura = fig_alertas_trimestrales(panel, alertas, cfg, corrida, ref_node)

    por_nodo = alertas.groupby("node_id").size()
    por_mes = alertas.groupby(alertas["timestamp"].dt.to_period("M")).size()

    save_metrics(cfg, "anomaly_predict", {
        "detector": "m4_mstl",
        "tasa_alerta_objetivo": cfg_an["alert_rate"],
        "periodo": [str(panel["timestamp"].min()), str(panel["timestamp"].max())],
        "filas_evaluadas": int(panel["m4_score"].notna().sum()),
        "alertas": len(alertas),
        "tasa_alerta_efectiva": round(len(alertas) / max(int(panel["m4_score"].notna().sum()), 1), 5),
        "alertas_por_nodo": por_nodo.to_dict(),
        "umbrales": umbrales,
        "figura": figura,
        "duracion_segundos": round(time.time() - t0, 1),
    }, corrida)

    print_summary("DETECCION DE ANOMALIAS COMPLETADA", {
        "Periodo": f"{panel['timestamp'].min():%Y-%m-%d} a {panel['timestamp'].max():%Y-%m-%d}",
        "Nodos": len(nodes),
        "Alertas emitidas": f"{len(alertas):,}",
        "Tasa efectiva": f"{len(alertas) / max(int(panel['m4_score'].notna().sum()), 1):.2%} "
                         f"(objetivo {cfg_an['alert_rate']:.1%})",
        "Alertas por nodo": f"min {por_nodo.min()}, max {por_nodo.max()}",
        "Mes con mas alertas": f"{por_mes.idxmax()} ({por_mes.max()})",
        "Salida": path_alertas,
        "Figura": figura or "omitida",
        "Duracion": f"{time.time() - t0:.0f}s",
    })


if __name__ == "__main__":
    main()
