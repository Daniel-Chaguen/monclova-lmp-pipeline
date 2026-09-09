"""
Pronostico operativo del PML del Mercado del Dia en Adelanto (MDA), 9 nodos de
Monclova.

Uso:
    python scripts/run_forecasting.py
    python scripts/run_forecasting.py --horizons day_ahead --verbose

Que hace
--------
Entrena con todo el historico disponible y emite el pronostico del siguiente
horizonte.

La evaluacion del modelo (backtesting, comparacion contra LSTM y ARIMAX,
tuning de hiperparametros) vive en notebooks/model_selection/forecasting.ipynb.


Dos horizontes
--------------
h1          Pronostica la siguiente hora. 

day_ahead   Pronostica las 24 horas del dia siguiente, con la informacion
            disponible al cierre del dia actual. 

Modelo
------
XGBoost con hiperparametros CONGELADOS en config/config.yaml, uno por nodo y por
horizonte (18 modelos por corrida). El tunning del modelo se hace en notebooks/model_selection/forecasting.ipynb.

Trazabilidad
------------
Cada modelo se serializa en outputs/models/ con el run_id en el nombre, en
formato nativo de XGBoost (.json). Cada fila del CSV de pronosticos referencia el archivo del
modelo que la produjo, asi que siempre se puede reconstruir que predijo que.

Salidas
-------
  outputs/predictions/forecast_<run_id>.csv
  outputs/models/xgb_<horizonte>_<nodo>_<run_id>.json
  outputs/metrics/forecasting_predict_<run_id>.json
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from build_features import add_calendar, add_frozen_dayahead, add_lags, add_rolling
from common import (PipelineError, ensure_dirs, load_config, print_summary,
                    run_id, save_metrics, setup_logging)


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
def fit_xgb(train: pd.DataFrame, features: list[str], params: dict) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(n_jobs=-1, **params)
    model.fit(train[features], train["pml"])
    return model


def save_model(model, cfg: dict, node: str, horizon: str, corrida: str) -> str:
    """
    Formato de XGBoost en .json.
    """
    out = Path(cfg["paths"]["models_dir"]) / f"xgb_{horizon}_{node}_{corrida}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out))
    return out.name


# --------------------------------------------------------------------------- #
# Modo predict
# --------------------------------------------------------------------------- #
def build_future_frame(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Extiende la serie con las horas futuras y recalcula los features con las
    mismas funciones de build_features.py.
    """
    cfg_f = cfg["features"]
    last_ts = df.loc[df["pml"].notna(), "timestamp"].max()

    # Resto del dia mas el dia siguiente completo
    future_end = last_ts.normalize() + pd.Timedelta(days=2) - pd.Timedelta(hours=1)
    future_index = pd.date_range(last_ts + pd.Timedelta(hours=1), future_end, freq="h")

    if len(future_index) == 0:
        raise PipelineError("No hay horizonte futuro que construir; revisar el historico.")

    base = df[["process", "node_id"]].drop_duplicates()
    future = (base.merge(pd.DataFrame({"timestamp": future_index}), how="cross")
              .assign(pml=np.nan, mda_imputed=False))

    keep = ["process", "node_id", "timestamp", "pml", "mda_imputed"]
    extended = (pd.concat([df[keep], future[keep]], ignore_index=True)
                .sort_values(["process", "node_id", "timestamp"])
                .reset_index(drop=True))

    extended = add_lags(extended, cfg_f["lags"], "pml")
    extended = add_rolling(extended, cfg_f["rolling_windows"], "pml")
    extended = add_calendar(extended)
    extended = add_frozen_dayahead(extended, "pml")

    return extended[extended["timestamp"].isin(future_index)].copy(), last_ts


def run_predict(df: pd.DataFrame, cfg: dict, horizons: list[str], corrida: str,
                logger) -> tuple[pd.DataFrame, dict]:
    cfg_fc = cfg["forecasting"]
    nodes = sorted(df["node_id"].unique())

    future, last_ts = build_future_frame(df, cfg)
    next_day = (last_ts.normalize() + pd.Timedelta(days=1)).date()
    first_hour = last_ts + pd.Timedelta(hours=1)

    preds_rows, skipped = [], []

    for hz in horizons:
        spec = cfg_fc["horizons"][hz]
        features = spec["features"]

        # h1 pronostica solo la siguiente hora; day_ahead, las 24 del dia siguiente.
        if hz == "h1":
            target = future[future["timestamp"] == first_hour]
        else:
            target = future[future["timestamp"].dt.date == next_day]

        for node in nodes:
            d = df[df["node_id"] == node].dropna(subset=features + ["pml"])
            if d.empty:
                raise PipelineError(f"Sin datos de entrenamiento para {node}/{hz}.")

            # Modo predict: se entrena con todo el historico.
            model = fit_xgb(d, features, spec["params"])
            model_file = save_model(model, cfg, node, hz, corrida)

            tgt = target[target["node_id"] == node]
            usable = tgt.dropna(subset=features)
            if len(usable) < len(tgt):
                skipped.append(f"{node}/{hz}: {len(tgt) - len(usable)}h sin features completos")
            if usable.empty:
                continue

            yhat = model.predict(usable[features])
            for ts, val in zip(usable["timestamp"], yhat):
                preds_rows.append({"nodo": node, "horizonte": spec["label"],
                                   "timestamp_objetivo": ts, "pml_pronosticado": round(float(val), 2),
                                   "modelo": model_file, "corrida": corrida,
                                   "ultimo_dato_observado": last_ts})
            logger.info("[predict] %s %s -> %d horas", node, hz, len(usable))

    return pd.DataFrame(preds_rows), {"last_ts": last_ts, "skipped": skipped}


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Pronostico operativo de PML (MDA)")
    parser.add_argument("--horizons", nargs="+", choices=["h1", "day_ahead"],
                        help="Horizontes a pronosticar (default: los del config)")
    parser.add_argument("--input", help="CSV de features del MDA")
    parser.add_argument("--config", help="Ruta a config.yaml")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    Path(cfg["paths"]["predictions_dir"]).mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.verbose)
    corrida = run_id()

    in_path = Path(args.input or cfg["paths"]["features_mda"])
    if not in_path.exists():
        raise PipelineError(f"No existe {in_path}. Correr build_features.py primero.")

    horizons = args.horizons or list(cfg["forecasting"]["horizons"])
    t0 = time.time()

    df = pd.read_csv(in_path, parse_dates=["timestamp", "fecha_dt"])
    df = df.sort_values(["node_id", "timestamp"]).reset_index(drop=True)
    logger.info("Cargadas %d filas de %s", len(df), in_path)

    preds, info = run_predict(df, cfg, horizons, corrida, logger)
    if preds.empty:
        raise PipelineError(
            "No se genero ninguna prediccion. Revisar que el historico llegue "
            "hasta una fecha reciente y tenga features completos.")

    out = Path(cfg["paths"]["predictions_dir"]) / f"forecast_{corrida}.csv"
    preds.to_csv(out, index=False)

    save_metrics(cfg, "forecasting_predict", {
        "horizontes": horizons,
        "ultimo_dato_observado": str(info["last_ts"]),
        "predicciones": len(preds),
        "nodos": int(preds["nodo"].nunique()),
        "rango_pronosticado": [str(preds["timestamp_objetivo"].min()),
                               str(preds["timestamp_objetivo"].max())],
        "modelos_guardados": int(preds["modelo"].nunique()),
        "horas_omitidas": info["skipped"],
        "duracion_segundos": round(time.time() - t0, 1),
    }, corrida)

    por_hz = preds.groupby("horizonte").size().to_dict()
    print_summary("PRONOSTICO GENERADO", {
        "Ultimo dato observado": info["last_ts"],
        **{f"Horas pronosticadas ({k})": v for k, v in por_hz.items()},
        "Rango": f"{preds['timestamp_objetivo'].min()} a "
                 f"{preds['timestamp_objetivo'].max()}",
        "Nodos": preds["nodo"].nunique(),
        "Modelos guardados": f"{preds['modelo'].nunique()} en {cfg['paths']['models_dir']}",
        "Salida": out,
        "Advertencias": "; ".join(info["skipped"]) if info["skipped"] else "ninguna",
        "Duracion": f"{time.time() - t0:.0f}s",
    })


if __name__ == "__main__":
    main()
