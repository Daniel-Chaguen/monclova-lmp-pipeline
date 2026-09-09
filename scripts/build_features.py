"""
Feature engineering para los dos tracks (forecasting sobre MDA, deteccion de anomalias sobre MTR).

Uso:
    python scripts/build_features.py
    python scripts/build_features.py --test-start 2025-07-01 --verbose

Todo se calcula agrupando por (process, node_id) para no mezclar series entre
nodos ni entre procesos.

Features obtenidas:
lags: 
Calcula Lags 1, 2, 3, 24, 48, 72 y 168h.

Rolling
Para evitar fuga de datos, se calcula la media y desviacion  sobre valores  pasados.

Codificacion ciclica:
Uso de valores de sen y cos para codificar hora del dia y dia de la semana, preservando la continuidad ciclica. Tambien se agregan columnas de mes y fin de semana.

signed_log1p:
Toma en cuenta la presencia de valores negativos

Ventanas "congeladas" para day-ahead: 
Utiliza ventanas que respetan la condicion de que al momento de predecir cualquier hora del dia D, solo hay datos de hasta un día anterior.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import (PipelineError, ensure_dirs, load_config, print_summary,
                    run_id, save_metrics, setup_logging)

GROUP = ["process", "node_id"]


def add_lags(data: pd.DataFrame, lags: list[int], target: str) -> pd.DataFrame:
    data = data.sort_values(GROUP + ["timestamp"]).copy()
    grp = data.groupby(GROUP)[target]
    for lag in lags:
        data[f"{target}_lag{lag}"] = grp.shift(lag)
    return data


def add_rolling(data: pd.DataFrame, windows: list[int], target: str) -> pd.DataFrame:
    """Media y desviacion moviles sobre valores estrictamente pasados."""
    data = data.sort_values(GROUP + ["timestamp"]).copy()
    shifted = data.groupby(GROUP)[target].shift(1)
    keys = [data[c] for c in GROUP]
    for w in windows:
        roll = shifted.groupby(keys).rolling(w, min_periods=w)
        data[f"{target}_roll{w}_mean"] = roll.mean().reset_index(level=[0, 1], drop=True)
        data[f"{target}_roll{w}_std"] = roll.std().reset_index(level=[0, 1], drop=True)
    return data


def add_calendar(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    ts = data["timestamp"]
    data["fecha_dt"] = ts.dt.normalize()
    data["hora_dia"] = ts.dt.hour
    data["dia_semana"] = ts.dt.dayofweek          # 0 = lunes
    data["is_weekend"] = (data["dia_semana"] >= 5).astype(int)
    data["mes"] = ts.dt.month
    data["hora_sin"] = np.sin(2 * np.pi * data["hora_dia"] / 24)
    data["hora_cos"] = np.cos(2 * np.pi * data["hora_dia"] / 24)
    data["dow_sin"] = np.sin(2 * np.pi * data["dia_semana"] / 7)
    data["dow_cos"] = np.cos(2 * np.pi * data["dia_semana"] / 7)
    return data


def signed_log1p(x):
    """log1p que tolera negativos, preservando el signo."""
    return np.sign(x) * np.log1p(np.abs(x))


def add_frozen_dayahead(data: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Agregados diarios desplazados un dia: la unica informacion de precio
    disponible al momento de publicar el MDA del dia siguiente.
    """
    data = data.copy()
    daily = (data.groupby(GROUP + ["fecha_dt"])[target]
             .agg(["mean", "std"])
             .rename(columns={"mean": "daily_mean", "std": "daily_std"})
             .reset_index()
             .sort_values(GROUP + ["fecha_dt"]))

    g = daily.groupby(GROUP)["daily_mean"]
    daily["roll24_frozen_mean"] = g.shift(1)
    daily["roll24_frozen_std"] = daily.groupby(GROUP)["daily_std"].shift(1)
    daily["roll168_frozen_mean"] = (g.shift(1).groupby([daily[c] for c in GROUP])
                                    .rolling(7, min_periods=7).mean()
                                    .reset_index(level=[0, 1], drop=True))
    daily["roll168_frozen_std"] = (g.shift(1).groupby([daily[c] for c in GROUP])
                                   .rolling(7, min_periods=7).std()
                                   .reset_index(level=[0, 1], drop=True))

    cols = GROUP + ["fecha_dt", "roll24_frozen_mean", "roll24_frozen_std",
                    "roll168_frozen_mean", "roll168_frozen_std"]
    return data.merge(daily[cols], on=GROUP + ["fecha_dt"], how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description="Construccion de features")
    parser.add_argument("--input", help="CSV limpio de preprocess.py")
    parser.add_argument("--config", help="Ruta a config.yaml")
    parser.add_argument("--test-start", help="Frontera del split temporal YYYY-MM-DD")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    logger = setup_logging(args.verbose)
    corrida = run_id()
    cfg_f = cfg["features"]

    in_path = Path(args.input or cfg["paths"]["clean"])
    if not in_path.exists():
        raise PipelineError(f"No existe el CSV limpio: {in_path}. Correr preprocess.py primero.")

    test_start = pd.Timestamp(args.test_start or cfg_f["test_start"])
    target = cfg_f["target_col"]

    t0 = time.time()
    df = pd.read_csv(in_path, parse_dates=["timestamp"])
    df = df.sort_values(GROUP + ["timestamp"]).reset_index(drop=True)
    logger.info("Cargadas %d filas", len(df))

    df = add_lags(df, cfg_f["lags"], target)
    df = add_rolling(df, cfg_f["rolling_windows"], target)
    df = add_calendar(df)
    df[f"{target}_log"] = signed_log1p(df[target])
    df = add_frozen_dayahead(df, target)

    df["split"] = np.where(df["timestamp"] >= test_start, "test", "train")

    if "mda_imputed" not in df.columns:
        raise PipelineError("Falta la columna 'mda_imputed'. El CSV de entrada no viene "
                            "de preprocess.py.")

    out_paths = {}
    for process, key in [("MDA", "features_mda"), ("MTR", "features_mtr")]:
        sub = df[df["process"] == process].reset_index(drop=True)
        if sub.empty:
            logger.warning("Sin filas para el proceso %s", process)
            continue
        path = Path(cfg["paths"][key])
        sub.to_csv(path, index=False)
        out_paths[process] = path

    n_train = int((df["split"] == "train").sum())
    n_test = int((df["split"] == "test").sum())
    max_lag = max(cfg_f["lags"] + cfg_f["rolling_windows"])
    n_groups = df.groupby(GROUP).ngroups

    save_metrics(cfg, "build_features", {
        "entrada": str(in_path),
        "filas": len(df),
        "columnas": len(df.columns),
        "grupos_process_nodo": n_groups,
        "test_start": str(test_start.date()),
        "filas_train": n_train, "filas_test": n_test,
        "nan_por_arranque_esperados": max_lag * n_groups,
        "salidas": {k: str(v) for k, v in out_paths.items()},
        "duracion_segundos": round(time.time() - t0, 1),
    }, corrida)

    print_summary("FEATURES CONSTRUIDOS", {
        "Filas": f"{len(df):,}  ({len(df.columns)} columnas)",
        "Grupos": f"{n_groups} (proceso x nodo)",
        "Split": f"train={n_train:,} / test={n_test:,}  (corte {test_start:%Y-%m-%d})",
        "NaN de arranque": f"~{max_lag * n_groups:,} filas (ventana maxima {max_lag}h)",
        **{f"Salida {k}": v for k, v in out_paths.items()},
    })


if __name__ == "__main__":
    main()
