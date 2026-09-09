"""
Preprocesamiento

Uso:
    python scripts/preprocess.py
    python scripts/preprocess.py --input data/raw/otro.csv --verbose

Se identifican valores donde las 4 columnas de precio son 0.0. 
Se imputan con el primer offset semanal disponible (t-168h, t+168h, t-336h, t+336h, ...). Si no hay donante disponible, se pasa a NaN.


"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import (DataQualityReport, PipelineError, ensure_dirs, load_config,
                    print_summary, run_id, save_metrics, setup_logging)

PRICE_COLS = ["pml", "pml_ene", "pml_per", "pml_cng"]


def find_zero_placeholders(data: pd.DataFrame, process: str) -> pd.DataFrame:
    """ 4 columnas de precio en 0.0."""
    sub = data[data["process"] == process]
    mask = np.logical_and.reduce([sub[c] == 0 for c in PRICE_COLS])
    return sub.loc[mask, ["process", "node_id", "timestamp"]].copy()


def build_offsets(primary_lag_h: int, max_tries: int) -> list[int]:
    """offsets: -168, +168, -336, +336, -504, +504 ..."""
    offsets = []
    for k in range(1, max_tries + 1):
        offsets.extend([-primary_lag_h * k, primary_lag_h * k])
    return offsets


def impute_placeholders(data: pd.DataFrame, process: str, primary_lag_h: int,
                        max_tries: int, report: DataQualityReport) -> tuple[pd.DataFrame, dict]:
    """
    Imputa cada placeholder con el primer offset semanal. Marca la trazabilidad en la columna 'mda_imputed'.
    """
    data = data.copy()
    data["mda_imputed"] = False

    flagged = find_zero_placeholders(data, process)
    if flagged.empty:
        return data, {"n_flagged": 0, "n_imputed": 0, "n_unresolved": 0, "offsets": {}}

    flagged_keys = set(zip(flagged["process"], flagged["node_id"], flagged["timestamp"]))
    lookup = data.set_index(["process", "node_id", "timestamp"])
    offsets = build_offsets(primary_lag_h, max_tries)

    idx_by_key = {k: i for i, k in enumerate(
        zip(data["process"], data["node_id"], data["timestamp"]))}

    offset_counts: dict[str, int] = {}
    unresolved: list[tuple] = []
    per_node_imputed: dict[str, int] = {}
    per_node_unresolved: dict[str, int] = {}

    for process_val, node, ts in flagged_keys:
        row_pos = idx_by_key[(process_val, node, ts)]
        donor_found = None

        for off in offsets:
            donor_key = (process_val, node, ts + pd.Timedelta(hours=off))
            if donor_key in flagged_keys or donor_key not in lookup.index:
                continue
            donor_found = (donor_key, off)
            break

        if donor_found is None:
            unresolved.append((process_val, node, ts))
            per_node_unresolved[node] = per_node_unresolved.get(node, 0) + 1
            continue

        donor_key, off = donor_found
        donor = lookup.loc[donor_key]
        for col in PRICE_COLS:
            data.iat[row_pos, data.columns.get_loc(col)] = float(donor[col])
        data.iat[row_pos, data.columns.get_loc("mda_imputed")] = True

        tag = f"t{off:+d}h"
        offset_counts[tag] = offset_counts.get(tag, 0) + 1
        per_node_imputed[node] = per_node_imputed.get(node, 0) + 1

    # Los no resueltos pasan a NaN.
    for process_val, node, ts in unresolved:
        row_pos = idx_by_key[(process_val, node, ts)]
        for col in PRICE_COLS:
            data.iat[row_pos, data.columns.get_loc(col)] = np.nan

    for node, n in sorted(per_node_imputed.items()):
        report.add(nodo=node, tipo_evento=f"placeholder_{process}", cantidad=n,
                   resolucion_aplicada="imputado con lag semanal",
                   detalle="; ".join(f"{k}={v}" for k, v in sorted(offset_counts.items())))

    for node, n in sorted(per_node_unresolved.items()):
        report.add(nodo=node, tipo_evento=f"placeholder_{process}_sin_donante", cantidad=n,
                   resolucion_aplicada="convertido a NaN",
                   detalle="ningun offset semanal disponible en la ventana cargada")

    stats = {
        "n_flagged": len(flagged_keys),
        "n_imputed": sum(per_node_imputed.values()),
        "n_unresolved": len(unresolved),
        "offsets": offset_counts,
        "unresolved_sample": [f"{n} {t:%Y-%m-%d %H:%M}" for _, n, t in unresolved[:5]],
    }
    return data, stats


def check_hard_invariants(df_clean: pd.DataFrame, flagged_mda: pd.DataFrame,
                          stats: dict, cfg_pre: dict) -> None:
    """
    Condiciones que nunca deberian darse con datos sanos. Si se dan, algo esta
    roto en la fuente o en la logica y continuar produciria resultados sin
    sentido, asi que se detiene el pipeline.

    Se distinguen de los eventos de calidad ESPERADOS (imputaciones, huecos
    aislados), que se registran en el reporte y no interrumpen nada.
    """
    # 1. MTR nunca ha tenido placeholders en 18 meses de historico. Si aparecen,
    #    la fuente cambio de comportamiento y hay que revisar antes de seguir.
    flagged_mtr = find_zero_placeholders(df_clean, "MTR")
    if len(flagged_mtr) > 0:
        raise PipelineError(
            f"INVARIANTE ROTO: {len(flagged_mtr)} filas de MTR con placeholders en 0.0. "
            f"Esto nunca ha ocurrido en el historico; la fuente pudo cambiar de "
            f"comportamiento. Primeras: "
            f"{flagged_mtr.head(3)[['node_id', 'timestamp']].to_dict('records')}")

    # 2. Coherencia de la imputacion.
    process = cfg_pre["placeholder_process"]
    non_imputed = df_clean[(df_clean["process"] == process) & (~df_clean["mda_imputed"])]
    overlap = non_imputed.merge(flagged_mda[["node_id", "timestamp"]],
                                on=["node_id", "timestamp"], how="inner")
    overlap = overlap[overlap["pml"].notna()]
    if len(overlap) > 0:
        raise PipelineError(
            f"INVARIANTE ROTO: {len(overlap)} filas no imputadas coinciden con fechas "
            f"marcadas como placeholder.")

    # 3. Huecos prolongados vs. casos aislados.
    n_process = int((df_clean["process"] == process).sum())
    frac = stats["n_unresolved"] / n_process if n_process else 0.0
    if frac > cfg_pre["max_unresolved_fraction"]:
        raise PipelineError(
            f"INVARIANTE ROTO: {stats['n_unresolved']} placeholders sin resolver "
            f"({frac:.2%} de {process}), por encima del limite "
            f"{cfg_pre['max_unresolved_fraction']:.2%}. No es un caso aislado sino un "
            f"hueco prolongado en la fuente."
            f"{stats.get('unresolved_sample')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocesamiento de PML")
    parser.add_argument("--input", help="CSV crudo")
    parser.add_argument("--output", help="CSV limpio de salida")
    parser.add_argument("--config", help="Ruta a config.yaml")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    logger = setup_logging(args.verbose)
    corrida = run_id()
    cfg_pre = cfg["preprocess"]

    in_path = Path(args.input or cfg["paths"]["raw_consolidated"])
    out_path = Path(args.output or cfg["paths"]["clean"])
    if not in_path.exists():
        raise PipelineError(f"No existe el CSV crudo: {in_path}. Correr extract.py primero.")

    t0 = time.time()
    # t-168h necesita historia disponible para encontrar donante.
    df = pd.read_csv(in_path, parse_dates=["timestamp"])
    logger.info("Cargadas %d filas de %s", len(df), in_path)

    process = cfg_pre["placeholder_process"]
    flagged_mda = find_zero_placeholders(df, process)

    report = DataQualityReport("preprocess", cfg, corrida)
    df_clean, stats = impute_placeholders(
        df, process, cfg_pre["primary_lag_hours"], cfg_pre["max_fallback_tries"], report)

    check_hard_invariants(df_clean, flagged_mda, stats, cfg_pre)

    # registra valores negativos
    mda = df_clean[df_clean["process"] == process]
    neg_original = int(((mda["pml"] < 0) & (~mda["mda_imputed"])).sum())
    neg_imputed = int(((mda["pml"] < 0) & (mda["mda_imputed"])).sum())

    df_clean.to_csv(out_path, index=False)
    report_path = report.save()

    save_metrics(cfg, "preprocess", {
        "entrada": str(in_path), "salida": str(out_path),
        "filas": len(df_clean),
        "placeholders_detectados": stats["n_flagged"],
        "placeholders_imputados": stats["n_imputed"],
        "placeholders_a_nan": stats["n_unresolved"],
        "offsets_usados": stats["offsets"],
        "negativos_originales": neg_original,
        "negativos_por_imputacion": neg_imputed,
        "duracion_segundos": round(time.time() - t0, 1),
    }, corrida)

    print_summary("PREPROCESAMIENTO COMPLETADO", {
        "Filas": f"{len(df_clean):,}",
        "Placeholders detectados": f"{stats['n_flagged']} ({stats['n_flagged'] / max(len(mda), 1):.2%} de {process})",
        "Imputados": f"{stats['n_imputed']}  [{', '.join(f'{k}={v}' for k, v in sorted(stats['offsets'].items())) or '-'}]",
        "Sin donante (a NaN)": stats["n_unresolved"],
        "Negativos conservados": f"{neg_original} originales + {neg_imputed} por imputacion",
        "Salida": out_path,
        "Reporte de calidad": report_path or "sin eventos",
    })


if __name__ == "__main__":
    main()
