"""
Extraccion de Precios Marginales Locales desde el SW-PML de CENACE.

Uso:
    python scripts/extract.py --start 2024-01-01 --end 2025-06-30
    python scripts/extract.py --start 2025-07-01 --end 2025-07-15 --processes MDA
    python scripts/extract.py --start 2024-01-01 --end 2024-01-31 --dry-run

El modo --dry-run construye y valida las URLs y el particionado en chunks sin
llamar a la API. Sirve para verificar parametros antes de una corrida larga.

Notas de la fuente (Manual Tecnico SW-PML):
  - La consulta admite de 1 a 7 Dias de Operacion y hasta 20 NodosP.
  - El MDA se publica un dia ANTES del dia de operacion.
  - El MTR (Expost) se publica hasta 7 dias DESPUES del dia de operacion, por lo
    que pedir MTR de los ultimos 7 dias devuelve vacio de forma legitima.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from common import (DataQualityReport, PipelineError, ensure_dirs, load_config,
                    print_summary, run_id, save_metrics, setup_logging)


def build_session(retries: int, backoff_factor: float) -> requests.Session:
    """Sesion persistente con reintentos exponenciales sobre 429 y 5xx."""
    session = requests.Session()
    strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def date_chunks(start: datetime, end: datetime, max_days: int):
    """Rangos consecutivos y sin traslape de nasta max_days (7 en esta api)."""
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=max_days - 1), end)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def available_window(process: str, start: datetime, end: datetime, cfg_ex: dict,
                     today: datetime | None = None):
    """
    Recorta el rango solicitado a lo que la api puede tener publicado.

      - MTR : disponible hasta el Dia de Operacion menos 7.
      - MDA: disponible hasta el Dia de Operacion actual o el siguiente.

    Devuelve (start, end) recortado, o None si no queda ningun dia consultable.
    """
    today = today or datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    if process == "MTR":
        latest = today - timedelta(days=cfg_ex["mtr_publication_lag_days"])
    else:
        latest = today + timedelta(days=cfg_ex["mda_max_future_days"])

    if start > latest:
        return None
    return start, min(end, latest)


def build_url(cfg_ex: dict, process: str, chunk_start: datetime, chunk_end: datetime) -> str:
    nodes = ",".join(cfg_ex["nodes"])
    return (f"{cfg_ex['base_url']}/{cfg_ex['system']}/{process}/{nodes}/"
            f"{chunk_start:%Y/%m/%d}/{chunk_end:%Y/%m/%d}/JSON")


def parse_payload(payload: dict, process: str) -> list[dict]:
    """Convierte el JSON a registros por nodo y hora."""
    records = []
    for nodo in payload.get("Resultados", []):
        node_id = nodo.get("clv_nodo")
        for val in nodo.get("Valores", []):
            records.append({
                "process": process,
                "node_id": node_id,
                "fecha": val.get("fecha"),
                "hora": int(val.get("hora")),
                "pml": float(val.get("pml")),
                "pml_ene": float(val.get("pml_ene")),
                "pml_per": float(val.get("pml_per")),
                "pml_cng": float(val.get("pml_cng")),
            })
    return records


def extract_process(session, cfg_ex, process, chunks, logger, report) -> pd.DataFrame:
    """Descarga un proceso completo (MDA o MTR) recorriendo todos los chunks."""
    records: list[dict] = []
    failed: list[str] = []

    logger.info("Extrayendo proceso %s (%d chunks)", process, len(chunks))

    for idx, (c_start, c_end) in enumerate(chunks, 1):
        url = build_url(cfg_ex, process, c_start, c_end)
        try:
            resp = session.get(url, timeout=cfg_ex["timeout_seconds"])

            if resp.status_code == 200:
                payload = resp.json()
                if payload.get("status") == "OK":
                    records.extend(parse_payload(payload, process))
                else:
                    # NO_DATA es esperado, por ejemplo, MTR de los ultimos 7 dias.
                    report.add(nodo="ALL", tipo_evento=f"{process}_sin_datos", cantidad=1,
                               resolucion_aplicada="chunk omitido",
                               detalle=f"{c_start:%Y-%m-%d} a {c_end:%Y-%m-%d} "
                                       f"status={payload.get('status')}")
            elif resp.status_code == 204:
                report.add(nodo="ALL", tipo_evento=f"{process}_sin_contenido", cantidad=1,
                           resolucion_aplicada="chunk omitido",
                           detalle=f"{c_start:%Y-%m-%d} a {c_end:%Y-%m-%d} HTTP 204")
            elif resp.status_code == 400:
                # Peticion rechazada: el rango excede la ventana o algun parametro no cumple el formato del manual.
                report.add(nodo="ALL", tipo_evento=f"{process}_peticion_rechazada", cantidad=1,
                           resolucion_aplicada="chunk omitido; revisar rango o parametros",
                           detalle=f"{c_start:%Y-%m-%d} a {c_end:%Y-%m-%d} HTTP 400")
                logger.error("[%s] HTTP 400 en %s a %s. El rango puede exceder la ventana "
                             "publicada por CENACE.", process, c_start.date(), c_end.date())
            else:
                failed.append(f"{c_start:%Y-%m-%d}/{c_end:%Y-%m-%d} HTTP {resp.status_code}")
                logger.error("[%s] HTTP %s en %s", process, resp.status_code, url)

        except (requests.RequestException, ValueError) as exc:
            failed.append(f"{c_start:%Y-%m-%d}/{c_end:%Y-%m-%d} {type(exc).__name__}")
            logger.error("[%s] fallo el chunk %s a %s: %s", process, c_start.date(),
                         c_end.date(), exc)

        time.sleep(cfg_ex["request_delay_seconds"])
        if idx % 20 == 0 or idx == len(chunks):
            logger.info("[%s] %d/%d chunks, %d registros", process, idx, len(chunks),
                        len(records))

    if failed:
        report.add(nodo="ALL", tipo_evento=f"{process}_chunks_fallidos", cantidad=len(failed),
                   resolucion_aplicada="reintentos agotados; faltan datos",
                   detalle="; ".join(failed[:10]))

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["fecha"]) + pd.to_timedelta(df["hora"] - 1, unit="h")
    return df.sort_values(["node_id", "timestamp"]).reset_index(drop=True)


def check_coverage(df: pd.DataFrame, cfg_ex: dict, start: datetime, end: datetime,
                   report: DataQualityReport) -> dict:
    """
    Compara horas recibidas contra un cálculo de horas esperadas por nodo y proceso.
    """
    expected_hours = (end - start).days * 24 + 24
    coverage = {}
    for (process, node), g in df.groupby(["process", "node_id"]):
        got = g["timestamp"].nunique()
        coverage[f"{process}/{node}"] = got
        if got < expected_hours:
            report.add(nodo=node, tipo_evento=f"{process}_cobertura_incompleta",
                       cantidad=expected_hours - got,
                       resolucion_aplicada="ninguna; revisar chunks fallidos",
                       detalle=f"esperadas {expected_hours}, recibidas {got}")
    return {"horas_esperadas_por_nodo": expected_hours, "cobertura": coverage}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extraccion de PML desde CENACE SW-PML")
    parser.add_argument("--start", help="Fecha inicial YYYY-MM-DD")
    parser.add_argument("--end", help="Fecha final YYYY-MM-DD")
    parser.add_argument("--processes", nargs="+", choices=["MDA", "MTR"],
                        help="Procesos a extraer (default: los del config)")
    parser.add_argument("--config", help="Ruta a config.yaml")
    parser.add_argument("--out", help="Ruta del CSV consolidado de salida")
    parser.add_argument("--dry-run", action="store_true",
                        help="Valida parametros y URLs sin llamar a la API")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    logger = setup_logging(args.verbose)
    corrida = run_id()

    cfg_ex = cfg["extraction"]
    start = datetime.strptime(args.start or cfg_ex["default_start"], "%Y-%m-%d")
    end = datetime.strptime(args.end or cfg_ex["default_end"], "%Y-%m-%d")
    if end < start:
        raise PipelineError(f"--end ({end:%Y-%m-%d}) es anterior a --start ({start:%Y-%m-%d})")

    processes = args.processes or cfg_ex["processes"]
    out_path = Path(args.out or cfg["paths"]["raw_consolidated"])

    windows = {p: available_window(p, start, end, cfg_ex) for p in processes}
    chunks = list(date_chunks(start, end, cfg_ex["max_days_per_chunk"]))

    if args.dry_run:
        ventana = {}
        for p in processes:
            w = windows[p]
            ventana[f"Ventana {p}"] = ("FUERA DE RANGO, se omite" if w is None
                                       else f"{w[0]:%Y-%m-%d} a {w[1]:%Y-%m-%d}")
        print_summary("EXTRACCION (dry-run, sin llamadas a la API)", {
            "Rango": f"{start:%Y-%m-%d} a {end:%Y-%m-%d} ({(end - start).days + 1} dias)",
            "Procesos": ", ".join(processes),
            **ventana,
            "Nodos": len(cfg_ex["nodes"]),
            "Chunks por proceso": len(chunks),
            "Peticiones totales": len(chunks) * len(processes),
            "Tiempo estimado": f"~{len(chunks) * len(processes) * 1.5 / 60:.1f} min",
            "Salida": out_path,
        })
        print("  Primera URL:")
        print(f"    {build_url(cfg_ex, processes[0], *chunks[0])}")
        print("  Ultima URL:")
        print(f"    {build_url(cfg_ex, processes[-1], *chunks[-1])}\n")
        return

    report = DataQualityReport("extract", cfg, corrida)
    session = build_session(cfg_ex["retries"], cfg_ex["backoff_factor"])

    frames = []
    skipped, attempted = {}, []
    t0 = time.time()

    for process in processes:
        window = windows[process]
        if window is None:
            lag = cfg_ex["mtr_publication_lag_days"] if process == "MTR" else 0
            motivo = (f"todo el rango cae dentro del rezago de publicacion "
                      f"({lag} dias para {process})")
            skipped[process] = motivo
            report.add(nodo="ALL", tipo_evento=f"{process}_fuera_de_ventana", cantidad=1,
                       resolucion_aplicada="proceso omitido, sin peticiones",
                       detalle=f"{start:%Y-%m-%d} a {end:%Y-%m-%d}: {motivo}")
            logger.warning("[%s] omitido: %s", process, motivo)
            continue

        w_start, w_end = window
        if w_end < end:
            report.add(nodo="ALL", tipo_evento=f"{process}_rango_recortado", cantidad=1,
                       resolucion_aplicada=f"consultado hasta {w_end:%Y-%m-%d}",
                       detalle=f"se pidio hasta {end:%Y-%m-%d}; el resto aun no se publica")
            logger.warning("[%s] rango recortado a %s (rezago de publicacion)",
                           process, w_end.date())

        attempted.append(process)
        proc_chunks = list(date_chunks(w_start, w_end, cfg_ex["max_days_per_chunk"]))
        df_proc = extract_process(session, cfg_ex, process, proc_chunks, logger, report)
        if df_proc.empty:
            logger.error("Proceso %s no devolvio registros", process)
            continue
        checkpoint = Path(cfg["paths"]["raw_dir"]) / f"monclova_{process.lower()}_checkpoint.csv"
        df_proc.to_csv(checkpoint, index=False)
        logger.info("[%s] checkpoint guardado: %s", process, checkpoint)
        frames.append(df_proc)

    if not frames:
        if not attempted:
            # el rango pedido simplemente aun no se publica. 
            report.save()
            print_summary("EXTRACCION SIN DATOS QUE PEDIR", {
                "Rango": f"{start:%Y-%m-%d} a {end:%Y-%m-%d}",
                **{f"Omitido {p}": m for p, m in skipped.items()},
                "Accion": "ninguna; el rango aun no se publica",
            })
            return
        raise PipelineError(
            f"Los procesos {attempted} tenian ventana consultable pero no devolvieron "
            f"datos. Revisar conectividad, los parametros y el estado del SW-PML.")

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out_path, index=False)

    coverage = check_coverage(df, cfg_ex, start, end, report)
    report_path = report.save()

    save_metrics(cfg, "extract", {
        "rango": f"{start:%Y-%m-%d}/{end:%Y-%m-%d}",
        "procesos": processes,
        "filas_totales": len(df),
        "filas_por_proceso": df["process"].value_counts().to_dict(),
        "nodos": df["node_id"].nunique(),
        "duracion_segundos": round(time.time() - t0, 1),
        **coverage,
    }, corrida)

    print_summary("EXTRACCION COMPLETADA", {
        "Rango": f"{start:%Y-%m-%d} a {end:%Y-%m-%d}",
        "Filas": f"{len(df):,}",
        "Procesos": ", ".join(f"{k}={v:,}" for k, v in df['process'].value_counts().items()),
        "Nodos": df["node_id"].nunique(),
        "Duracion": f"{time.time() - t0:.0f}s",
        "Salida": out_path,
        **{f"Omitido {p}": m for p, m in skipped.items()},
        "Eventos de calidad": report.n_events if report.n_events else "ninguno",
        "Reporte": report_path or "-",
    })


if __name__ == "__main__":
    main()
