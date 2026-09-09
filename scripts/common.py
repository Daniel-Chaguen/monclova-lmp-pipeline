"""
Funciones compartidas por todas las etapas del pipeline.

"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# Raiz del repo: este archivo vive en <repo>/scripts/
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"


class PipelineError(Exception):
    """
    Los eventos de calidad esperados (imputaciones, huecos aislados) NO usan
    esta excepcion: se registran en el reporte de calidad y el pipeline sigue.
    """


# --------------------------------------------------------------------------- #
# Configuracion
# --------------------------------------------------------------------------- #
def load_config(path: str | Path | None = None) -> dict:
    """Carga el YAML de configuracion y resuelve las rutas contra la raiz del repo."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise PipelineError(f"No se encontro el archivo de configuracion: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    cfg["paths"] = {k: str(REPO_ROOT / v) for k, v in cfg["paths"].items()}
    return cfg


def ensure_dirs(cfg: dict) -> None:
    """Crea los directorios de salida si no existen."""
    for key, value in cfg["paths"].items():
        target = Path(value)
        if key.endswith("_dir"):
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Por defecto solo WARNING y errores llegan a consola; el resumen de cada etapa
    se imprime aparte con print(). Con --verbose se activa el detalle INFO.
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
    return logging.getLogger("pipeline")


def run_id() -> str:
    """Identificador de corrida, usado para nombrar reportes y artefactos."""
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def print_summary(title: str, rows: dict) -> None:
    """Resumen corto y legible en consola. Es la unica salida estandar por defecto."""
    width = max((len(k) for k in rows), default=0)
    print(f"\n{title}")
    print("-" * max(len(title), 40))
    for key, value in rows.items():
        print(f"  {key:<{width}}  {value}")
    print()


# --------------------------------------------------------------------------- #
# Reporte de trazabilidad de calidad de datos
# --------------------------------------------------------------------------- #
class DataQualityReport:
    """
    Acumula eventos de calidad  y los persiste en CSV.

    """

    COLUMNS = ["timestamp_corrida", "etapa", "nodo", "tipo_evento",
               "cantidad", "resolucion_aplicada", "detalle"]

    def __init__(self, etapa: str, cfg: dict, corrida: str | None = None):
        self.etapa = etapa
        self.cfg = cfg
        self.corrida = corrida or run_id()
        self._rows: list[dict] = []

    def add(self, nodo: str, tipo_evento: str, cantidad: int,
            resolucion_aplicada: str, detalle: str = "") -> None:
        self._rows.append({
            "timestamp_corrida": self.corrida,
            "etapa": self.etapa,
            "nodo": nodo,
            "tipo_evento": tipo_evento,
            "cantidad": int(cantidad),
            "resolucion_aplicada": resolucion_aplicada,
            "detalle": detalle,
        })

    @property
    def n_events(self) -> int:
        return len(self._rows)

    @property
    def total_rows_affected(self) -> int:
        return int(sum(r["cantidad"] for r in self._rows))

    def save(self) -> Path | None:
        """Escribe el reporte. Si no hubo eventos, no genera archivo vacio."""
        if not self._rows:
            return None
        out = Path(self.cfg["paths"]["reports_dir"]) / \
            f"data_quality_report_{self.etapa}_{self.corrida}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self._rows, columns=self.COLUMNS).to_csv(out, index=False)
        return out


# --------------------------------------------------------------------------- #
# Metricas por corrida
# --------------------------------------------------------------------------- #
def save_metrics(cfg: dict, etapa: str, metrics: dict, corrida: str | None = None) -> Path:
    """Persiste las metricas de una etapa en JSON, una por corrida."""
    corrida = corrida or run_id()
    out = Path(cfg["paths"]["metrics_dir"]) / f"{etapa}_{corrida}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"etapa": etapa, "timestamp_corrida": corrida, **metrics}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    return out
