"""
Invariantes de logica del pipeline.


    pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_features import (add_calendar, add_frozen_dayahead, add_lags,
                            add_rolling, signed_log1p)
from common import PipelineError
from preprocess import (build_offsets, check_hard_invariants,
                        find_zero_placeholders, impute_placeholders)


@pytest.fixture
def sample():
    """Dos nodos, dos procesos, 30 dias horarios."""
    ts = pd.date_range("2024-01-01", periods=24 * 30, freq="h")
    frames = []
    rng = np.random.default_rng(0)
    for process in ["MDA", "MTR"]:
        for node in ["N1", "N2"]:
            frames.append(pd.DataFrame({
                "process": process, "node_id": node, "timestamp": ts,
                "pml": rng.uniform(100, 900, len(ts)).round(2),
                "pml_ene": 1.0, "pml_per": 1.0, "pml_cng": 1.0,
            }))
    return pd.concat(frames, ignore_index=True)


class FakeReport:
    def __init__(self):
        self.rows = []

    def add(self, **kw):
        self.rows.append(kw)


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def test_rolling_no_leakage(sample):
    """La media movil de la fila t NO debe incluir el valor de la fila t."""
    df = add_rolling(sample, [24], "pml")
    g = df[(df["process"] == "MDA") & (df["node_id"] == "N1")].reset_index(drop=True)

    pos = 100
    manual = g.loc[pos - 24:pos - 1, "pml"].mean()   # las 24 horas ANTERIORES
    assert abs(g.loc[pos, "pml_roll24_mean"] - manual) < 1e-9

    con_actual = g.loc[pos - 23:pos, "pml"].mean()
    assert abs(g.loc[pos, "pml_roll24_mean"] - con_actual) > 1e-9


def test_lags_no_cruzan_grupos(sample):
    """Un lag nunca debe tomar el valor de otro nodo o de otro proceso."""
    df = add_lags(sample, [1, 24], "pml").sort_values(["process", "node_id", "timestamp"])
    for _, g in df.groupby(["process", "node_id"]):
        g = g.reset_index(drop=True)
        assert g.loc[:23, "pml_lag24"].isna().all()      # arranque sin historia
        assert np.allclose(g.loc[24:, "pml_lag24"].values, g.loc[:len(g) - 25, "pml"].values)


def test_codificacion_ciclica_continua(sample):
    """23h y 0h deben quedar contiguas en el espacio (sin, cos), no en extremos."""
    df = add_calendar(sample)
    p = {h: (df.loc[df["hora_dia"] == h, "hora_sin"].iloc[0],
             df.loc[df["hora_dia"] == h, "hora_cos"].iloc[0]) for h in (0, 12, 23)}
    d_23_0 = np.hypot(p[23][0] - p[0][0], p[23][1] - p[0][1])
    d_0_12 = np.hypot(p[0][0] - p[12][0], p[0][1] - p[12][1])
    assert d_23_0 < d_0_12


def test_signed_log1p_invertible():
    """Debe tolerar negativos y ser invertible dentro de tolerancia numerica."""
    x = np.array([-5000.0, -45.3, -1e-6, 0.0, 1e-6, 45.3, 9000.0])
    y = signed_log1p(x)
    back = np.sign(y) * (np.expm1(np.abs(y)))
    assert np.max(np.abs(back - x)) < 1e-6
    assert np.all(np.sign(y) == np.sign(x))


def test_frozen_dayahead_solo_usa_pasado(sample):
    """
    Las features congeladas de un dia D deben provenir de dias <= D-1.
    Si el valor de D coincidiera con la media del propio dia D, habria fuga.
    """
    df = add_frozen_dayahead(add_calendar(sample), "pml")
    g = df[(df["process"] == "MDA") & (df["node_id"] == "N1")]
    dias = sorted(g["fecha_dt"].unique())

    d_prev, d_cur = dias[5], dias[6]
    media_prev = g.loc[g["fecha_dt"] == d_prev, "pml"].mean()
    media_cur = g.loc[g["fecha_dt"] == d_cur, "pml"].mean()
    frozen = g.loc[g["fecha_dt"] == d_cur, "roll24_frozen_mean"].iloc[0]

    assert abs(frozen - media_prev) < 1e-9
    assert abs(frozen - media_cur) > 1e-9


# --------------------------------------------------------------------------- #
# Preprocesamiento
# --------------------------------------------------------------------------- #
def test_offsets_alternan_signo():
    assert build_offsets(168, 3) == [-168, 168, -336, 336, -504, 504]


def test_imputacion_resuelve_y_marca(sample):
    """Un placeholder con donante disponible debe quedar imputado y marcado."""
    df = sample.copy()
    victim = (df["process"] == "MDA") & (df["node_id"] == "N1") & \
             (df["timestamp"] == pd.Timestamp("2024-01-15 10:00"))
    df.loc[victim, ["pml", "pml_ene", "pml_per", "pml_cng"]] = 0.0

    donor = df.loc[(df["process"] == "MDA") & (df["node_id"] == "N1") &
                   (df["timestamp"] == pd.Timestamp("2024-01-08 10:00")), "pml"].iloc[0]

    out, stats = impute_placeholders(df, "MDA", 168, 3, FakeReport())
    fixed = out.loc[victim]

    assert stats["n_imputed"] == 1 and stats["n_unresolved"] == 0
    assert abs(fixed["pml"].iloc[0] - donor) < 1e-9
    assert bool(fixed["mda_imputed"].iloc[0]) is True


def test_sin_donante_va_a_nan_no_a_cero(sample):
    """
    Sin ningun offset semanal disponible, la fila debe quedar en NaN.
    Dejarla en 0.0 la propagaria en silencio a lags y rolling.
    """
    ts = pd.date_range("2024-01-01", periods=24 * 3, freq="h")   # solo 3 dias
    df = pd.DataFrame({"process": "MDA", "node_id": "N1", "timestamp": ts,
                       "pml": 500.0, "pml_ene": 1.0, "pml_per": 1.0, "pml_cng": 1.0})
    df.loc[df["timestamp"] == ts[10], ["pml", "pml_ene", "pml_per", "pml_cng"]] = 0.0

    out, stats = impute_placeholders(df, "MDA", 168, 3, FakeReport())
    val = out.loc[out["timestamp"] == ts[10], "pml"].iloc[0]

    assert stats["n_unresolved"] == 1
    assert pd.isna(val)
    assert val != 0


def test_placeholder_en_mtr_trona(sample):
    """MTR con placeholders nunca ha ocurrido: debe detener el pipeline."""
    df = sample.copy()
    df["mda_imputed"] = False
    df.loc[(df["process"] == "MTR") & (df["node_id"] == "N1"),
           ["pml", "pml_ene", "pml_per", "pml_cng"]] = 0.0

    cfg_pre = {"placeholder_process": "MDA", "max_unresolved_fraction": 0.01}
    with pytest.raises(PipelineError, match="MTR"):
        check_hard_invariants(df, pd.DataFrame(columns=["node_id", "timestamp"]),
                              {"n_unresolved": 0}, cfg_pre)


def test_demasiados_sin_resolver_trona(sample):
    """Por encima del umbral no es un caso aislado sino un hueco prolongado."""
    df = sample.copy()
    df["mda_imputed"] = False
    cfg_pre = {"placeholder_process": "MDA", "max_unresolved_fraction": 0.01}
    n_mda = int((df["process"] == "MDA").sum())

    with pytest.raises(PipelineError, match="sin resolver"):
        check_hard_invariants(df, pd.DataFrame(columns=["node_id", "timestamp"]),
                              {"n_unresolved": int(n_mda * 0.05),
                               "unresolved_sample": []}, cfg_pre)


def test_deteccion_exige_las_cuatro_columnas_en_cero(sample):
    """Un pml en 0 con componentes distintos de 0 es un precio real, no placeholder."""
    df = sample.copy()
    df.loc[df.index[0], "pml"] = 0.0          # solo pml
    assert len(find_zero_placeholders(df, "MDA")) == 0

    df.loc[df.index[0], ["pml_ene", "pml_per", "pml_cng"]] = 0.0
    assert len(find_zero_placeholders(df, "MDA")) == 1


# --------------------------------------------------------------------------- #
# Ventanas de publicacion de la fuente
# --------------------------------------------------------------------------- #
from datetime import datetime

from extract import available_window

CFG_EX = {"mtr_publication_lag_days": 7, "mda_max_future_days": 1}
HOY = datetime(2026, 9, 6)


def _w(process, a, b):
    return available_window(process, datetime.strptime(a, "%Y-%m-%d"),
                            datetime.strptime(b, "%Y-%m-%d"), CFG_EX, HOY)


def test_mtr_dentro_del_rezago_se_omite():
    """Pedir MTR de los ultimos 7 dias devuelve HTTP 400: no hay que pedirlo."""
    assert _w("MTR", "2026-09-03", "2026-09-06") is None


def test_mtr_parcial_se_recorta():
    start, end = _w("MTR", "2026-08-01", "2026-09-06")
    assert end == datetime(2026, 8, 30)      # hoy - 7 dias
    assert start == datetime(2026, 8, 1)


def test_mtr_historico_intacto():
    assert _w("MTR", "2024-01-01", "2025-06-30") == (datetime(2024, 1, 1),
                                                     datetime(2025, 6, 30))


def test_mda_admite_dia_siguiente():
    """El MDA se publica un dia antes del Dia de Operacion."""
    assert _w("MDA", "2026-09-01", "2026-09-20")[1] == datetime(2026, 9, 7)


# --------------------------------------------------------------------------- #
# Modo predict: el frame futuro no puede inventar informacion
# --------------------------------------------------------------------------- #
# Escala robusta movil del detector de anomalias
# --------------------------------------------------------------------------- #
from run_anomaly_detection import causal_rolling_mad, causal_scale


# --------------------------------------------------------------------------- #
from run_forecasting import build_future_frame


def test_frame_futuro_usa_solo_pasado():
    """
    Los features de una hora futura deben provenir exclusivamente de horas ya
    observadas. Si lag24 de la hora futura no coincide con el valor real de 24h
    antes, el modo predict estaria alimentando al modelo con basura.
    """
    ts = pd.date_range("2024-01-01", periods=24 * 40, freq="h")
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"process": "MDA", "node_id": "N1", "timestamp": ts,
                       "pml": rng.uniform(100, 900, len(ts)).round(2),
                       "mda_imputed": False})

    cfg = {"features": {"lags": [1, 24, 168], "rolling_windows": [24],
                        "target_col": "pml"}}
    future, last_ts = build_future_frame(df, cfg)

    assert last_ts == ts[-1]
    assert future["pml"].isna().all()           # el futuro no tiene valor observado

    primera = future.sort_values("timestamp").iloc[0]
    assert primera["timestamp"] == last_ts + pd.Timedelta(hours=1)
    assert abs(primera["pml_lag1"] - df["pml"].iloc[-1]) < 1e-9
    assert abs(primera["pml_lag24"] - df.loc[df["timestamp"] ==
               primera["timestamp"] - pd.Timedelta(hours=24), "pml"].iloc[0]) < 1e-9

    # La media movil de la primera hora futura son las 24 horas observadas previas
    assert abs(primera["pml_roll24_mean"] - df["pml"].iloc[-24:].mean()) < 1e-9


def test_frame_futuro_cubre_dia_siguiente():
    ts = pd.date_range("2024-01-01", periods=24 * 40, freq="h")
    df = pd.DataFrame({"process": "MDA", "node_id": "N1", "timestamp": ts,
                       "pml": 500.0, "mda_imputed": False})
    cfg = {"features": {"lags": [24], "rolling_windows": [24], "target_col": "pml"}}
    future, last_ts = build_future_frame(df, cfg)

    dia_siguiente = (last_ts.normalize() + pd.Timedelta(days=1)).date()
    assert (future["timestamp"].dt.date == dia_siguiente).sum() == 24


def test_mad_movil_es_causal():
    """
    La MAD de la fila t debe calcularse con [t-window, t-1], nunca incluyendo t.
    Se verifica contra el calculo manual sobre esa ventana exacta.
    """
    rng = np.random.default_rng(3)
    x = rng.uniform(100, 900, 300)
    w = 48
    mad = causal_rolling_mad(x, window=w, min_periods=24)

    t = 200
    ventana = x[t - w:t]                      # estrictamente anterior a t
    esperado = np.median(np.abs(ventana - np.median(ventana)))
    assert abs(mad[t] - esperado) < 1e-9

    con_t = x[t - w + 1:t + 1]                # la version CON fuga
    fuga = np.median(np.abs(con_t - np.median(con_t)))
    assert abs(mad[t] - fuga) > 1e-12


def test_mad_resiste_un_outlier():
    """
    Un pico aislado NO debe inflar la escala. Es la razon de usar MAD y no
    desviacion estandar: con std, el propio outlier agranda el denominador y
    termina enmascarandose a si mismo.
    """
    x = np.full(300, 100.0)
    x[150] = 100000.0
    w = 48

    t = 160                                   # ventana [112:160], contiene el pico
    ventana = x[t - w:t]
    assert (ventana > 1000).any()

    mad = causal_rolling_mad(x, window=w, min_periods=24)
    assert mad[t] == 0.0                      # la mediana ignora el pico
    assert ventana.std() > 10000              # la std si se dispara


def test_escala_neutraliza_ceros():
    """Una MAD de cero produciria division por cero: debe devolver NaN."""
    x = np.full(300, 500.0)
    s = causal_scale(x, window=48, min_periods=24)
    assert np.isnan(s[100])
