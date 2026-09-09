# Monclova LMP — Pipeline de analítica de precios

Pipeline  para los **Precios Marginales Locales** de los 9 nodos de la zona de carga
Monclova (Sistema Interconectado Nacional, CENACE), sobre 18 meses de datos horarios:

- **Pronóstico** del MDA (Mercado del Día en Adelanto), 1 hora y día completo.
- **Detección de anomalías** sobre el MTR (Mercado de Tiempo Real).

236,304 registros extraídos del servicio web público de CENACE.

---

## Instalación

```bash
pip install -r requirements.txt            # pipeline
pip install -r requirements-modeling.txt   # además, para los notebooks
```

El pipeline **no** requiere TensorFlow. LSTM, ARIMAX, Isolation Forest y LOF se evaluaron, se
descartaron, y viven solo en los notebooks de selección de modelo.

---

## Etapas

| # | Etapa | Script | Entrada | Salida |
|---|---|---|---|---|
| 1 | Extracción | `scripts/extract.py` | API SW-PML de CENACE | `data/raw/monclova_lmp_mda_mtr.csv` |
| 2 | Preprocesamiento | `scripts/preprocess.py` | CSV crudo | `data/processed/monclova_lmp_clean.csv` |
| 3 | Features | `scripts/build_features.py` | CSV limpio | `monclova_features_{mda,mtr}.csv` |
| 4a | Pronóstico | `scripts/run_forecasting.py` | features MDA | CSV de predicciones + modelos |
| 4b | Anomalías | `scripts/run_anomaly_detection.py` | features MTR | CSV de alertas + figura |

Las etapas 4a y 4b son independientes entre sí y ambas dependen de la 3.

**Los scripts son el camino operativo.** No calculan métricas ni comparan modelos: implementan el
modelo ya decidido. La evaluación vive en los notebooks (ver más abajo).

---

## Cómo correrlo

### Todo junto

```bash
./run_pipeline.sh daily                        # ruta diaria: MDA -> pronóstico
./run_pipeline.sh mtr                          # ruta retraso: MTR -> alertas
./run_pipeline.sh full 2024-01-01 2025-06-30   # carga histórica completa
```



### Correr cada etapa por separado

```bash
python scripts/extract.py --start 2024-01-01 --end 2025-06-30
python scripts/preprocess.py --verbose
python scripts/build_features.py --test-start 2025-04-01
python scripts/run_forecasting.py
python scripts/run_anomaly_detection.py
```
---

## Resultados

### Pronóstico (MDA)

Evaluación sobre la ventana de test (abril-junio 2025), nodo piloto `06MON-115`:

| Horizonte | XGBoost | Baseline | Mejora |
|---|---|---|---|
| 1h adelante | **86.5** | 167.5 | **−48%** |
| día completo | **135.4** | 205.7 | **−34%** |

El baseline naive estacional (mismo valor de la hora equivalente del día o la semana
anterior).

XGBoost se comparó contra LSTM y ARIMAX+Fourier y ganó en ambos horizontes.


### Detección de anomalías (MTR)

Cuatro métodos evaluados contra la etiqueta diferencia MDA y MTR, sobre el nodo piloto:

| Detector | Precisión | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| **M4 — residual MSTL** | **0.833** | 0.246 | 0.380 | **0.531** |
| M1 — residual robusto | 0.520 | 0.295 | 0.377 | 0.423 |
| M2 — Isolation Forest | 0.469 | 0.197 | 0.277 | 0.307 |
| M3 — naive estacional (baseline) | 0.200 | 0.098 | 0.132 | 0.163 |

---

## Hallazgos que condicionan la operación

### Los nueve nodos son una sola señal de precio

Correlación de **1.0000** entre ellos.

Consecuencia: un solo conjunto de hiperparámetros para los nueve.


---

## Cómo se define una anomalía 



El MDA es la expectativa publicada del mercado; el MTR es la realización. La divergencia entre
ambos define el evento.

**Limitación declarada:** es una definición, no una verdad.

---

## Configuración

Todo vive en `config/config.yaml`: rutas, nodos, rezagos de publicación, lags, ventanas,
hiperparámetros y umbrales. Los scripts no hardcodean parámetros.

**Los hiperparámetros están congelados y no se re-tunean en producción.** Provienen del tuning con
`RandomizedSearchCV` + `TimeSeriesSplit` documentado en el notebook. 

El notebook incluye una celda que **compara los hiperparámetros encontrados contra el config y
avisa si difieren**, con un interruptor para reescribirlos preservando comentarios y formato.

---

## Dónde vive cada salida

```
data/raw/             CSV crudo consolidado y checkpoints por proceso
data/processed/       CSV limpio, features por proceso, alertas de anomalías
outputs/metrics/      métricas por corrida (JSON), una por etapa
outputs/figures/      figuras generadas por el pipeline y los notebooks
outputs/reports/      trazabilidad de calidad de datos, una fila por evento
outputs/models/       modelos s
outputs/predictions/  pronósticos del modo operativo
```

Todos los artefactos llevan el `run_id` (`YYYYMMDDTHHMMSS`) en el nombre, y cada fila del CSV de
pronósticos referencia el archivo de modelo que la produjo.


---

## Notebooks

```
notebooks/
├── EDA.ipynb                          exploración: nulos, huecos, ACF/PACF, estacionariedad
└── model_selection/
    ├── forecasting.ipynb              tuning y comparación XGBoost / ARIMAX / LSTM
    └── anomaly_detection.ipynb        4 detectores, etiqueta diferencia, evaluación
```

**No forman parte del pipeline recurrente y el orquestador no los ejecuta.** Son donde se comparan
y justifican los modelos: la evaluación completa, el backtesting y el tuning viven ahí.

---

### Nota 

Este proyecto se desarrolló con asistencia de IA para acelerar la implementación. Las decisiones de
diseño, la validación de resultados y la interpretación son propias.
