# Monclova LMP — Pipeline de analítica de precios

Pipeline reproducible para los **Precios Marginales Locales** de los 9 nodos de la zona de carga
Monclova (Sistema Interconectado Nacional, CENACE), con dos tracks en producción:

- **Forecasting** del MDA (Mercado del Día en Adelanto), horizontes de 1 hora y day-ahead.
- **Detección de anomalías** sobre el MTR (Mercado de Tiempo Real).

---

## Instalación

```bash
pip install -r requirements.txt          # pipeline de producción
pip install -r requirements-modeling.txt # además, para correr los notebooks
```

El pipeline de producción **no** requiere TensorFlow ni scikit-learn. LSTM, ARIMAX, Isolation Forest
y LOF se evaluaron y descartaron; viven solo en los notebooks de selección de modelo.

---

## Etapas

| # | Etapa | Script | Entrada | Salida |
|---|---|---|---|---|
| 1 | Extracción | `scripts/extract.py` | API SW-PML de CENACE | `data/raw/monclova_lmp_mda_mtr.csv` |
| 2 | Preprocesamiento | `scripts/preprocess.py` | CSV crudo | `data/processed/monclova_lmp_clean.csv` |
| 3 | Features | `scripts/build_features.py` | CSV limpio | `monclova_features_{mda,mtr}.csv` |
| 4a | Forecasting | `scripts/run_forecasting.py` | features MDA | métricas, figuras, pronósticos |
| 4b | Anomalías | `scripts/run_anomaly_detection.py` | features MTR + MDA | métricas, figuras, alertas |

Las etapas 4a y 4b son independientes entre sí y ambas dependen de la 3.

---

## Cómo correrlo

### Todo junto

```bash
./run_pipeline.sh daily      # ruta diaria: MDA -> pronóstico
./run_pipeline.sh mtr        # ruta rezagada: MTR -> alertas
./run_pipeline.sh backtest   # reevaluación de ambos modelos
./run_pipeline.sh full 2024-01-01 2025-06-30   # carga histórica completa
./run_pipeline.sh test       # tests de invariantes
```

**Por qué dos rutas y no una.** Los dos tracks no pueden correr en la misma cadencia. El MDA se
publica un día antes del Día de Operación, así que el pronóstico se actualiza a diario. El MTR
(Expost) se publica hasta **siete días después**, así que la detección de anomalías opera
necesariamente con ese rezago. Forzarlas al mismo horario haría que la ruta de anomalías pidiera
datos que aún no existen.

### Cada etapa por separado

```bash
python scripts/extract.py --start 2025-07-01 --end 2025-07-15 --processes MDA
python scripts/extract.py --start 2025-07-01 --end 2025-07-15 --dry-run   # sin llamar a la API
python scripts/preprocess.py --verbose
python scripts/build_features.py --test-start 2025-04-01
python scripts/run_forecasting.py --mode backtest
python scripts/run_forecasting.py --mode predict
python scripts/run_anomaly_detection.py --mode backtest
python scripts/run_anomaly_detection.py --mode predict --lookback-days 30
```

Todos aceptan `--config`, `--verbose` y `--help`. Por defecto imprimen solo un resumen corto;
`--verbose` activa el detalle.

---

## Resultados

### Forecasting (MDA)

Evaluación sobre la ventana de test (abril-junio 2025), promedio de los 9 nodos:

| Horizonte | XGBoost | Naive estacional | Mejora |
|---|---|---|---|
| 1h adelante | **86.7** | 167.8 | **48%** |
| day-ahead | **132.8** | 206.1 | **36%** |

MAE en $/MWh. El baseline naive estacional (mismo valor de la hora anterior equivalente) es el
control que responde si el modelo aporta valor sobre una regla trivial. La desviación entre nodos es
de 0.26 y 0.55 respectivamente, coherente con que los 9 nodos son la misma señal de precio.

### Detección de anomalías (MTR)

| Detector | Recall de evento | Precisión de evento | F1 |
|---|---|---|---|
| M4 (residual MSTL) | 0.306 | **0.833** | 0.380 |
| M1 (residual robusto) | **0.368** | 0.750 | 0.377 |

Contra una línea base aleatoria de 0.0093. La severidad sale del consenso entre ambos: la tasa de
acierto pasa de **0.6%** sin detectores a **39.7%** con uno y **80.0%** con los dos.

### Validación de estabilidad

Además del corte único, el pipeline soporta backtesting **rolling-origin** sobre cuatro épocas del
año (`--rolling 4`). XGBoost supera al baseline estacional en los cuatro pliegues y en ambos
horizontes; la magnitud del error varía de forma estacional, siendo el verano la ventana más
difícil. Los resultados detallados quedan en `outputs/metrics/`.

## Los dos modos de los scripts de modelado

| | `--mode backtest` | `--mode predict` |
|---|---|---|
| Entrenamiento | solo `split == train` | **todo** el histórico disponible |
| Evaluación | contra el test conocido | ninguna: la verdad aún no existe |
| Salida | métricas + figuras | CSV de pronósticos / alertas |
| Cuándo | al cambiar el pipeline, o periódicamente | en cada corrida operativa |

La diferencia no es cosmética. En backtest hay que **excluir** del entrenamiento todo lo posterior
al corte, o las métricas mienten por fuga. En predict hay que **incluirlo todo**, o se desperdicia
información reciente. Misma lógica de features, frontera temporal invertida.

---

## Dónde vive cada salida

```
data/raw/          CSV crudo consolidado y checkpoints por proceso
data/processed/    CSV limpio, features por proceso, anomalías del backtest
outputs/metrics/   métricas por corrida (JSON y CSV), una por etapa
outputs/figures/   solo las figuras validadas para presentar
outputs/reports/   trazabilidad de calidad de datos, una fila por evento
outputs/models/    modelos serializados (formato nativo XGBoost, no pickle)
outputs/predictions/  pronósticos y alertas del modo predict
```

Todos los artefactos llevan el `run_id` (`YYYYMMDDTHHMMSS`) en el nombre, y cada fila del CSV de
pronósticos referencia el archivo de modelo que la produjo. Si dentro de un año alguien pregunta
por qué el pronóstico de una hora decía lo que decía, el modelo exacto está en disco.

---

## Configuración

Todo vive en `config/config.yaml`: rutas, nodos, rezagos de publicación, lags, ventanas,
hiperparámetros y umbrales. Los scripts no hardcodean parámetros.

**Los hiperparámetros están congelados y no se re-tunean en producción.** Provienen del tuning con
`RandomizedSearchCV` + `TimeSeriesSplit` sobre el nodo piloto `06MON-115`. Se comparten entre los 9
nodos porque el MAE entre nodos fue 86.71 ± 0.21 (h1) y 133.89 ± 0.82 (day-ahead): con esa
dispersión, el tuning individual capturaba ruido de muestreo, no diferencias reales.

---

## Validaciones de calidad: dos categorías

**Invariantes duros** (detienen el pipeline con `PipelineError`):

- Placeholders en 0.0 en MTR. Nunca han ocurrido; si ocurren, la fuente cambió de comportamiento.
- Filas no imputadas que coinciden con fechas marcadas como placeholder.
- Fracción de placeholders sin resolver por encima de `max_unresolved_fraction`.

**Eventos de calidad esperados** (se registran en `outputs/reports/` y el pipeline continúa):

- Cada imputación de placeholder, con el offset usado.
- Placeholders sin donante disponible: se convierten a **NaN**, no a 0.0. El cero se propaga en
  silencio a lags y rolling (una hora contaminada ensucia ~200 filas de features) y en detección de
  anomalías dispara una alerta falsa masiva. NaN hace que esas filas se excluyan solas vía `dropna`.
- Rangos recortados o procesos omitidos por los rezagos de publicación de CENACE.
- Cobertura incompleta: horas recibidas por debajo de las esperadas.

Los `assert` que validaban **lógica** del código (ausencia de fuga en rolling, continuidad de la
codificación cíclica, invertibilidad de `signed_log1p`) se movieron a `tests/`: en un pipeline
recurrente correrían idénticos miles de veces sin aportar nada.

```bash
python -m pytest tests/ -v     # 24 tests
```

---

## Notebooks

```
notebooks/
├── 2_EDA.ipynb                    exploración: nulos, huecos, ACF/PACF, estacionariedad
└── model_selection/
    ├── forecasting_pilot.ipynb    tuning y comparación XGBoost / ARIMAX / LSTM (nodo piloto)
    ├── forecasting_all_nodes.ipynb  generalización a los 9 nodos
    └── anomaly_detection.ipynb    5 detectores, 6 capas de evaluación, tabla de decisión
```

**Los notebooks no forman parte del pipeline recurrente y el orquestador no los ejecuta.** Su valor
es documental: son donde se compararon y justificaron los modelos. Los scripts de producción
implementan únicamente el ganador de cada track, con los parámetros que esos notebooks fijaron.

Si se quiere cambiar de modelo o re-tunear, el trabajo se hace en el notebook y el resultado se
traslada a `config/config.yaml`. Nunca al revés.

---

## Modelos en producción

**Forecasting: XGBoost**, un modelo por nodo y horizonte (18 artefactos por corrida). Ganó por
margen amplio: MAE 86.7 contra 104.3 de LSTM y 146.9 de ARIMAX+Fourier. Se acompaña de un baseline
naive estacional como control continuo de que el modelo aporta valor sobre una regla trivial.

**Anomalías: dos detectores complementarios** (métricas en la sección de resultados).

Se operan juntos a propósito. M4 gana en el agregado pero M1 encuentra más eventos, y en monitoreo
de precios el costo de un episodio no detectado supera al de revisar una alerta de más. Además M4
es el peor en la métrica sin etiquetas (`EM_proxy` 0.646 contra 0.821 de M1), lo que abre la
posibilidad de que parte de su ventaja venga de estar alineado con la definición de la etiqueta
proxy y no de ser mejor detector. Correr ambos cubre ese riesgo sin costo relevante.

La severidad sale del consenso: **1 detector = informativo, 2 = revisión del analista**.

Ninguno de los dos detectores usa el MDA como feature. El MDA solo entra en la etiqueta de
evaluación, y esa separación es lo que evita la circularidad.

---

## Hallazgos que condicionan la operación

**Los 9 nodos son una sola señal de precio.** Correlación mínima entre nodos de 1.0000, y la
desviación estándar del F1 entre nodos es 0.005. El spread entre nodos se explica por pérdidas
marginales; la congestión intrazonal es prácticamente nula.

**El riesgo de tiempo real es unidireccional.** En la ventana de prueba, 174 de 183 sorpresas
etiquetadas fueron al alza. Las sorpresas a la baja vienen de sobreoferta, un proceso suave que
rara vez produce colas extremas.

**Hay cambio de régimen entre periodos, y rompe los umbrales fijos.** La volatilidad de la
divergencia MDA-MTR cae ~43% entre el periodo de entrenamiento y el de prueba. Consecuencia medida:
en modo predict los detectores alertan a menos de la mitad de su tasa objetivo. Esta es la
justificación empírica de la recalibración periódica, no una buena práctica genérica.

---

## Limitaciones conocidas y siguientes pasos

| Prioridad | Acción | Por qué |
|---|---|---|
| 1 | Umbral adaptativo con ventana móvil de 30 días | Resuelve el sub-alertado documentado arriba; es causal, sin fuga |
| 2 | Extracción incremental | Ver `docs/incremental_extraction_design.md` |
| 3 | Ajustar MSTL una sola vez y proyectar a los 9 nodos | Bajaría la corrida de anomalías de ~270s a ~30s. Cambia la metodología validada, así que requiere revalidar |
| 4 | Validación con experto sobre 150 horas estratificadas | Convierte la etiqueta proxy de supuesto a hipótesis verificada |
| 5 | Incorporar demanda, temperatura y precio del gas | Separaría escasez explicable de anomalía genuina |
| 6 | Forecasting del spread MTR−MDA | El spread es la señal de riesgo; pronosticarlo vale más que pronosticar el MDA |

Otras limitaciones a tener presentes: los resultados principales corresponden a un corte único de
tres meses, y la validación rolling-origin muestra que la magnitud del error varía de forma
estacional, por lo que las cifras deben leerse como orden de magnitud y no como precisión anual; los 9 nodos no son 9 observaciones independientes, así que la σ entre nodos es informativa
de la redundancia y no un intervalo de confianza; y la etiqueta proxy es una definición defendible,
no una verdad, que por construcción no marca eventos que ambos mercados anticiparon.
