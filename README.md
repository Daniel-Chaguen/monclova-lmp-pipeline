# Monclova LMP — Pipeline de analítica de precios

Pipeline reproducible para los **Precios Marginales Locales** de los 9 nodos de la zona de carga
Monclova (Sistema Interconectado Nacional, CENACE), sobre 18 meses de datos horarios:

- **Pronóstico** del MDA (Mercado del Día en Adelanto), horizontes de 1 hora y día completo.
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
./run_pipeline.sh mtr                          # ruta rezagada: MTR -> alertas
./run_pipeline.sh full 2024-01-01 2025-06-30   # carga histórica completa
./run_pipeline.sh test                         # tests de invariantes
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
python scripts/run_forecasting.py
python scripts/run_anomaly_detection.py
```

Todos aceptan `--config`, `--verbose` y `--help`. Por defecto imprimen un resumen corto;
`--verbose` activa el detalle.

### Si ya tienes el CSV crudo

Colócalo en `data/raw/monclova_lmp_mda_mtr.csv` y salta la extracción. El resto del pipeline tarda
unos 5 minutos, dominados por el ajuste de MSTL en la detección de anomalías.

---

## Resultados

### Pronóstico (MDA)

Evaluación sobre la ventana de test (abril-junio 2025), nodo piloto `06MON-115`:

| Horizonte | XGBoost | Naive estacional | Mejora |
|---|---|---|---|
| 1h adelante | **86.5** | 167.5 | **−48%** |
| día completo | **135.4** | 205.7 | **−34%** |

MAE en $/MWh. El baseline naive estacional (mismo valor de la hora equivalente del día o la semana
anterior) es lo que haría un analista sin modelo. **Ese porcentaje es el resultado**, no el MAE
absoluto: sin referencia, un error de 86 no dice si el modelo aporta algo.

XGBoost se comparó contra LSTM y ARIMAX+Fourier y ganó en ambos horizontes. Detalle en
`notebooks/model_selection/forecasting.ipynb`.

El horizonte de día completo tiene más error porque tiene **estrictamente menos información**: al
pronosticar las 24 horas del día siguiente solo se dispone de datos hasta el cierre del día actual,
así que no hay lags de corto plazo y las medias móviles se sustituyen por agregados diarios
congelados. No es que el modelo sea peor; el problema es más difícil.

### Detección de anomalías (MTR)

Cuatro métodos evaluados contra la etiqueta proxy, sobre el nodo piloto:

| Detector | Precisión | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| **M4 — residual MSTL** | **0.833** | 0.246 | 0.380 | **0.531** |
| M1 — residual robusto | 0.520 | 0.295 | 0.377 | 0.423 |
| M2 — Isolation Forest | 0.469 | 0.197 | 0.277 | 0.307 |
| M3 — naive estacional (baseline) | 0.200 | 0.098 | 0.132 | 0.163 |

La línea base aleatoria ronda **0.009**, así que un PR-AUC de 0.531 es unas 57 veces mejor que el
azar. En detección no supervisada evaluada contra una etiqueta construida, valores cercanos a 1
serían motivo de sospecha, no de tranquilidad.

**Producción opera M4**, que gana en precisión y en PR-AUC, la métrica independiente del umbral.
Sobre los 18 meses emite **1,125 alertas**, con una tasa efectiva del 1.01% frente al 1.0%
objetivo, repartidas uniformemente entre los nueve nodos (125 cada uno).

---

## Hallazgos que condicionan la operación

### Los nueve nodos son una sola señal de precio

Correlación de **1.0000** entre ellos. La explicación es física, no estadística: comparten el
componente de energía y no hay congestión intrazonal, así que solo difieren en pérdidas marginales.

Consecuencia: un solo conjunto de hiperparámetros para los nueve. Tunear por nodo capturaría ruido
de muestreo, no diferencias reales. El detector de anomalías emite exactamente 125 alertas por
nodo, que es la confirmación empírica de lo mismo.

### El mercado cambia de régimen, y eso rompe los umbrales fijos

Los umbrales se calibran al 1% de tasa de alerta sobre el periodo de entrenamiento. En la ventana
de prueba, **los cuatro detectores caen por debajo de la mitad de su tasa objetivo**:

| Detector | train | test |
|---|---|---|
| M1 | 0.94% | 0.53% |
| M2 | 0.99% | 0.39% |
| M3 | 0.94% | 0.46% |
| M4 | 0.94% | **0.28%** |

No es una peculiaridad de un método: es una propiedad del periodo. La dispersión de la divergencia
entre lo esperado y lo realizado cae sustancialmente entre ambas ventanas, y un umbral calibrado en
el régimen volátil queda demasiado alto para el tranquilo.

**Esta es la justificación empírica de la recalibración periódica**, no una buena práctica citada
de un manual. Es la prioridad 1 de los siguientes pasos.

### Los huecos de publicación se concentran en feriados

252 registros (28 horas × 9 nodos, el 0.21% del MDA) con los cuatro componentes de precio en 0.0
simultáneamente. Cinco de las seis fechas afectadas son feriado o domingo: 1 de enero de 2024 y
2025, 25 de diciembre, Domingo de Pascua. No es un precio de mercado, es un hueco de la fuente.

---

## Cómo se define una anomalía sin etiquetas

CENACE no publica un catálogo de horas anómalas, así que la referencia hay que construirla, y esa
construcción es una decisión que hay que argumentar.

> Una anomalía de tiempo real no es "un precio alto", es **un precio que el día en adelanto no
> anticipó**.

El MDA es la expectativa publicada del mercado; el MTR es la realización. La divergencia entre
ambos define el evento. Dos propiedades la hacen válida:

**No es circular.** Los detectores se construyen exclusivamente sobre la serie MTR; el MDA nunca
entra como variable de entrada. La etiqueta es información externa al detector.

**Es desplegable.** No es un artificio de evaluación: en producción el spread se puede calcular
siete días después del Día de Operación y sirve como monitoreo continuo del propio sistema.

La construcción tiene cuatro pasos (log-ratio, remoción del perfil horario, escala móvil causal de
30 días, umbral por cuantil) y el catálogo completo de alternativas consideradas está en
`notebooks/model_selection/anomaly_detection.ipynb`.

**Limitación declarada:** es una definición, no una verdad. Un evento que ambos mercados
anticiparon no queda marcado, aunque sea operativamente relevante. Es una elección deliberada: se
detecta lo inesperado, no lo caro.

---

## Validaciones de calidad: dos categorías

**Invariantes duros** (detienen el pipeline con `PipelineError`):

- Placeholders en 0.0 en MTR. Nunca han ocurrido; si ocurren, la fuente cambió de comportamiento.
- Filas no imputadas que coinciden con fechas marcadas como placeholder.
- Fracción de placeholders sin resolver por encima de `max_unresolved_fraction`.

**Eventos de calidad esperados** (se registran en `outputs/reports/` y el pipeline continúa):

- Cada imputación aplicada, con el offset usado.
- Placeholders sin donante: se convierten a **NaN**, no a 0.0. El cero se propaga en silencio a
  lags y medias móviles (una hora contaminada ensucia ~200 filas de features) y en detección de
  anomalías dispara una alerta falsa masiva. NaN hace que esas filas se excluyan solas vía `dropna`.
- Rangos recortados o procesos omitidos por los rezagos de publicación de CENACE.
- Cobertura incompleta: horas recibidas por debajo de las esperadas.

Si todo detuviera el pipeline, se caería por el 0.21% de los datos y el equipo aprendería a ignorar
los errores. Si nada lo detuviera, un cambio en la fuente pasaría meses desapercibido.

Los `assert` que validaban **lógica del código** (ausencia de fuga en rolling, continuidad de la
codificación cíclica, invertibilidad de `signed_log1p`) están en `tests/`: en un pipeline recurrente
correrían idénticos miles de veces sin aportar nada.

```bash
python -m pytest tests/ -v     # 20 tests
```

---

## Configuración

Todo vive en `config/config.yaml`: rutas, nodos, rezagos de publicación, lags, ventanas,
hiperparámetros y umbrales. Los scripts no hardcodean parámetros.

**Los hiperparámetros están congelados y no se re-tunean en producción.** Provienen del tuning con
`RandomizedSearchCV` + `TimeSeriesSplit` documentado en el notebook. Tres razones: el tuning es no
determinista, es lento, y si el modelo se reajustara en cada corrida nadie sabría qué versión está
operando.

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
outputs/models/       modelos serializados (formato nativo XGBoost, no pickle)
outputs/predictions/  pronósticos del modo operativo
```

Todos los artefactos llevan el `run_id` (`YYYYMMDDTHHMMSS`) en el nombre, y cada fila del CSV de
pronósticos referencia el archivo de modelo que la produjo. Si dentro de un año alguien pregunta
por qué un pronóstico decía lo que decía, el modelo exacto está en disco.

`data/` y `outputs/` están en `.gitignore`: se regeneran corriendo el pipeline.

---

## Notebooks

```
notebooks/
├── EDA.ipynb                          exploración: nulos, huecos, ACF/PACF, estacionariedad
└── model_selection/
    ├── forecasting.ipynb              tuning y comparación XGBoost / ARIMAX / LSTM
    └── anomaly_detection.ipynb        4 detectores, etiqueta proxy, evaluación
```

**No forman parte del pipeline recurrente y el orquestador no los ejecuta.** Son donde se comparan
y justifican los modelos: la evaluación completa, el backtesting y el tuning viven ahí.

Si se quiere cambiar de modelo o re-tunear, el trabajo se hace en el notebook y el resultado se
traslada a `config/config.yaml`. Nunca al revés.

---

## Limitaciones y siguientes pasos

| Prioridad | Acción | Por qué |
|---|---|---|
| 1 | Umbral adaptativo con ventana móvil | Los cuatro detectores sub-alertan cuando cambia el régimen; es causal, sin fuga |
| 2 | Extracción incremental | Ver `docs/incremental_extraction_design.md` |
| 3 | Validación con experto sobre ~150 horas | Convierte la etiqueta proxy de supuesto a hipótesis verificada |
| 4 | Incorporar demanda, temperatura y precio del gas | Separaría escasez explicable de anomalía genuina |
| 5 | Pronóstico del spread MTR−MDA | El spread es la señal de riesgo; vale más que pronosticar el MDA |
| 6 | Ajustar MSTL una vez y proyectar a los 9 nodos | Bajaría la corrida de ~5 min a ~30 s; requiere revalidar |

**Otras limitaciones a tener presentes.** Los resultados corresponden a un corte único de tres
meses de verano, así que las cifras son orden de magnitud y no precisión anual. La evaluación se
hace sobre el nodo piloto, justificada por la correlación de 1.0000 pero no verificada nodo por
nodo. Los nueve nodos no son nueve observaciones independientes, así que la dispersión entre ellos
mide redundancia, no incertidumbre. Y el pipeline no incorpora variables exógenas, que explicarían
buena parte tanto del error de pronóstico como de los eventos detectados.

---

## Nota sobre el desarrollo

Este proyecto se desarrolló con asistencia de IA para acelerar la implementación. Las decisiones de
diseño, la validación de resultados y la interpretación son propias.
