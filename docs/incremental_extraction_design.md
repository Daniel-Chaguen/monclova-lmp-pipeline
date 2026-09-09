# Diseño de extracción incremental

**Estado: diseño, no implementado.** Hoy `extract.py` recibe el rango por línea de comandos
(`--start`, `--end`) y siempre extrae la ventana completa que se le pide. Este documento describe
cómo se implementaría el modo incremental cuando haga falta, con las decisiones ya razonadas para
que quien lo escriba no tenga que redescubrirlas.

---

## 1. El problema: el MTR se republica

La complicación no es detectar dónde quedó la última extracción. Es que **una fecha ya extraída
puede cambiar después**.

El manual técnico del SW-PML lo dice: el MDA se publica un día antes del Día de Operación y el MTR
(Expost) hasta siete días después. Un job que corre el día D y extrae "todo lo nuevo desde D-1"
nunca obtendría el MTR de D-7, porque para cuando ese dato se publica, la ventana de extracción ya
pasó de largo.

Cualquier diseño incremental que solo avance hacia adelante pierde datos de MTR de forma
sistemática y silenciosa. Es el modo de falla principal a evitar.

---

## 2. Cómo detectar "hasta dónde ya extraje"

Dos opciones, con distinta relación entre simplicidad y robustez.

### Opción A — Leer el máximo timestamp del CSV consolidado

```python
df = pd.read_csv(RAW_CONSOLIDATED, usecols=["process", "timestamp"], parse_dates=["timestamp"])
ultimo = df.groupby("process")["timestamp"].max()
```

**A favor:** no introduce estado nuevo. El dato es la fuente de verdad y no puede desincronizarse
de sí mismo. Si alguien borra el CSV y vuelve a extraer, todo sigue coherente.

**En contra:** hay que leer un archivo que crece sin límite. Con 236 mil filas cuesta menos de un
segundo leyendo solo dos columnas; con cinco años de datos y más nodos empieza a pesar. Y no
distingue entre "no extraje ese día" y "lo extraje y venía vacío".

### Opción B — Archivo de estado `data/raw/last_run.json`

```json
{
  "MDA": {"ultima_fecha_extraida": "2025-07-14", "corrida": "20250715T060000", "filas": 216},
  "MTR": {"ultima_fecha_extraida": "2025-07-07", "corrida": "20250715T060000", "filas": 216},
  "ultima_corrida_exitosa": "20250715T060000"
}
```

**A favor:** lectura instantánea, y permite registrar cosas que el CSV no sabe: cuándo fue la
última corrida exitosa, qué rangos se intentaron y vinieron vacíos, cuántos reintentos hubo.

**En contra:** es estado duplicado, y el estado duplicado se desincroniza. Si el job escribe el
JSON pero falla al guardar el CSV, el pipeline cree tener datos que no tiene y nunca los vuelve a
pedir. Ese hueco es invisible hasta que alguien nota que faltan días.

### Recomendación

**Opción A como fuente de verdad, Opción B como caché y bitácora.** Al arrancar, se lee el JSON;
se valida contra el máximo timestamp del CSV; si discrepan, gana el CSV y se reescribe el JSON con
un warning en el reporte de calidad. Así se tiene la velocidad de B con la seguridad de A, y la
discrepancia queda registrada en vez de pasar desapercibida.

---

## 3. La ventana de traslape

Es la pieza central del diseño. La regla, por proceso:

| Proceso | Desde | Hasta | Por qué |
|---|---|---|---|
| MDA | `ultima_fecha - 2 días` | `hoy + 1 día` | Traslape corto: el MDA rara vez se corrige. El +1 aprovecha que ya está publicado el día siguiente. |
| MTR | `min(ultima_fecha, hoy - 10 días)` | `hoy - 7 días` | Diez días de traslape sobre los siete de rezago, con tres de margen. |

El margen de tres días no es paranoia. Cubre que el job no haya corrido un fin de semana, que
CENACE se retrase en publicar, o que una corrida haya fallado y nadie lo haya notado hasta el
lunes. Es barato: son tres días × 9 nodos, unas dos peticiones adicionales.

**Consecuencia obligatoria:** si se re-extraen días ya presentes, la escritura **no puede ser
append**. Tiene que ser un upsert con clave `(process, node_id, timestamp)`, quedándose con la
versión más reciente:

```python
combinado = pd.concat([historico, nuevo], ignore_index=True)
combinado = (combinado
             .sort_values("fecha_extraccion")
             .drop_duplicates(["process", "node_id", "timestamp"], keep="last")
             .sort_values(["process", "node_id", "timestamp"]))
```

Esto exige añadir una columna `fecha_extraccion` al CSV crudo, que hoy no existe. Sin ella no hay
forma de saber cuál de dos versiones de la misma hora es la buena.

**Vale la pena reportar cuántas filas cambiaron de valor** en el traslape, no solo cuántas se
reescribieron. Si el MTR de un día se corrige sustancialmente después de publicado, eso es
información operativa relevante y hoy nadie la vería.

---

## 4. Reprocesamiento aguas abajo

Un punto que se pasa por alto: extraer incrementalmente **no** significa preprocesar
incrementalmente.

`preprocess.py` ya carga el CSV consolidado completo, y debe seguir haciéndolo: la imputación de
placeholders busca donantes en `t-168h`, y con una ventana de diez días esa búsqueda fallaría
sistemáticamente. Lo mismo con `build_features.py`, cuyos lags de 168 horas y ventanas móviles
necesitan historia.

Con 236 mil filas ambas etapas tardan segundos, así que reprocesar todo cada vez es la opción
correcta por mucho tiempo. El día que deje de serlo, la solución no es procesar solo lo nuevo sino
procesar una ventana deslizante de, digamos, los últimos 90 días, que garantiza historia suficiente
para cualquier lag del pipeline.

---

## 5. Manejo de errores en un cron real

**Fallo parcial.** Si un chunk falla tras agotar reintentos, hoy el script lo registra y continúa,
guardando lo que sí obtuvo. En modo incremental eso es peligroso: el estado avanzaría dejando un
hueco permanente. La regla debe ser **no avanzar el estado si hubo chunks fallidos**; el traslape
de la siguiente corrida los recupera solo.

**Corridas solapadas.** Si una corrida tarda más de lo previsto y arranca la siguiente, dos
procesos escriben el mismo CSV. Un lock de archivo lo resuelve:

```python
import fcntl
with open(LOCK_PATH, "w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)   # falla rápido si ya hay una corrida
```

**Días sin publicación.** Ya está resuelto: `available_window()` recorta el rango antes de pedir, y
si no queda nada consultable el script termina con código 0 y lo registra. Un cron diario cae en
ese caso de forma rutinaria y no debe alertar.

**Reintentos.** La sesión ya reintenta con backoff exponencial sobre 429 y 5xx. Lo que falta para
un cron es un reintento **de la corrida completa**: si falla, volver a intentar en 30 minutos, y
alertar solo tras el segundo fallo consecutivo. Eso se resuelve en el orquestador, no en el script.

**Cuándo alertar a un humano.** Distinguir tres niveles evita que el equipo aprenda a ignorar los
avisos: silencio si no había nada que extraer; registro si hubo chunks fallidos que el traslape
recuperará; alerta si dos corridas consecutivas fallan, o si la cobertura queda incompleta después
del traslape, o si `preprocess.py` truena por un invariante duro.

---

## 6. Ejemplo de cron

```cron
# Ruta diaria: MDA y pronóstico. 6:00, ya publicado el MDA del día siguiente.
0 6 * * *  cd /opt/monclova && ./run_pipeline.sh daily >> logs/daily.log 2>&1

# Ruta rezagada: MTR y anomalías. Lunes 7:00, cubriendo la semana completa.
0 7 * * 1  cd /opt/monclova && ./run_pipeline.sh mtr >> logs/mtr.log 2>&1

# Reevaluación mensual: mide si los modelos siguen siendo buenos.
0 3 1 * *  cd /opt/monclova && ./run_pipeline.sh backtest >> logs/backtest.log 2>&1
```

---

## 7. Resumen de cambios necesarios

| Cambio | Dónde | Esfuerzo |
|---|---|---|
| Columna `fecha_extraccion` en el CSV crudo | `extract.py` | trivial |
| Upsert por `(process, node_id, timestamp)` en vez de concat | `extract.py` | bajo |
| Flag `--incremental` y lectura del estado | `extract.py` | bajo |
| Ventana de traslape por proceso | `extract.py` + `config.yaml` | bajo |
| No avanzar el estado si hubo chunks fallidos | `extract.py` | bajo |
| Lock de archivo contra corridas solapadas | `common.py` | bajo |
| Reporte de filas que cambiaron de valor en el traslape | `extract.py` | medio |
| Reintento de corrida completa y política de alertas | orquestador | medio |
