#!/usr/bin/env bash
#
# Orquestador del pipeline Monclova LMP.
#
#   ./run_pipeline.sh daily                         ruta diaria (MDA -> pronostico)
#   ./run_pipeline.sh mtr                           ruta rezagada (MTR -> alertas)
#   ./run_pipeline.sh full 2024-01-01 2025-06-30    carga historica completa
#   ./run_pipeline.sh test                          tests de invariantes
#
# Cada etapa tambien corre de forma independiente:
#   python scripts/extract.py --start 2025-07-01 --end 2025-07-15
#   python scripts/preprocess.py --verbose
#   python scripts/build_features.py
#   python scripts/run_forecasting.py
#   python scripts/run_anomaly_detection.py
#
# La EVALUACION de los modelos (backtesting, comparacion entre metodos, tuning)
# no esta aqui: vive en los notebooks de notebooks/model_selection/ y se corre a
# mano cuando se quiere re-evaluar. El orquestador solo ejecuta el camino
# operativo con los modelos ya decididos.
#
# Por que dos rutas y no una
# --------------------------
# Los dos tracks NO pueden correr en la misma cadencia. El MDA se publica un dia
# antes del Dia de Operacion, asi que el pronostico se actualiza a diario. El MTR
# (Expost) se publica hasta 7 dias despues, asi que la deteccion de anomalias
# opera necesariamente con ese rezago. Forzarlas al mismo horario haria que la
# ruta de anomalias pidiera datos que aun no existen.

set -euo pipefail

cd "$(dirname "$0")"
COMANDO="${1:-help}"

log() { printf '\n=== %s ===\n' "$1"; }

case "$COMANDO" in

  daily)
    # Ruta diaria: solo MDA. Ventana corta con traslape para tolerar
    # republicaciones de CENACE sobre dias ya extraidos.
    DESDE="${2:-$(date -d '10 days ago' +%Y-%m-%d)}"
    HASTA="${3:-$(date -d 'tomorrow' +%Y-%m-%d)}"

    log "Extraccion MDA ${DESDE} a ${HASTA}"
    python scripts/extract.py --start "$DESDE" --end "$HASTA" --processes MDA

    log "Preprocesamiento"
    python scripts/preprocess.py

    log "Features"
    python scripts/build_features.py

    log "Pronostico"
    python scripts/run_forecasting.py
    ;;

  mtr)
    # Ruta rezagada: MTR hasta D-7. El script recorta el rango solo si se le
    # piden dias que aun no se publican.
    DESDE="${2:-$(date -d '21 days ago' +%Y-%m-%d)}"
    HASTA="${3:-$(date +%Y-%m-%d)}"

    log "Extraccion MTR ${DESDE} a ${HASTA}"
    python scripts/extract.py --start "$DESDE" --end "$HASTA" --processes MTR

    log "Preprocesamiento"
    python scripts/preprocess.py

    log "Features"
    python scripts/build_features.py

    log "Deteccion de anomalias"
    python scripts/run_anomaly_detection.py
    ;;

  full)
    # Carga historica completa. La extraccion tarda ~35 min.
    DESDE="${2:?Uso: ./run_pipeline.sh full <desde> <hasta>}"
    HASTA="${3:?Uso: ./run_pipeline.sh full <desde> <hasta>}"

    log "Extraccion completa ${DESDE} a ${HASTA}"
    python scripts/extract.py --start "$DESDE" --end "$HASTA" --verbose

    log "Preprocesamiento"
    python scripts/preprocess.py --verbose

    log "Features"
    python scripts/build_features.py --verbose

    log "Pronostico"
    python scripts/run_forecasting.py

    log "Deteccion de anomalias"
    python scripts/run_anomaly_detection.py
    ;;

  test)
    python -m pytest tests/ -v
    ;;

  *)
    sed -n '3,22p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac

log "Pipeline completado"
