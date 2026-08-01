#!/usr/bin/env bash
# Configura variables de entorno para Airflow apuntando a este repo
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export AIRFLOW_HOME="${AIRFLOW_HOME:-$ROOT/.airflow}"
export AIRFLOW__CORE__DAGS_FOLDER="$ROOT/airflow/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__EXECUTOR=LocalExecutor
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"

mkdir -p "$AIRFLOW_HOME"

if [[ ! -f "$AIRFLOW_HOME/airflow.db" ]]; then
  echo "Inicializando Airflow DB..."
  airflow db init
  airflow users create \
    --username admin \
    --password admin \
    --firstname Focus \
    --lastname AI \
    --role Admin \
    --email admin@focusai.local || true
fi

echo "Airflow UI -> http://127.0.0.1:8080 (admin/admin)"
echo "DAG: focusai_productivity_pipeline"
airflow webserver --port 8080 &
WEB_PID=$!
airflow scheduler &
SCHED_PID=$!

trap 'kill $WEB_PID $SCHED_PID 2>/dev/null || true' EXIT
wait
