#!/usr/bin/env bash
# Script de bootstrap: inicializa Airflow y ejecuta el DAG en modo test CLI
# AIRFLOW_HOME se sitúa en filesystem Linux nativo para evitar
# el error "disk I/O error" de SQLite sobre NTFS/WSL (/mnt/c/).
set -euo pipefail

export PATH="/home/gabo/.local/bin:$PATH"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONSAFEPATH=1

# AIRFLOW_HOME en Linux nativo (no en /mnt/c → evita disk I/O error de SQLite)
export AIRFLOW_HOME="/home/gabo/airflow-focusai"
mkdir -p "$AIRFLOW_HOME"

# Los DAGs se leen desde el proyecto en Windows via /mnt/c
export AIRFLOW__CORE__DAGS_FOLDER="/mnt/c/Users/Gabo/Desktop/FocusAI/airflow/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES="False"
export AIRFLOW__CORE__EXECUTOR="SequentialExecutor"
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="sqlite:////home/gabo/airflow-focusai/airflow.db"
export PYTHONPATH="/mnt/c/Users/Gabo/Desktop/FocusAI"
export MLFLOW_ALLOW_FILE_STORE=true

echo "=== [1/3] Inicializando base de datos de Airflow ==="
airflow db init 2>&1

echo ""
echo "=== [2/3] Verificando que el DAG se parsea correctamente ==="
airflow dags list 2>&1 | grep -E "focusai|dag_id|ERROR" || true

echo ""
echo "=== [3/3] Ejecutando pipeline: focusai_productivity_pipeline ==="
EXEC_DATE=$(date -u +%Y-%m-%dT%H:%M:%S)
airflow dags test focusai_productivity_pipeline "$EXEC_DATE" 2>&1

echo ""
echo "=== Pipeline finalizado ==="
