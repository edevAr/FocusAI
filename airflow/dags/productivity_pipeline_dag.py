"""
DAG Airflow: Pipeline MLOps FocusAI
extract_data -> clean_data -> wait_for_cleaned_csv -> feature_engineering -> train_model -> evaluate_model

Cambios v2 (MLOps hardening):
  - FileSensor: espera la existencia de CLEANED_CSV_PATH antes de feature_engineering.
  - default_args robustecidos: retries=3, retry_exponential_backoff=True, email_on_failure=True.
  - _task_train_model: verifica disponibilidad de MLflow vía HttpHook antes de entrenar;
    si el servidor no responde, imprime advertencia y continúa con la URI de settings.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.hooks.http_hook import HttpHook
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Callables de las tareas Python
# ---------------------------------------------------------------------------


def _task_extract_data(**_context):
    from src.nlp.preprocess import extract_data

    df = extract_data()
    return {"n_rows": len(df), "columns": list(df.columns)}


def _task_clean_data(**_context):
    from src.nlp.preprocess import clean_data, extract_data

    cleaned = clean_data(extract_data())
    return {"n_cleaned": len(cleaned)}


def _task_feature_engineering(**_context):
    import pandas as pd

    from config.settings import CLEANED_CSV_PATH
    from src.nlp.preprocess import feature_engineering

    df = pd.read_csv(CLEANED_CSV_PATH)
    features, vectorizer = feature_engineering(df, fit=True)
    return {
        "n_samples": len(features),
        "n_features": features.shape[1] - 1,
        "vocab_size": len(vectorizer.vocabulary_),
    }


def _task_train_model(**_context):
    """
    Verifica la disponibilidad de MLflow mediante un HttpHook apuntando a la
    conexión 'mlflow_default' (host configurado en Airflow Connections).
    Si el servidor no responde, emite una advertencia pero continúa el
    entrenamiento usando la URI definida en config/settings.py.
    """
    from config.settings import MLFLOW_TRACKING_URI
    from src.training.train_model import train_and_evaluate

    # --- Health-check de MLflow -------------------------------------------
    mlflow_reachable = False
    try:
        http_hook = HttpHook(method="GET", http_conn_id="mlflow_default")
        # El endpoint raíz de MLflow devuelve 200 OK cuando el servidor está activo
        response = http_hook.run(endpoint="/")
        if response.status_code == 200:
            mlflow_reachable = True
            print(
                f"[MLflow Health-check] Servidor alcanzable (HTTP {response.status_code}). "
                "Procediendo con el entrenamiento."
            )
        else:
            print(
                f"[MLflow Health-check] ADVERTENCIA: el servidor respondió con "
                f"HTTP {response.status_code}. Se usará la URI de fallback: {MLFLOW_TRACKING_URI}"
            )
    except Exception as exc:  # noqa: BLE001
        print(
            f"[MLflow Health-check] ADVERTENCIA: no se pudo conectar al servidor MLflow "
            f"('mlflow_default'). Error: {exc}. "
            f"Se usará la URI de fallback definida en settings: {MLFLOW_TRACKING_URI}"
        )

    if not mlflow_reachable:
        # Forzamos la URI desde settings para que el código de entrenamiento la use
        import os

        os.environ.setdefault("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)

    # --- Entrenamiento -------------------------------------------------------
    metrics = train_and_evaluate(register_model=True)
    return metrics


def _task_evaluate_model(**_context):
    from src.training.train_model import evaluate_model

    return evaluate_model()


def _task_init_database(**_context):
    from src.database.db import init_db

    path = init_db()
    return {"database_path": str(path)}


# ---------------------------------------------------------------------------
# Argumentos por defecto — robustecidos
# ---------------------------------------------------------------------------

default_args = {
    "owner": "focusai",
    "depends_on_past": False,
    # Notificación por correo en caso de fallo (dirección dummy para demo)
    "email": ["mlops-alerts@focusai.local"],
    "email_on_failure": True,
    "email_on_retry": False,
    # Política de reintentos: 3 intentos con backoff exponencial
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    # Tiempo máximo de espera entre reintentos (evita tiempos de espera excesivos)
    "max_retry_delay": timedelta(minutes=30),
}

# ---------------------------------------------------------------------------
# Definición del DAG
# ---------------------------------------------------------------------------

with DAG(
    dag_id="focusai_productivity_pipeline",
    default_args=default_args,
    description=(
        "Pipeline NLP + Ensemble (PyCaret) + MLflow para clasificar "
        "entradas de diario como Productivo o Procrastinación"
    ),
    schedule_interval=None,  # Manual / botón en UI
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "nlp", "pycaret", "mlflow", "focusai"],
) as dag:

    # -----------------------------------------------------------------------
    # Tarea 1: Extracción de datos raw
    # -----------------------------------------------------------------------
    extract_data = PythonOperator(
        task_id="extract_data",
        python_callable=_task_extract_data,
    )

    # -----------------------------------------------------------------------
    # Tarea 2: Limpieza de datos
    # -----------------------------------------------------------------------
    clean_data = PythonOperator(
        task_id="clean_data",
        python_callable=_task_clean_data,
    )

    # -----------------------------------------------------------------------
    # Sensor: espera hasta que el CSV limpio exista en disco antes de avanzar
    # -----------------------------------------------------------------------
    # CLEANED_CSV_PATH se importa aquí solo para obtener el str del path;
    # la importación se hace en tiempo de parseo del DAG (ligero, solo Path ops).
    from config.settings import CLEANED_CSV_PATH  # noqa: E402

    wait_for_cleaned_csv = FileSensor(
        task_id="wait_for_cleaned_csv",
        filepath=str(CLEANED_CSV_PATH),
        # Comprueba cada 30 s; timeout de 10 min para no bloquear el scheduler
        poke_interval=30,
        timeout=600,
        mode="reschedule",  # Libera el worker slot entre comprobaciones
        soft_fail=False,    # Si expira el timeout, la tarea falla (no se omite)
    )

    # -----------------------------------------------------------------------
    # Tarea 3: Feature engineering (bloqueada por el sensor)
    # -----------------------------------------------------------------------
    feature_engineering = PythonOperator(
        task_id="feature_engineering",
        python_callable=_task_feature_engineering,
    )

    # -----------------------------------------------------------------------
    # Tarea 4: Entrenamiento con health-check de MLflow
    # -----------------------------------------------------------------------
    train_model = PythonOperator(
        task_id="train_model",
        python_callable=_task_train_model,
    )

    # -----------------------------------------------------------------------
    # Tarea 5: Evaluación del modelo
    # -----------------------------------------------------------------------
    evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable=_task_evaluate_model,
    )

    # -----------------------------------------------------------------------
    # Tarea paralela: Inicialización de base de datos (no depende de ML)
    # -----------------------------------------------------------------------
    init_database = PythonOperator(
        task_id="init_database",
        python_callable=_task_init_database,
    )

    # -----------------------------------------------------------------------
    # Dependencias del pipeline
    # extract_data >> clean_data >> wait_for_cleaned_csv >> feature_engineering
    #              >> train_model >> evaluate_model
    # extract_data >> init_database  (rama paralela, sin cambios)
    # -----------------------------------------------------------------------
    (
        extract_data
        >> clean_data
        >> wait_for_cleaned_csv
        >> feature_engineering
        >> train_model
        >> evaluate_model
    )
    extract_data >> init_database
