"""
DAG Airflow: Pipeline MLOps FocusAI
extract_data -> clean_data -> feature_engineering -> train_model -> evaluate_model
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
    from src.training.train_model import train_and_evaluate

    metrics = train_and_evaluate(register_model=True)
    return metrics


def _task_evaluate_model(**_context):
    from src.training.train_model import evaluate_model

    return evaluate_model()


def _task_init_database(**_context):
    from src.database.db import init_db

    path = init_db()
    return {"database_path": str(path)}


default_args = {
    "owner": "focusai",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

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
    extract_data = PythonOperator(
        task_id="extract_data",
        python_callable=_task_extract_data,
    )

    clean_data = PythonOperator(
        task_id="clean_data",
        python_callable=_task_clean_data,
    )

    feature_engineering = PythonOperator(
        task_id="feature_engineering",
        python_callable=_task_feature_engineering,
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=_task_train_model,
    )

    evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable=_task_evaluate_model,
    )

    init_database = PythonOperator(
        task_id="init_database",
        python_callable=_task_init_database,
    )

    (
        extract_data
        >> clean_data
        >> feature_engineering
        >> train_model
        >> evaluate_model
    )
    extract_data >> init_database
