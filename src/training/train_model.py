from __future__ import annotations
"""Entrenamiento Ensemble (PyCaret preferido, sklearn fallback) + MLflow."""
# ---------------------------------------------------------------------------
# Limitar hilos de librerías matemáticas ANTES de cualquier import numérico.
# Evita que OpenBLAS/MKL/OpenMP lancen workers paralelos que bloquean Windows.
# ---------------------------------------------------------------------------
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
# ---------------------------------------------------------------------------

import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    CV_FOLDS,
    METRICS_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    RANDOM_STATE,
    VECTORIZED_CSV_PATH,
)
from src.nlp.preprocess import run_nlp_pipeline

warnings.filterwarnings("ignore")


def _configure_mlflow(tracking_uri: str | None = None) -> str:
    # Forzar tracking local absoluto con SQLite — sin servidor web, sin file_store.
    # IMPORTANTE: La DB debe vivir en filesystem nativo Linux (/home/),
    # NO en /mnt/c/ donde SQLite WAL causa "disk I/O error".
    import platform
    if platform.system() != "Windows" and os.path.exists("/home/gabo"):
        # WSL: usar filesystem nativo Linux
        db_path = "/home/gabo/mlflow_tracking.db"
    else:
        # Windows nativo: usar ruta del proyecto
        db_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../../mlflow_tracking.db')
        )
    sqlite_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(sqlite_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    return sqlite_uri


def _pycaret_include() -> list[str]:
    include = ["rf"]
    try:
        import xgboost  # noqa: F401

        include.append("xgboost")
    except Exception:  # noqa: BLE001
        pass
    try:
        import lightgbm  # noqa: F401

        include.append("lightgbm")
    except Exception:  # noqa: BLE001
        pass
    return include


def _load_features(path: Path | str | None = None) -> pd.DataFrame:
    features_path = Path(path) if path else VECTORIZED_CSV_PATH
    if not features_path.exists():
        run_nlp_pipeline()
    df = pd.read_csv(features_path)
    if "etiqueta" not in df.columns:
        raise ValueError("El dataset vectorizado debe incluir la columna 'etiqueta'")
    return df


def _build_candidates() -> dict:
    from sklearn.ensemble import GradientBoostingClassifier

    candidates: dict = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }
    try:
        from xgboost import XGBClassifier

        candidates["XGBoost"] = XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"XGBoost no disponible: {exc}")

    try:
        from lightgbm import LGBMClassifier

        candidates["LightGBM"] = LGBMClassifier(
            n_estimators=120,
            learning_rate=0.08,
            num_leaves=31,
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"LightGBM no disponible: {exc}")

    return candidates


def _evaluate_cv(model, X: pd.DataFrame, y: pd.Series) -> dict:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    pipe = Pipeline([("scaler", StandardScaler(with_mean=False)), ("model", model)])
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")
    preds = np.argmax(proba, axis=1)
    metrics = {
        "Accuracy": float(accuracy_score(y, preds)),
        "F1": float(f1_score(y, preds, average="weighted")),
        "Precision": float(precision_score(y, preds, average="weighted", zero_division=0)),
        "Recall": float(recall_score(y, preds, average="weighted", zero_division=0)),
    }
    try:
        metrics["AUC"] = float(roc_auc_score(y, proba[:, 1]))
    except Exception:  # noqa: BLE001
        pass
    return metrics


def _train_with_sklearn(data: pd.DataFrame) -> tuple[object, dict, pd.DataFrame]:
    X = data.drop(columns=["etiqueta"])
    y = data["etiqueta"].astype(int)

    rows = []
    best_name = None
    best_score = -1.0
    best_model = None
    best_metrics: dict = {}

    for name, model in _build_candidates().items():
        metrics = _evaluate_cv(model, X, y)
        rows.append({"Model": name, **metrics})
        if metrics["F1"] > best_score:
            best_score = metrics["F1"]
            best_name = name
            best_metrics = metrics
            best_model = model

    comparison = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)

    final_pipe = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("model", best_model),
        ]
    )
    final_pipe.fit(X, y)
    best_metrics = {
        **best_metrics,
        "model_name": best_name,
        "trainer": "sklearn_cv",
        "cv_folds": CV_FOLDS,
        "n_samples": int(len(data)),
        "n_features": int(X.shape[1]),
    }
    return final_pipe, best_metrics, comparison


def _train_with_pycaret(data: pd.DataFrame) -> tuple[object, dict, pd.DataFrame]:
    """Entrenamiento con PyCaret compare_models — Modo Seguro WSL.

    Reglas de seguridad:
      - n_jobs=1          → PROHIBIDO paralelismo (deadlock C++ en WSL)
      - log_experiment=False → logueo manual a MLflow, no interferir con Airflow
      - html=False        → sin renderizado HTML (entorno CLI/Airflow)
      - verbose=False      → salida limpia en logs de Airflow
    """
    from pycaret.classification import (
        compare_models,
        finalize_model,
        pull,
        save_model,
        setup,
    )

    # ── PyCaret Setup (Modo Seguro WSL) ─────────────────────────────────
    setup(
        data=data,
        target="etiqueta",
        session_id=RANDOM_STATE,   # reproducibilidad
        n_jobs=1,                  # ⛔ PROHIBIDO >1: deadlock en WSL
        log_experiment=False,      # ⛔ PROHIBIDO True: MLflow se loguea manualmente
        html=False,                # sin renderizado en Airflow/CLI
        verbose=False,             # salida limpia
        use_gpu=False,             # sin GPU
    )

    # ── Comparación multi-modelo con cross-validation ───────────────────
    # PyCaret compara internamente: lr, rf, dt, knn, svm, etc.
    # fold=5 para validación cruzada, sort='Accuracy' para elegir el mejor
    include_models = _pycaret_include()  # rf + xgboost/lightgbm si disponibles
    best_model = compare_models(
        fold=CV_FOLDS,
        sort="Accuracy",
        include=include_models,
        n_select=1,                # devolver solo el mejor modelo
        verbose=False,
    )
    comparison = pull()  # DataFrame con métricas de todos los modelos comparados

    # ── Finalizar (re-entrena con 100% de los datos) ────────────────────
    finalized = finalize_model(best_model)
    save_model(finalized, str(MODEL_PATH))

    # ── Extraer métricas del mejor modelo ───────────────────────────────
    best_row = comparison.iloc[0]
    metrics = {
        "Accuracy": float(best_row.get("Accuracy", 0.0)),
        "F1": float(best_row.get("F1", 0.0)),
        "Precision": float(best_row.get("Prec.", best_row.get("Precision", 0.0))),
        "Recall": float(best_row.get("Recall", 0.0)) if "Recall" in best_row else None,
        "AUC": float(best_row.get("AUC", 0.0)) if "AUC" in best_row else None,
        "model_name": str(best_row.get("Model", type(best_model).__name__)),
        "trainer": "pycaret_compare_models",
        "cv_folds": CV_FOLDS,
        "n_samples": int(len(data)),
        "n_features": int(data.shape[1] - 1),
        "models_compared": ", ".join(include_models),
    }
    metrics = {k: v for k, v in metrics.items() if v is not None}
    return finalized, metrics, comparison


def train_and_evaluate(
    features_path: Path | str | None = None,
    model_path: Path | str | None = None,
    tracking_uri: str | None = None,
    register_model: bool = True,
    prefer_pycaret: bool = True,
) -> dict:
    """Compara XGBoost / RF / LightGBM y registra métricas/artefactos en MLflow."""
    used_uri = _configure_mlflow(tracking_uri)
    data = _load_features(features_path)
    out_model = Path(model_path) if model_path else MODEL_PATH
    out_model.parent.mkdir(parents=True, exist_ok=True)

    trainer_error = None
    model = None
    metrics: dict = {}
    comparison = pd.DataFrame()

    if prefer_pycaret:
        try:
            model, metrics, comparison = _train_with_pycaret(data)
        except Exception as exc:  # noqa: BLE001
            trainer_error = f"PyCaret falló ({exc}); usando sklearn fallback"
            model, metrics, comparison = _train_with_sklearn(data)
    else:
        model, metrics, comparison = _train_with_sklearn(data)

    # Artefacto principal para serving (joblib, sin depender de PyCaret en runtime)
    joblib_path = Path(f"{out_model}.joblib")
    joblib.dump(model, joblib_path)

    # Compatibilidad con predict/load_model de PyCaret si aplica
    pycaret_pkl = Path(f"{out_model}.pkl")
    if not pycaret_pkl.exists():
        joblib.dump(model, pycaret_pkl)

    comparison_path = out_model.parent / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    with mlflow.start_run(run_name="ensemble_compare") as run:
        metrics["mlflow_run_id"] = run.info.run_id
        metrics["mlflow_tracking_uri"] = used_uri
        if trainer_error:
            metrics["trainer_warning"] = trainer_error

        mlflow.log_params(
            {
                "cv_folds": CV_FOLDS,
                "models_compared": "xgboost,rf,lightgbm",
                "target": "etiqueta",
                "best_model": metrics.get("model_name", "unknown"),
                "trainer": metrics.get("trainer", "unknown"),
            }
        )
        mlflow.log_metrics(
            {
                k: float(v)
                for k, v in metrics.items()
                if k in {"Accuracy", "F1", "AUC", "Precision", "Recall"}
                and isinstance(v, (int, float))
            }
        )
        mlflow.log_artifact(str(joblib_path), artifact_path="model")
        mlflow.log_artifact(str(comparison_path), artifact_path="metrics")

        if register_model:
            try:
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="sklearn_model",
                    registered_model_name=MLFLOW_MODEL_NAME,
                )
                client = MlflowClient()
                versions = client.search_model_versions(f"name='{MLFLOW_MODEL_NAME}'")
                if versions:
                    latest = sorted(versions, key=lambda v: int(v.version))[-1]
                    metrics["registered_version"] = latest.version
            except Exception as exc:  # noqa: BLE001
                metrics["registry_warning"] = str(exc)

        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(str(METRICS_PATH), artifact_path="metrics")

    return metrics


def evaluate_model(metrics_path: Path | str | None = None) -> dict:
    """Lee métricas persistidas y las re-loguea en MLflow (tarea evaluate_model)."""
    path = Path(metrics_path) if metrics_path else METRICS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No hay métricas en {path}. Ejecuta train_and_evaluate primero."
        )
    with open(path, encoding="utf-8") as f:
        metrics = json.load(f)

    _configure_mlflow()
    with mlflow.start_run(run_name="evaluate_model"):
        loggable = {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float))
            and k in {"Accuracy", "F1", "AUC", "Recall", "Precision"}
        }
        if loggable:
            mlflow.log_metrics(loggable)
        mlflow.log_params({"source_metrics_path": str(path)})
        mlflow.log_artifact(str(path))
    return metrics


if __name__ == "__main__":
    result = train_and_evaluate()
    print("Training OK:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print("Evaluate OK:", evaluate_model())
