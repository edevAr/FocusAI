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
# ---------------------------------------------------------------------------

import json
import sys
import warnings
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    CALIBRATION_METHOD,
    CV_FOLDS,
    HOLDOUT_METRICS_PATH,
    HOLDOUT_TEST_SIZE,
    LABEL_PROCRASTINATION,
    LABEL_PRODUCTIVE,
    METRICS_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    PER_CLASS_METRICS_PATH,
    RANDOM_STATE,
    TUNING_ITERATIONS,
    TUNING_RESULTS_PATH,
    VECTORIZED_CSV_PATH,
    VECTORIZER_PATH,
    QUALITY_MIN_ACCURACY,
    QUALITY_MIN_F1,
)
from src.training.model_bundle import TextPredictionModel, model_artifacts
from src.training.quality import evaluate_quality
from src.nlp.preprocess import run_nlp_pipeline

warnings.filterwarnings("ignore")


def _configure_mlflow(tracking_uri: str | None = None) -> str:
    uri = tracking_uri or MLFLOW_TRACKING_URI
    if not uri:
        raise ValueError("MLFLOW_TRACKING_URI must point to the running MLflow server.")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    return uri


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

    import platform
    if platform.system() != "Windows":
        # LightGBM produce access violation en Windows con datasets pequeños
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


def _param_grid(model) -> dict:
    """Devuelve el espacio de hiperparámetros según el tipo de modelo."""
    name = type(model).__name__
    if "RandomForest" in name:
        return {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__max_features": ["sqrt", "log2"],
        }
    if "GradientBoosting" in name:
        return {
            "model__n_estimators": [80, 120, 200],
            "model__learning_rate": [0.05, 0.08, 0.1, 0.2],
            "model__max_depth": [2, 3, 4, 5],
            "model__subsample": [0.7, 0.8, 0.9, 1.0],
        }
    if "XGB" in name:
        return {
            "model__n_estimators": [80, 120, 200],
            "model__max_depth": [3, 4, 5, 6],
            "model__learning_rate": [0.05, 0.08, 0.1, 0.2],
            "model__subsample": [0.7, 0.8, 0.9],
            "model__colsample_bytree": [0.7, 0.8, 0.9],
        }
    if "LGBM" in name:
        return {
            "model__n_estimators": [80, 120, 200],
            "model__learning_rate": [0.05, 0.08, 0.1, 0.2],
            "model__num_leaves": [15, 31, 63],
            "model__min_child_samples": [5, 10, 20],
        }
    return {}


def _tune_sklearn(
    pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[Pipeline, dict]:
    """Tuning via RandomizedSearchCV; devuelve el pipeline tuneado y los mejores params."""
    grid = _param_grid(pipe.named_steps["model"])
    if not grid:
        return pipe, {}

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=grid,
        n_iter=TUNING_ITERATIONS,
        scoring="f1_weighted",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    return search.best_estimator_, best_params


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


def _calibrate(model, X_train: pd.DataFrame, y_train: pd.Series):
    """Envuelve el modelo ya entrenado con CalibratedClassifierCV (cv='prefit').

    Usa Platt scaling ('sigmoid') por defecto — robusto con datasets pequeños.
    Isotonic requiere al menos ~1000 muestras para ser confiable.
    """
    calibrated = CalibratedClassifierCV(
        estimator=model,
        method=CALIBRATION_METHOD,
        cv="prefit",  # modelo ya entrenado; solo ajusta el calibrador
    )
    calibrated.fit(X_train, y_train)
    return calibrated


def _evaluate_holdout(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evalúa el modelo final sobre el conjunto hold-out (nunca visto en entrenamiento)."""
    preds = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)
    metrics = {
        "Accuracy": float(accuracy_score(y_test, preds)),
        "F1": float(f1_score(y_test, preds, average="weighted")),
        "Precision": float(precision_score(y_test, preds, average="weighted", zero_division=0)),
        "Recall": float(recall_score(y_test, preds, average="weighted", zero_division=0)),
        "n_samples_test": int(len(y_test)),
    }
    try:
        metrics["AUC"] = float(roc_auc_score(y_test, proba[:, 1]))
    except Exception:  # noqa: BLE001
        pass
    return metrics


def _compute_per_class_metrics(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Matriz de confusión + reporte por clase sobre el hold-out."""
    # Etiquetas legibles: 0 = Procrastinación, 1 = Productivo
    class_names = [LABEL_PROCRASTINATION, LABEL_PRODUCTIVE]

    preds = pipe.predict(X_test)

    # Matriz de confusión como lista de listas (serializable a JSON)
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    cm_dict = {
        "labels": class_names,
        "matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
    }

    # Classification report como dict
    report = classification_report(
        y_test, preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "confusion_matrix": cm_dict,
        "classification_report": report,
    }


def _train_with_sklearn(
    X_train: pd.DataFrame, y_train: pd.Series, n_total: int
) -> tuple[object, dict, pd.DataFrame]:
    rows = []
    best_name = None
    best_score = -1.0
    best_model = None
    best_metrics: dict = {}

    for name, model in _build_candidates().items():
        try:
            metrics = _evaluate_cv(model, X_train, y_train)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"{name} falló durante CV: {exc}")
            continue
        rows.append({"Model": name, **metrics})
        if metrics["F1"] > best_score:
            best_score = metrics["F1"]
            best_name = name
            best_metrics = metrics
            best_model = model

    if best_model is None:
        raise RuntimeError("Ningún modelo pudo entrenarse. Revisa las dependencias.")

    comparison = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)

    base_pipe = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("model", best_model),
        ]
    )

    # Tuning de hiperparámetros sobre el mejor candidato
    tuned_pipe, tuned_params = _tune_sklearn(base_pipe, X_train, y_train)

    best_metrics = {
        **best_metrics,
        "model_name": best_name,
        "trainer": "sklearn_cv",
        "cv_folds": CV_FOLDS,
        "n_samples_train": int(len(y_train)),
        "n_samples_total": n_total,
        "n_features": int(X_train.shape[1]),
        "tuned_params": tuned_params,
    }
    return tuned_pipe, best_metrics, comparison


def _train_with_pycaret(
    X_train: pd.DataFrame, y_train: pd.Series, n_total: int
) -> tuple[object, dict, pd.DataFrame]:
    from pycaret.classification import compare_models, finalize_model, pull, setup, tune_model

    train_data = X_train.copy()
    train_data["etiqueta"] = y_train.values

    setup(
        data=train_data,
        target="etiqueta",
        fold=CV_FOLDS,
        fold_strategy="stratifiedkfold",
        session_id=RANDOM_STATE,
        n_jobs=1,           # evita bloqueos OpenMP en Windows/WSL
        log_experiment=False,  # nosotros controlamos el logging en MLflow
        verbose=False,
    )

    include = _pycaret_include()
    # Agregar GBC (Gradient Boosting) para paridad con el path sklearn
    if "gbc" not in include:
        include.append("gbc")

    best = compare_models(
        fold=CV_FOLDS,
        sort="F1",
        include=include,
        n_select=1,
        verbose=False,
    )
    comparison = pull()

    # Tuneo de hiperparámetros sobre el mejor modelo
    tuned = tune_model(
        best,
        optimize="F1",
        n_iter=TUNING_ITERATIONS,
        search_library="scikit-learn",
        search_algorithm="random",
        fold=CV_FOLDS,
        verbose=False,
    )
    tuning_results = pull()

    # finalize_model re-entrena el modelo tuneado sobre todo X_train (no toca X_test)
    finalized = finalize_model(tuned)

    best_row = comparison.iloc[0]
    tuned_row = tuning_results.iloc[0] if not tuning_results.empty else best_row
    metrics = {
        "Accuracy": float(tuned_row.get("Accuracy", best_row.get("Accuracy", 0.0))),
        "F1": float(tuned_row.get("F1", best_row.get("F1", 0.0))),
        "Precision": float(tuned_row.get("Prec.", tuned_row.get("Precision", 0.0))),
        "Recall": float(tuned_row.get("Recall", 0.0)) if "Recall" in tuned_row else None,
        "AUC": float(tuned_row.get("AUC", 0.0)) if "AUC" in tuned_row else None,
        "model_name": str(best_row.get("Model", type(best).__name__)),
        "trainer": "pycaret",
        "cv_folds": CV_FOLDS,
        "tuning_iterations": TUNING_ITERATIONS,
        "n_samples_train": int(len(y_train)),
        "n_samples_total": n_total,
        "n_features": int(X_train.shape[1]),
    }
    metrics = {k: v for k, v in metrics.items() if v is not None}

    # Exportar tabla comparativa con resultados de tuning
    tuning_export = pd.concat(
        [comparison.assign(stage="compare"), tuning_results.assign(stage="tuned")],
        ignore_index=True,
    )
    return finalized, metrics, tuning_export


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

    # --- Hold-out split (estratificado, nunca toca el test en entrenamiento) ---
    X_all = data.drop(columns=["etiqueta"])
    y_all = data["etiqueta"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all,
        test_size=HOLDOUT_TEST_SIZE,
        stratify=y_all,
        random_state=RANDOM_STATE,
    )

    trainer_error = None
    model = None
    metrics: dict = {}
    comparison = pd.DataFrame()

    if prefer_pycaret:
        try:
            model, metrics, comparison = _train_with_pycaret(X_train, y_train, len(data))
        except Exception as exc:  # noqa: BLE001
            trainer_error = f"PyCaret falló ({exc}); usando sklearn fallback"
            model, metrics, comparison = _train_with_sklearn(X_train, y_train, len(data))
    else:
        model, metrics, comparison = _train_with_sklearn(X_train, y_train, len(data))

    # --- Calibración de probabilidades ---
    model = _calibrate(model, X_train, y_train)

    # --- Evaluación en hold-out ---
    holdout_metrics = _evaluate_holdout(model, X_test, y_test)
    holdout_metrics["holdout_test_size"] = HOLDOUT_TEST_SIZE

    # --- Métricas por clase ---
    per_class = _compute_per_class_metrics(model, X_test, y_test)

    # Artefacto principal para serving (joblib, sin depender de PyCaret en runtime)
    joblib_path = Path(f"{out_model}.joblib")
    joblib.dump(model, joblib_path)

    # Compatibilidad con predict/load_model de PyCaret si aplica
    pycaret_pkl = Path(f"{out_model}.pkl")
    if not pycaret_pkl.exists():
        joblib.dump(model, pycaret_pkl)

    comparison_path = out_model.parent / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    # Tabla de tuning (separada de la comparativa base)
    tuning_path = TUNING_RESULTS_PATH
    comparison.to_csv(tuning_path, index=False)

    with mlflow.start_run(run_name="ensemble_compare") as run:
        metrics["mlflow_run_id"] = run.info.run_id
        metrics["mlflow_tracking_uri"] = used_uri
        if trainer_error:
            metrics["trainer_warning"] = trainer_error

        mlflow.log_params(
            {
                "cv_folds": CV_FOLDS,
                "holdout_test_size": HOLDOUT_TEST_SIZE,
                "tuning_iterations": TUNING_ITERATIONS,
                "calibration_method": CALIBRATION_METHOD,
                "models_compared": "xgboost,rf,lightgbm,gbc",
                "target": "etiqueta",
                "best_model": metrics.get("model_name", "unknown"),
                "trainer": metrics.get("trainer", "unknown"),
            }
        )
        # Logear hiperparámetros del modelo tuneado (path sklearn)
        tuned_params = metrics.pop("tuned_params", {})
        if tuned_params:
            mlflow.log_params({f"tuned_{k}": str(v) for k, v in tuned_params.items()})
        # Métricas CV (prefijo cv_)
        mlflow.log_metrics(
            {
                f"cv_{k}": float(v)
                for k, v in metrics.items()
                if k in {"Accuracy", "F1", "AUC", "Precision", "Recall"}
                and isinstance(v, (int, float))
            }
        )
        # Métricas hold-out (prefijo holdout_)
        mlflow.log_metrics(
            {
                f"holdout_{k}": float(v)
                for k, v in holdout_metrics.items()
                if isinstance(v, (int, float))
                and k in {"Accuracy", "F1", "AUC", "Precision", "Recall"}
            }
        )

        mlflow.log_artifact(str(comparison_path), artifact_path="metrics")
        mlflow.log_artifact(str(tuning_path), artifact_path="metrics")

        if register_model:
            try:
                model_info = mlflow.pyfunc.log_model(
                    artifact_path="text_prediction_bundle",
                    python_model=TextPredictionModel(),
                    artifacts=model_artifacts(joblib_path, VECTORIZER_PATH),
                    code_path=["src"],
                    registered_model_name=MLFLOW_MODEL_NAME,
                )
                metrics["model_uri"] = model_info.model_uri
                versions = MlflowClient().search_model_versions(
                    f"name='{MLFLOW_MODEL_NAME}'"
                )
                if versions:
                    metrics["registered_version"] = max(
                        versions, key=lambda version: int(version.version)
                    ).version
            except Exception as exc:  # noqa: BLE001
                metrics["registry_warning"] = str(exc)

        gate = evaluate_quality(holdout_metrics, QUALITY_MIN_ACCURACY, QUALITY_MIN_F1)
        metrics["quality_eligible"] = gate.eligible
        metrics["quality_warnings"] = list(gate.warnings)
        mlflow.set_tags(
            {
                **gate.tags(),
                "candidate.run_id": run.info.run_id,
                "candidate.model_version": str(metrics.get("registered_version", "unavailable")),
            }
        )
        mlflow.log_metrics(
            {
                "quality_min_accuracy": QUALITY_MIN_ACCURACY,
                "quality_min_f1": QUALITY_MIN_F1,
                "quality_eligible": float(gate.eligible),
            }
        )

        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(str(METRICS_PATH), artifact_path="metrics")

        # Guardar métricas hold-out en archivo separado
        with open(HOLDOUT_METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(holdout_metrics, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(str(HOLDOUT_METRICS_PATH), artifact_path="metrics")

        # Guardar métricas por clase
        with open(PER_CLASS_METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(per_class, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(str(PER_CLASS_METRICS_PATH), artifact_path="metrics")

        # Logear F1 por clase en MLflow (valores escalares)
        report = per_class["classification_report"]
        for label in [LABEL_PROCRASTINATION, LABEL_PRODUCTIVE]:
            if label in report:
                mlflow.log_metrics({
                    f"f1_{label[:4].lower()}": float(report[label]["f1-score"]),
                    f"precision_{label[:4].lower()}": float(report[label]["precision"]),
                    f"recall_{label[:4].lower()}": float(report[label]["recall"]),
                })

    return {**metrics, "holdout": holdout_metrics, "per_class": per_class}


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
