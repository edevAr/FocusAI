"""Inferencia: carga vectorizer + modelo entrenado."""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    LABEL_PRODUCTIVE,
    LABEL_PROCRASTINATION,
    MODEL_PATH,
    VECTORIZER_PATH,
)
from src.nlp.preprocess import clean_text, transform_texts


@lru_cache(maxsize=1)
def load_model(model_path: str | None = None):
    path = Path(model_path) if model_path else MODEL_PATH
    joblib_path = Path(f"{path}.joblib")
    pkl_path = Path(f"{path}.pkl")

    if joblib_path.exists():
        return joblib.load(joblib_path)

    if pkl_path.exists():
        try:
            from pycaret.classification import load_model as pycaret_load_model

            return pycaret_load_model(str(path))
        except Exception:
            return joblib.load(pkl_path)

    raise FileNotFoundError(
        f"Modelo no encontrado en {joblib_path} ni {pkl_path}. "
        "Ejecuta el pipeline de entrenamiento primero."
    )


def predict_texts(texts: list[str]) -> list[dict[str, Any]]:
    if not texts:
        return []

    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            f"Vectorizer no encontrado: {VECTORIZER_PATH}. Ejecuta el pipeline NLP."
        )

    features = transform_texts(texts)
    model = load_model()

    # Algunos wrappers de PyCaret exponen .predict sobre DataFrame completo
    try:
        preds = model.predict(features)
    except Exception:
        preds = model.predict(features.values)

    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(features)
        except Exception:  # noqa: BLE001
            try:
                proba = model.predict_proba(features.values)
            except Exception:  # noqa: BLE001
                proba = None

    results: list[dict[str, Any]] = []
    for i, text in enumerate(texts):
        label_id = int(preds[i])
        label = LABEL_PRODUCTIVE if label_id == 1 else LABEL_PROCRASTINATION
        confidence = float(max(proba[i])) if proba is not None else None
        results.append(
            {
                "texto": text,
                "texto_limpio": clean_text(text),
                "prediccion": label,
                "label_id": label_id,
                "probabilidad": confidence,
            }
        )
    return results


def predict_one(text: str) -> dict[str, Any]:
    return predict_texts([text])[0]
