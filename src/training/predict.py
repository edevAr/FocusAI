"""Inference through the MLflow Production alias bundle."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.training.registry import get_production_model


def predict_texts(texts: list[str]) -> list[dict[str, Any]]:
    if not texts:
        return []
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Cada texto debe ser texto no vacío.")

    loaded = get_production_model()
    output = loaded.model.predict(pd.DataFrame({"texto": texts}))
    required = {"prediccion", "label_id", "probabilidad", "texto_limpio"}
    if not isinstance(output, pd.DataFrame) or not required.issubset(output.columns):
        raise RuntimeError("The Production bundle returned an invalid prediction contract.")

    identity = loaded.identity.as_dict()
    return [
        {**row, "model": identity}
        for row in output.loc[:, ["prediccion", "label_id", "probabilidad", "texto_limpio"]]
        .to_dict(orient="records")
    ]


def predict_one(text: str) -> dict[str, Any]:
    return predict_texts([text])[0]
