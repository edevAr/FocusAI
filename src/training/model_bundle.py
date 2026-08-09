"""MLflow PyFunc bundle for the public ``texto`` prediction contract."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import mlflow.pyfunc

from src.nlp.preprocess import clean_text, feature_column_names


class TextPredictionModel(mlflow.pyfunc.PythonModel):
    """Package cleaner, vectorizer, and classifier in one MLflow model."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self.classifier = joblib.load(context.artifacts["classifier"])
        payload = joblib.load(context.artifacts["vectorizer"])
        if isinstance(payload, dict) and "vectorizer" in payload:
            self.vectorizer = payload["vectorizer"]
        else:
            self.vectorizer = payload

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame) -> pd.DataFrame:
        if "texto" not in model_input.columns:
            raise ValueError("El modelo requiere una columna 'texto'.")

        texts = model_input["texto"]
        if texts.isna().any() or not texts.map(lambda value: isinstance(value, str) and value.strip()).all():
            raise ValueError("La columna 'texto' debe contener texto no vacío.")

        cleaned = texts.map(clean_text)
        if not cleaned.str.len().gt(0).all():
            raise ValueError("La columna 'texto' no contiene términos utilizables.")

        matrix = self.vectorizer.transform(cleaned)
        features = pd.DataFrame(
            matrix.toarray(),
            columns=feature_column_names(matrix.shape[1]),
            index=model_input.index,
        )
        labels = self.classifier.predict(features)
        probabilities = self.classifier.predict_proba(features)
        return pd.DataFrame(
            {
                "prediccion": ["Productivo" if int(label) == 1 else "Procrastinación" for label in labels],
                "label_id": [int(label) for label in labels],
                "probabilidad": probabilities.max(axis=1),
                "texto_limpio": cleaned,
            },
            index=model_input.index,
        )


def model_artifacts(classifier_path: Path, vectorizer_path: Path) -> dict[str, str]:
    """Return the two local artifacts that are embedded in the PyFunc model."""
    return {"classifier": str(classifier_path), "vectorizer": str(vectorizer_path)}
