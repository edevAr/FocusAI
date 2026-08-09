"""Compara BoW vs TF-IDF con CV estratificado y deja evidencia documentada."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    CV_FOLDS,
    RANDOM_STATE,
    VECTORIZER_COMPARISON_CSV,
    VECTORIZER_COMPARISON_MD,
    VECTORIZER_METHOD,
)
from src.nlp.preprocess import clean_data, extract_data, feature_engineering


def _evaluate_method(method: str, cleaned: pd.DataFrame) -> dict:
    features, vectorizer = feature_engineering(
        cleaned,
        fit=True,
        method=method,
        output_path=PROJECT_ROOT / "data" / "processed" / f"vectorized_{method}.csv",
        vectorizer_path=PROJECT_ROOT / "data" / "models" / f"{method}_vectorizer.joblib",
    )
    X = features.drop(columns=["etiqueta"])
    y = features["etiqueta"].astype(int)
    pipe = Pipeline(
        [
            ("scaler", MaxAbsScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    f1 = cross_val_score(pipe, X, y, cv=cv, scoring="f1_weighted")
    return {
        "method": method,
        "n_samples": int(len(features)),
        "n_features": int(X.shape[1]),
        "vocab_size": int(len(vectorizer.vocabulary_)),
        "cv_accuracy_mean": float(acc.mean()),
        "cv_accuracy_std": float(acc.std()),
        "cv_f1_mean": float(f1.mean()),
        "cv_f1_std": float(f1.std()),
    }


def compare_vectorizers() -> pd.DataFrame:
    raw = extract_data(apply_quality=True)
    cleaned = clean_data(raw)
    rows = [_evaluate_method("bow", cleaned), _evaluate_method("tfidf", cleaned)]
    comparison = pd.DataFrame(rows).sort_values("cv_f1_mean", ascending=False).reset_index(drop=True)

    VECTORIZER_COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(VECTORIZER_COMPARISON_CSV, index=False)

    winner = comparison.iloc[0]["method"]
    recommendation = (
        "TF-IDF" if winner == "tfidf" else "BoW (CountVectorizer)"
    )
    md = f"""# Comparación BoW vs TF-IDF

Experimento reproducible del módulo NLP (Edén — FocusAI).

## Setup

- Corpus: `data/raw/journal_entries.csv` tras pipeline de calidad
- Limpieza: NLTK tokenizer + stopwords + lematización spaCy (`es_core_news_sm`)
- Clasificador proxy: RandomForest (200 árboles, `class_weight=balanced`)
- Validación: Stratified K-Fold (`k={CV_FOLDS}`, `random_state={RANDOM_STATE}`)
- Métricas: Accuracy y F1 weighted

## Resultados

| method | n_samples | n_features | vocab | Accuracy CV | F1 CV |
|--------|-----------|------------|-------|-------------|-------|
"""
    for _, row in comparison.iterrows():
        md += (
            f"| {row['method']} | {row['n_samples']} | {row['n_features']} | "
            f"{row['vocab_size']} | {row['cv_accuracy_mean']:.4f} ± {row['cv_accuracy_std']:.4f} | "
            f"{row['cv_f1_mean']:.4f} ± {row['cv_f1_std']:.4f} |\n"
        )

    md += f"""
## Conclusión

- **Ganador por F1:** `{winner}` → se recomienda **{recommendation}** como vectorizer por defecto.
- Default actual del proyecto (`VECTORIZER_METHOD`): `{VECTORIZER_METHOD}`.
- CSV detallado: `{VECTORIZER_COMPARISON_CSV.relative_to(PROJECT_ROOT)}`.

### Por qué TF-IDF suele ganar en este dominio

- Penaliza tokens muy frecuentes poco discriminativos.
- Resalta n-gramas distintivos de productividad vs procrastinación.
- BoW sigue disponible vía `VECTORIZER_METHOD=bow` para experimentos.

## Cómo reproducir

```bash
source .venv/bin/activate
export PYTHONPATH="$(pwd)"
python -m src.nlp.vectorizer_comparison
```
"""
    VECTORIZER_COMPARISON_MD.parent.mkdir(parents=True, exist_ok=True)
    VECTORIZER_COMPARISON_MD.write_text(md, encoding="utf-8")
    return comparison


if __name__ == "__main__":
    df = compare_vectorizers()
    print(df.to_string(index=False))
    print(f"\nReporte Markdown: {VECTORIZER_COMPARISON_MD}")
    print(f"CSV: {VECTORIZER_COMPARISON_CSV}")
