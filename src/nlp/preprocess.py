"""Pipeline NLP: tokenización, lematización, stopwords y TF-IDF."""
from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    CLEANED_CSV_PATH,
    LABEL_MAP,
    RAW_CSV_PATH,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    VECTORIZED_CSV_PATH,
    VECTORIZER_PATH,
)


def _ensure_nltk_resources() -> None:
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


_ensure_nltk_resources()

_STOPWORDS = set(stopwords.words("spanish")) | set(stopwords.words("english"))


@lru_cache(maxsize=1)
def _spacy_nlp():
    """Carga spaCy es_core_news_sm si está instalado; si no, None."""
    try:
        import spacy

        return spacy.load("es_core_news_sm")
    except Exception:
        return None


def normalize_label(value) -> int:
    if value in LABEL_MAP:
        return int(LABEL_MAP[value])
    text = str(value).strip().lower()
    if text in LABEL_MAP:
        return int(LABEL_MAP[text])
    raise ValueError(f"Etiqueta desconocida: {value!r}")


def _lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Lematiza con spaCy si hay modelo; si no, conserva tokens."""
    nlp = _spacy_nlp()
    if nlp is not None:
        doc = nlp(" ".join(tokens))
        return [t.lemma_.lower() for t in doc if t.lemma_.strip()]
    return tokens


def clean_text(text: str) -> str:
    """Tokenización + remoción de stopwords/puntuación + lematización (spaCy si existe)."""
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-záéíóúüñàèìòù\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [t for t in word_tokenize(text) if t not in _STOPWORDS and len(t) >= 3]
    tokens = _lemmatize_tokens(tokens)
    return " ".join(tokens)


def extract_data(input_path: Path | str | None = None) -> pd.DataFrame:
    path = Path(input_path) if input_path else RAW_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {path}")

    df = pd.read_csv(path)
    required = {"texto", "etiqueta"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en CSV: {missing}")

    df = df.dropna(subset=["texto", "etiqueta"]).copy()
    df["etiqueta"] = df["etiqueta"].map(normalize_label)
    return df.reset_index(drop=True)


def clean_data(df: pd.DataFrame | None = None, output_path: Path | str | None = None) -> pd.DataFrame:
    if df is None:
        df = extract_data()

    cleaned = df.copy()
    cleaned["texto_limpio"] = cleaned["texto"].astype(str).map(clean_text)
    cleaned = cleaned[cleaned["texto_limpio"].str.len() > 0].reset_index(drop=True)

    out = Path(output_path) if output_path else CLEANED_CSV_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out, index=False)
    return cleaned


def feature_engineering(
    df: pd.DataFrame | None = None,
    fit: bool = True,
    vectorizer_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> tuple[pd.DataFrame, TfidfVectorizer]:
    """Convierte texto limpio a matriz TF-IDF y persiste vectorizer + CSV."""
    if df is None:
        cleaned_path = CLEANED_CSV_PATH
        if not cleaned_path.exists():
            df = clean_data()
        else:
            df = pd.read_csv(cleaned_path)

    if "texto_limpio" not in df.columns:
        df = clean_data(df)

    vec_path = Path(vectorizer_path) if vectorizer_path else VECTORIZER_PATH
    vec_path.parent.mkdir(parents=True, exist_ok=True)

    if fit:
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=1,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(df["texto_limpio"].astype(str))
        joblib.dump(vectorizer, vec_path)
    else:
        if not vec_path.exists():
            raise FileNotFoundError(f"Vectorizer no encontrado: {vec_path}")
        vectorizer = joblib.load(vec_path)
        matrix = vectorizer.transform(df["texto_limpio"].astype(str))

    feature_names = [f"tfidf_{i}" for i in range(matrix.shape[1])]
    features = pd.DataFrame(matrix.toarray(), columns=feature_names)
    features["etiqueta"] = df["etiqueta"].values

    out = Path(output_path) if output_path else VECTORIZED_CSV_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out, index=False)
    return features, vectorizer


def transform_texts(texts: list[str], vectorizer_path: Path | str | None = None) -> pd.DataFrame:
    """Transforma textos crudos para inferencia (limpia + TF-IDF)."""
    vec_path = Path(vectorizer_path) if vectorizer_path else VECTORIZER_PATH
    if not vec_path.exists():
        raise FileNotFoundError(f"Vectorizer no encontrado: {vec_path}")

    vectorizer: TfidfVectorizer = joblib.load(vec_path)
    cleaned = [clean_text(t) for t in texts]
    matrix = vectorizer.transform(cleaned)
    feature_names = [f"tfidf_{i}" for i in range(matrix.shape[1])]
    return pd.DataFrame(matrix.toarray(), columns=feature_names)


def run_nlp_pipeline(
    input_path: Path | str | None = None,
    cleaned_path: Path | str | None = None,
    vectorized_path: Path | str | None = None,
) -> dict:
    """Ejecuta extract -> clean -> TF-IDF de punta a punta."""
    raw = extract_data(input_path)
    cleaned = clean_data(raw, output_path=cleaned_path)
    features, vectorizer = feature_engineering(
        cleaned,
        fit=True,
        output_path=vectorized_path,
    )
    return {
        "n_samples": len(cleaned),
        "n_features": features.shape[1] - 1,
        "cleaned_path": str(cleaned_path or CLEANED_CSV_PATH),
        "vectorized_path": str(vectorized_path or VECTORIZED_CSV_PATH),
        "vectorizer_path": str(VECTORIZER_PATH),
        "vocabulary_size": len(vectorizer.vocabulary_),
    }


if __name__ == "__main__":
    result = run_nlp_pipeline()
    print("NLP pipeline OK:")
    for key, value in result.items():
        print(f"  {key}: {value}")
