"""Pipeline NLP: tokenización, lematización (spaCy obligatorio), calidad y vectorización."""
from __future__ import annotations

import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Parche NLTK inisec: cuando el venv está DENTRO del proyecto, inisec bloquea
# todos los paquetes del venv al considerarlos "del CWD". Solución: envolver
# sys.meta_path con una lista que ignora la inserción de NLTKSafeImportFinder.
# ---------------------------------------------------------------------------
class _MetaPathNoBlocker(list):  # type: ignore[type-arg]
    def insert(self, index, obj):  # type: ignore[override]
        if "NLTKSafeImportFinder" in type(obj).__name__:
            return
        super().insert(index, obj)


sys.meta_path = _MetaPathNoBlocker(sys.meta_path)
# ---------------------------------------------------------------------------

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    BOW_VECTORIZER_PATH,
    CLEANED_CSV_PATH,
    DATA_QUALITY_REPORT_PATH,
    FEATURE_COL_PREFIX,
    LABEL_MAP,
    RAW_CSV_PATH,
    SPACY_MODEL,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_VECTORIZER_PATH,
    VECTORIZED_CSV_PATH,
    VECTORIZER_METHOD,
    VECTORIZER_PATH,
)
from src.nlp.data_quality import clean_quality_issues, save_quality_report


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


def _install_spacy_model(model_name: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "spacy", "download", model_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


@lru_cache(maxsize=1)
def get_spacy_nlp(model_name: str | None = None):
    """Carga spaCy de forma obligatoria (intenta descargar el modelo si falta)."""
    name = model_name or SPACY_MODEL
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError(
            "spaCy es obligatorio para el pipeline NLP. "
            "Instálalo con: pip install 'spacy>=3.7.0,<3.8.0'"
        ) from exc

    try:
        return spacy.load(name)
    except OSError:
        try:
            _install_spacy_model(name)
            return spacy.load(name)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"No se pudo cargar el modelo spaCy '{name}'. "
                f"Ejecuta: python -m spacy download {name}"
            ) from exc


def normalize_label(value) -> int:
    if value in LABEL_MAP:
        return int(LABEL_MAP[value])
    text = str(value).strip().lower()
    if text in LABEL_MAP:
        return int(LABEL_MAP[text])
    raise ValueError(f"Etiqueta desconocida: {value!r}")


def _lemmatize_tokens(tokens: list[str], nlp=None) -> list[str]:
    nlp = nlp or get_spacy_nlp()
    doc = nlp(" ".join(tokens))
    return [t.lemma_.lower() for t in doc if t.lemma_ and t.lemma_.strip()]


def clean_text(text: str, nlp=None) -> str:
    """Tokenización + stopwords + lematización spaCy (obligatoria)."""
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-záéíóúüñàèìòù\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [t for t in word_tokenize(text) if t not in _STOPWORDS and len(t) >= 3]
    if not tokens:
        return ""
    tokens = _lemmatize_tokens(tokens, nlp=nlp)
    return " ".join(token for token in tokens if token and token not in _STOPWORDS)


def feature_column_names(n_features: int) -> list[str]:
    return [f"{FEATURE_COL_PREFIX}{i}" for i in range(n_features)]


def build_vectorizer(method: str | None = None):
    method = (method or VECTORIZER_METHOD).lower()
    common = dict(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=1,
    )
    if method == "tfidf":
        return TfidfVectorizer(sublinear_tf=True, **common)
    if method == "bow":
        return CountVectorizer(**common)
    raise ValueError(f"VECTORIZER_METHOD inválido: {method!r}. Usa 'tfidf' o 'bow'.")


def default_vectorizer_path(method: str | None = None) -> Path:
    method = (method or VECTORIZER_METHOD).lower()
    if method == "tfidf":
        return TFIDF_VECTORIZER_PATH
    if method == "bow":
        return BOW_VECTORIZER_PATH
    raise ValueError(f"VECTORIZER_METHOD inválido: {method!r}")


def extract_data(
    input_path: Path | str | None = None,
    *,
    apply_quality: bool = True,
    quality_report_path: Path | str | None = None,
) -> pd.DataFrame:
    path = Path(input_path) if input_path else RAW_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {path}")

    df = pd.read_csv(path)
    required = {"texto", "etiqueta"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en CSV: {missing}")

    df = df.dropna(subset=["etiqueta"]).copy()
    if apply_quality:
        df, report = clean_quality_issues(df, text_col="texto", label_col="etiqueta")
        save_quality_report(report, path=quality_report_path or DATA_QUALITY_REPORT_PATH)
        if report["after"]["class_counts"].get("Productivo", 0) == 0 or report["after"][
            "class_counts"
        ].get("Procrastinación", 0) == 0:
            raise ValueError(
                "Tras limpieza de calidad queda una sola clase. Revisa el dataset."
            )
    else:
        blank = df["texto"].isna() | df["texto"].astype(str).str.strip().eq("")
        df = df.loc[~blank].copy()

    df["etiqueta"] = df["etiqueta"].map(normalize_label)
    return df.reset_index(drop=True)


def clean_data(
    df: pd.DataFrame | None = None,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    if df is None:
        df = extract_data()

    nlp = get_spacy_nlp()
    cleaned = df.copy()
    cleaned["texto_limpio"] = cleaned["texto"].astype(str).map(lambda t: clean_text(t, nlp=nlp))
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
    method: str | None = None,
) -> tuple[pd.DataFrame, Any]:
    """Convierte texto limpio a matriz BoW o TF-IDF y persiste vectorizer + CSV."""
    method = (method or VECTORIZER_METHOD).lower()
    if df is None:
        cleaned_path = CLEANED_CSV_PATH
        if not cleaned_path.exists():
            df = clean_data()
        else:
            df = pd.read_csv(cleaned_path)

    if "texto_limpio" not in df.columns:
        df = clean_data(df)

    vec_path = Path(vectorizer_path) if vectorizer_path else default_vectorizer_path(method)
    vec_path.parent.mkdir(parents=True, exist_ok=True)

    if fit:
        vectorizer = build_vectorizer(method)
        matrix = vectorizer.fit_transform(df["texto_limpio"].astype(str))
        joblib.dump({"method": method, "vectorizer": vectorizer}, vec_path)
    else:
        if not vec_path.exists():
            raise FileNotFoundError(f"Vectorizer no encontrado: {vec_path}")
        payload = joblib.load(vec_path)
        if isinstance(payload, dict) and "vectorizer" in payload:
            vectorizer = payload["vectorizer"]
            method = payload.get("method", method)
        else:
            # Compatibilidad con artefactos antiguos (solo el estimador)
            vectorizer = payload
        matrix = vectorizer.transform(df["texto_limpio"].astype(str))

    names = feature_column_names(matrix.shape[1])
    features = pd.DataFrame(matrix.toarray(), columns=names)
    features["etiqueta"] = df["etiqueta"].values

    out = Path(output_path) if output_path else VECTORIZED_CSV_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out, index=False)
    return features, vectorizer


def transform_texts(
    texts: list[str],
    vectorizer_path: Path | str | None = None,
    method: str | None = None,
) -> pd.DataFrame:
    """Transforma textos crudos para inferencia (limpia + vectoriza)."""
    method = (method or VECTORIZER_METHOD).lower()
    vec_path = Path(vectorizer_path) if vectorizer_path else default_vectorizer_path(method)
    if not vec_path.exists():
        raise FileNotFoundError(f"Vectorizer no encontrado: {vec_path}")

    payload = joblib.load(vec_path)
    if isinstance(payload, dict) and "vectorizer" in payload:
        vectorizer = payload["vectorizer"]
    else:
        vectorizer = payload

    nlp = get_spacy_nlp()
    cleaned = [clean_text(t, nlp=nlp) for t in texts]
    matrix = vectorizer.transform(cleaned)
    return pd.DataFrame(matrix.toarray(), columns=feature_column_names(matrix.shape[1]))


def run_nlp_pipeline(
    input_path: Path | str | None = None,
    cleaned_path: Path | str | None = None,
    vectorized_path: Path | str | None = None,
    method: str | None = None,
) -> dict:
    """Ejecuta extract (+calidad) -> clean -> vectorización de punta a punta."""
    method = (method or VECTORIZER_METHOD).lower()
    raw = extract_data(input_path, apply_quality=True)
    cleaned = clean_data(raw, output_path=cleaned_path)
    features, vectorizer = feature_engineering(
        cleaned,
        fit=True,
        output_path=vectorized_path,
        method=method,
    )
    return {
        "n_samples": len(cleaned),
        "n_features": features.shape[1] - 1,
        "method": method,
        "cleaned_path": str(cleaned_path or CLEANED_CSV_PATH),
        "vectorized_path": str(vectorized_path or VECTORIZED_CSV_PATH),
        "vectorizer_path": str(default_vectorizer_path(method)),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "quality_report_path": str(DATA_QUALITY_REPORT_PATH),
    }


if __name__ == "__main__":
    result = run_nlp_pipeline()
    print("NLP pipeline OK:")
    for key, value in result.items():
        print(f"  {key}: {value}")
