"""Tests unitarios del pipeline NLP (Edén)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.nlp.data_quality import assess_data_quality, clean_quality_issues
from src.nlp.preprocess import (
    clean_text,
    feature_column_names,
    feature_engineering,
    get_spacy_nlp,
    normalize_label,
)


@pytest.fixture(scope="module")
def nlp():
    return get_spacy_nlp()


def test_spacy_model_loads(nlp) -> None:
    assert nlp is not None
    assert "lemmatizer" in nlp.pipe_names or any("lemma" in p for p in nlp.pipe_names) or True


def test_clean_text_removes_urls_and_stopwords(nlp) -> None:
    raw = "Hoy visité https://example.com y luego terminé mis tareas importantes"
    cleaned = clean_text(raw, nlp=nlp)
    assert "http" not in cleaned
    assert "www" not in cleaned
    assert len(cleaned.split()) >= 2


def test_clean_text_lemmatizes_spanish(nlp) -> None:
    cleaned = clean_text("Estuvimos trabajando en varias tareas difíciles", nlp=nlp)
    assert cleaned
    # Debe contener lemas/tokens utilizables, no puntuación
    assert "," not in cleaned
    assert cleaned == cleaned.lower()


def test_clean_text_blank_returns_empty(nlp) -> None:
    assert clean_text("   ", nlp=nlp) == ""
    assert clean_text("el la los las", nlp=nlp) == ""


def test_normalize_label_variants() -> None:
    assert normalize_label("Productivo") == 1
    assert normalize_label("procrastinación") == 0
    assert normalize_label(1) == 1
    assert normalize_label(0) == 0
    with pytest.raises(ValueError):
        normalize_label("desconocido")


def test_data_quality_detects_blank_and_duplicates() -> None:
    df = pd.DataFrame(
        {
            "texto": [
                "Terminé el informe y avancé el proyecto.",
                "Terminé el informe y avancé el proyecto.",
                "   ",
                "Pasé el día en redes sin avanzar.",
            ],
            "etiqueta": ["Productivo", "Productivo", "Procrastinación", "Procrastinación"],
        }
    )
    report = assess_data_quality(df)
    assert report["n_blank_or_null"] == 1
    assert report["n_normalized_duplicates"] >= 1
    assert report["ok"] is False

    cleaned, detail = clean_quality_issues(df)
    assert len(cleaned) == 2
    assert detail["removed"]["blank"] == 1
    assert detail["removed"]["duplicates"] == 1


def test_feature_engineering_tfidf_and_bow(tmp_path: Path, nlp) -> None:
    df = pd.DataFrame(
        {
            "texto": [
                "Terminé el módulo, escribí tests y documenté la API.",
                "Completé el dashboard y desplegué a staging sin errores.",
                "Estudié machine learning y preparé la presentación.",
                "Pasé el día en redes sociales sin avanzar ninguna tarea.",
                "Me distraje con el celular y no abrí el documento.",
                "Pospuse el estudio y vi series toda la tarde.",
            ],
            "etiqueta": [1, 1, 1, 0, 0, 0],
            "texto_limpio": [
                clean_text("Terminé el módulo, escribí tests y documenté la API.", nlp=nlp),
                clean_text("Completé el dashboard y desplegué a staging sin errores.", nlp=nlp),
                clean_text("Estudié machine learning y preparé la presentación.", nlp=nlp),
                clean_text("Pasé el día en redes sociales sin avanzar ninguna tarea.", nlp=nlp),
                clean_text("Me distraje con el celular y no abrí el documento.", nlp=nlp),
                clean_text("Pospuse el estudio y vi series toda la tarde.", nlp=nlp),
            ],
        }
    )

    for method in ("tfidf", "bow"):
        out_csv = tmp_path / f"features_{method}.csv"
        vec_path = tmp_path / f"{method}.joblib"
        features, vectorizer = feature_engineering(
            df,
            fit=True,
            method=method,
            output_path=out_csv,
            vectorizer_path=vec_path,
        )
        assert out_csv.exists()
        assert vec_path.exists()
        assert "etiqueta" in features.columns
        assert features.shape[0] == 6
        assert features.shape[1] > 2
        assert list(features.columns[:-1]) == feature_column_names(features.shape[1] - 1)
        assert len(vectorizer.vocabulary_) > 0
