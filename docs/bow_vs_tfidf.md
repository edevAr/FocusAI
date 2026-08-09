# Comparación BoW vs TF-IDF

Experimento reproducible del módulo NLP (Edén — FocusAI).

## Setup

- Corpus: `data/raw/journal_entries.csv` tras pipeline de calidad
- Limpieza: NLTK tokenizer + stopwords + lematización spaCy (`es_core_news_sm`)
- Clasificador proxy: RandomForest (200 árboles, `class_weight=balanced`)
- Validación: Stratified K-Fold (`k=5`, `random_state=42`)
- Métricas: Accuracy y F1 weighted

## Resultados

| method | n_samples | n_features | vocab | Accuracy CV | F1 CV |
|--------|-----------|------------|-------|-------------|-------|
| tfidf | 256 | 1000 | 1000 | 0.9100 ± 0.0321 | 0.9098 ± 0.0322 |
| bow | 256 | 1000 | 1000 | 0.8985 ± 0.0335 | 0.8981 ± 0.0337 |

## Conclusión

- **Ganador por F1:** `tfidf` → se recomienda **TF-IDF** como vectorizer por defecto.
- Default actual del proyecto (`VECTORIZER_METHOD`): `tfidf`.
- CSV detallado: `data/processed/vectorizer_comparison.csv`.

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
