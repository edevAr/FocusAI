#!/usr/bin/env bash
# Ejecuta el pipeline completo sin Airflow (útil para desarrollo local).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "==> [1/5] NLP: extract + clean + TF-IDF"
python -m src.nlp.preprocess

echo "==> [2/5] Init SQLite database"
python -m src.database.db

echo "==> [3/5] Train ensemble (PyCaret + MLflow)"
python -m src.training.train_model

echo "==> [4/5] Smoke predict"
python - <<'PY'
from src.training.predict import predict_one
samples = [
    "Completé el módulo, escribí tests unitarios y documenté la API.",
    "Pasé el día en redes sociales y no avancé ninguna tarea.",
]
for s in samples:
    r = predict_one(s)
    print(f"  [{r['prediccion']}] ({(r.get('probabilidad') or 0):.2f}) {s[:60]}")
PY

echo "==> Pipeline local completado"
