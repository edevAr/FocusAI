#!/usr/bin/env bash
# Levanta MLflow Tracking Server local
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/artifacts/mlruns" "$ROOT/artifacts/mlflow-artifacts"

export MLFLOW_BACKEND_STORE_URI="file://$ROOT/artifacts/mlruns"
export MLFLOW_DEFAULT_ARTIFACT_ROOT="$ROOT/artifacts/mlflow-artifacts"

echo "MLflow UI -> http://127.0.0.1:5000"
exec mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
  --default-artifact-root "$MLFLOW_DEFAULT_ARTIFACT_ROOT"
