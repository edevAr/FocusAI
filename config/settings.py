"""Configuración central del proyecto FocusAI."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Datos
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_MODELS_DIR = PROJECT_ROOT / "data" / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RAW_CSV_PATH = DATA_RAW_DIR / "journal_entries.csv"
CLEANED_CSV_PATH = DATA_PROCESSED_DIR / "cleaned_entries.csv"
VECTORIZED_CSV_PATH = DATA_PROCESSED_DIR / "vectorized_features.csv"
VECTORIZER_PATH = DATA_MODELS_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = DATA_MODELS_DIR / "productivity_classifier"
METRICS_PATH = DATA_PROCESSED_DIR / "metrics.json"
HOLDOUT_METRICS_PATH = DATA_PROCESSED_DIR / "holdout_metrics.json"
PER_CLASS_METRICS_PATH = DATA_PROCESSED_DIR / "per_class_metrics.json"
TUNING_RESULTS_PATH = DATA_MODELS_DIR / "tuning_results.csv"
HOLDOUT_TEST_SIZE = 0.2
TUNING_ITERATIONS = 20
CALIBRATION_METHOD = "sigmoid"  # "sigmoid" (Platt) o "isotonic" (requiere mas datos)

# Base de datos
DATABASE_PATH = PROJECT_ROOT / "data" / "database.db"

# MLflow
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "productivity_classifier"
MLFLOW_MODEL_NAME = "productivity_ensemble"
MLFLOW_BACKEND_STORE = str(ARTIFACTS_DIR / "mlruns")

# Etiquetas
LABEL_PRODUCTIVE = "Productivo"
LABEL_PROCRASTINATION = "Procrastinación"
LABEL_MAP = {
    LABEL_PRODUCTIVE: 1,
    LABEL_PROCRASTINATION: 0,
    "productivo": 1,
    "procrastinación": 0,
    "procrastinacion": 0,
    1: 1,
    0: 0,
}

# NLP
TFIDF_MAX_FEATURES = 1000
TFIDF_NGRAM_RANGE = (1, 2)
RANDOM_STATE = 42
CV_FOLDS = 5

# API
API_HOST = "0.0.0.0"
API_PORT = 8000

for path in (DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_MODELS_DIR, ARTIFACTS_DIR):
    path.mkdir(parents=True, exist_ok=True)
