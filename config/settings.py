"""Configuración central del proyecto FocusAI."""
import os
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
MODEL_PATH = DATA_MODELS_DIR / "productivity_classifier"
METRICS_PATH = DATA_PROCESSED_DIR / "metrics.json"
HOLDOUT_METRICS_PATH = DATA_PROCESSED_DIR / "holdout_metrics.json"
PER_CLASS_METRICS_PATH = DATA_PROCESSED_DIR / "per_class_metrics.json"
TUNING_RESULTS_PATH = DATA_MODELS_DIR / "tuning_results.csv"
HOLDOUT_TEST_SIZE = 0.2
TUNING_ITERATIONS = 20
CALIBRATION_METHOD = "sigmoid"  # "sigmoid" (Platt) o "isotonic" (requiere mas datos)

# Base de datos
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", PROJECT_ROOT / "data" / "database.db"))

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = "productivity_classifier"
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "productivity_ensemble")
MLFLOW_BACKEND_STORE = str(ARTIFACTS_DIR / "mlruns")
MLFLOW_STAGING_ALIAS = os.getenv("MLFLOW_STAGING_ALIAS", "staging")
MLFLOW_PRODUCTION_ALIAS = os.getenv("MLFLOW_PRODUCTION_ALIAS", "production")
QUALITY_MIN_ACCURACY = float(os.getenv("QUALITY_MIN_ACCURACY", "0.75"))
QUALITY_MIN_F1 = float(os.getenv("QUALITY_MIN_F1", "0.75"))
MODEL_CACHE_TTL_SECONDS = int(os.getenv("MODEL_CACHE_TTL_SECONDS", "60"))

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
# "tfidf" (default) o "bow"
VECTORIZER_METHOD = os.getenv("VECTORIZER_METHOD", "tfidf").strip().lower()
SPACY_MODEL = os.getenv("SPACY_MODEL", "es_core_news_sm")
MIN_SAMPLES_PER_CLASS = int(os.getenv("MIN_SAMPLES_PER_CLASS", "50"))
MAX_CLASS_IMBALANCE_RATIO = float(os.getenv("MAX_CLASS_IMBALANCE_RATIO", "1.5"))
DATA_QUALITY_REPORT_PATH = DATA_PROCESSED_DIR / "data_quality_report.json"
VECTORIZER_COMPARISON_CSV = DATA_PROCESSED_DIR / "vectorizer_comparison.csv"
VECTORIZER_COMPARISON_MD = PROJECT_ROOT / "docs" / "bow_vs_tfidf.md"
DATA_VERSIONS_DIR = PROJECT_ROOT / "data" / "versions"
DATASET_VERSION = os.getenv("DATASET_VERSION", "v1.1.0")
TFIDF_VECTORIZER_PATH = DATA_MODELS_DIR / "tfidf_vectorizer.joblib"
BOW_VECTORIZER_PATH = DATA_MODELS_DIR / "bow_vectorizer.joblib"
FEATURE_COL_PREFIX = "feat_"
RANDOM_STATE = 42
CV_FOLDS = 5

# Vectorizer activo según método
VECTORIZER_PATH = TFIDF_VECTORIZER_PATH if VECTORIZER_METHOD == "tfidf" else BOW_VECTORIZER_PATH

# API
API_HOST = "0.0.0.0"
API_PORT = 8000

for path in (DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_MODELS_DIR, ARTIFACTS_DIR, DATA_VERSIONS_DIR):
    path.mkdir(parents=True, exist_ok=True)
