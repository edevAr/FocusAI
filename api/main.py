"""API FastAPI para servir el clasificador de productividad."""
from __future__ import annotations

import sqlite3
import sys
from urllib.error import URLError
from urllib.request import urlopen
from pathlib import Path
from typing import Optional

import bcrypt
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.db import (
    authenticate_user,
    create_user,
    get_diarios_by_user,
    get_procrastination_series,
    init_db,
    insert_diario,
)
from src.database.crud import (
    listar_usuarios,
    obtener_usuario_por_username,
    registrar_usuario,
)
from src.training.predict import predict_one, predict_texts
from src.training.registry import ProductionModelUnavailableError, get_production_model
from config.settings import DATABASE_PATH, MLFLOW_TRACKING_URI

app = FastAPI(
    title="FocusAI API",
    description="Clasificador Productivo vs Procrastinación",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    texto: str = Field(..., min_length=1, description="Entrada de diario")
    usuario_id: Optional[int] = Field(None, description="Si se envía, se guarda en DB")


class BatchPredictRequest(BaseModel):
    textos: list[str] = Field(..., min_length=1)


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3)
    email: str = Field(..., min_length=5)
    password: str = Field(..., min_length=4)


class LoginRequest(BaseModel):
    username: str
    password: str


class StAuthRegisterRequest(BaseModel):
    """Registro para el flujo de streamlit-authenticator (hash bcrypt)."""

    username: str = Field(..., min_length=3)
    email: str = Field(..., min_length=5)
    name: Optional[str] = Field(None, description="Nombre para mostrar")
    password: str = Field(..., min_length=4)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


def _readiness_components() -> dict[str, str]:
    components: dict[str, str] = {}
    try:
        with sqlite3.connect(DATABASE_PATH) as connection:
            connection.execute("SELECT version_num FROM alembic_version").fetchone()
            connection.execute("SELECT 1 FROM usuarios LIMIT 1").fetchone()
    except sqlite3.Error as exc:
        components["schema"] = f"unavailable: {exc}"

    try:
        with urlopen(f"{MLFLOW_TRACKING_URI.rstrip('/')}/health", timeout=2) as response:
            if response.status != 200:
                components["mlflow"] = f"unavailable: HTTP {response.status}"
    except (URLError, OSError) as exc:
        components["mlflow"] = f"unavailable: {exc.reason if isinstance(exc, URLError) else exc}"

    try:
        get_production_model()
    except ProductionModelUnavailableError as exc:
        components["model"] = f"unavailable: {exc}"
    return components


@app.get("/health")
@app.get("/health/live")
def health_live():
    return {"status": "live", "service": "focusai-api"}


@app.get("/health/ready")
def health_ready():
    causes = _readiness_components()
    if causes:
        raise HTTPException(status_code=503, detail={"status": "degraded", "causes": causes})
    return {"status": "ready", "service": "focusai-api"}


def _quality_status(identity: dict[str, str]) -> dict[str, object]:
    """Read candidate gate evidence from MLflow; this never changes aliases."""
    unavailable = {
        "eligible": False,
        "accuracy": None,
        "min_accuracy": None,
        "f1": None,
        "min_f1": None,
        "warnings": ["Quality-gate evidence is unavailable."],
    }
    try:
        version = MlflowClient(MLFLOW_TRACKING_URI).get_model_version(
            identity["name"], identity["version"]
        )
        tags = MlflowClient(MLFLOW_TRACKING_URI).get_run(version.run_id).data.tags
    except Exception as exc:  # MLflow can be reachable while metadata is unavailable.
        return {**unavailable, "warnings": [f"Quality-gate evidence is unavailable: {exc}"]}

    def value(key: str) -> float | None:
        try:
            return float(tags[key])
        except (KeyError, TypeError, ValueError):
            return None

    warnings = tags.get("quality.warnings", "none")
    return {
        "eligible": tags.get("quality.eligible") == "true",
        "accuracy": value("quality.accuracy"),
        "min_accuracy": value("quality.min_accuracy"),
        "f1": value("quality.f1"),
        "min_f1": value("quality.min_f1"),
        "warnings": [] if warnings == "none" else warnings.split(" | "),
    }


@app.get("/mlops/status")
def mlops_status():
    """Expose observed readiness and MLflow evidence without promotion controls."""
    causes = _readiness_components()
    production: dict[str, str] | None = None
    try:
        production = get_production_model().identity.as_dict()
    except ProductionModelUnavailableError:
        pass
    quality = _quality_status(production) if production else {
        "eligible": False,
        "accuracy": None,
        "min_accuracy": None,
        "f1": None,
        "min_f1": None,
        "warnings": ["Production alias is unavailable; quality gates cannot be evaluated."],
    }
    return {
        "readiness": {"status": "degraded", "causes": causes} if causes else {"status": "ready"},
        "production": production,
        "quality": quality,
        "checklist": {
            "production_alias": production is not None,
            "quality_gates": quality["eligible"],
        },
        "authority": "Compare runs and assign aliases through native MLflow UI/API; FocusAI is read-only.",
    }


@app.post("/predict")
def predict(payload: PredictRequest):
    try:
        result = predict_one(payload.texto)
    except ProductionModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Error de predicción: {exc}") from exc

    saved = None
    if payload.usuario_id is not None:
        saved = insert_diario(
            usuario_id=payload.usuario_id,
            texto=payload.texto,
            prediccion=result["prediccion"],
            probabilidad=result.get("probabilidad"),
        )
        result["diario_id"] = saved["id"]

    return result


@app.post("/predict/batch")
def predict_batch(payload: BatchPredictRequest):
    try:
        return {"predictions": predict_texts(payload.textos)}
    except ProductionModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/auth/register")
def register(payload: RegisterRequest):
    try:
        user = create_user(payload.username, payload.email, payload.password)
        return {"message": "Usuario creado", "user": user}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/auth/login")
def login(payload: LoginRequest):
    user = authenticate_user(payload.username, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Credenciales inválidas")
    return {"message": "Login OK", "user": user}


@app.post("/auth/st/register")
def st_register(payload: StAuthRegisterRequest):
    """Registra un usuario con hash bcrypt compatible con streamlit-authenticator."""
    password_hash = bcrypt.hashpw(payload.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    try:
        user_id = registrar_usuario(
            nombre=payload.username,
            email=payload.email,
            password_hash=password_hash,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"message": "Usuario creado", "id": user_id, "username": payload.username.strip()}


@app.get("/auth/st/credentials")
def st_credentials():
    """Devuelve credenciales en el formato que consume streamlit-authenticator.

    Solo incluye usuarios con hash bcrypt (prefijo ``$2``) para evitar que la
    verificación de la librería falle con hashes heredados (SHA-256).
    """
    usernames: dict[str, dict[str, str]] = {}
    for user in listar_usuarios():
        password_hash = user.get("password_hash") or ""
        if not password_hash.startswith("$2"):
            continue
        usernames[user["username"]] = {
            "email": user["email"],
            "name": user["username"],
            "password": password_hash,
        }
    return {"credentials": {"usernames": usernames}}


@app.get("/auth/st/user/{username}")
def st_user(username: str):
    """Devuelve datos públicos del usuario (id, username, email) por username."""
    user = obtener_usuario_por_username(username)
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return {"id": user["id"], "username": user["username"], "email": user["email"]}


@app.get("/users/{usuario_id}/diarios")
def list_diarios(usuario_id: int, limit: int = 100):
    return {"diarios": get_diarios_by_user(usuario_id, limit=limit)}


@app.get("/users/{usuario_id}/procrastination-series")
def procrastination_series(usuario_id: int):
    return {"series": get_procrastination_series(usuario_id)}
