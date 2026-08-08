"""Tests de contrato de la API de serving (Mireya — Deployment & Serving).

Cubren /predict, el flujo de auth heredado y los endpoints de
streamlit-authenticator (/auth/st/*). El modelo de producción y la capa de
base de datos se sustituyen con monkeypatch para probar solo el contrato HTTP.
"""
from __future__ import annotations

import bcrypt
import pytest
from fastapi.testclient import TestClient

from api import main


@pytest.fixture
def client(monkeypatch) -> TestClient:
    # El arranque solo asegura el directorio de la BD; lo neutralizamos.
    monkeypatch.setattr(main, "init_db", lambda: None)
    return TestClient(main.app)


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------


def test_predict_returns_prediction_contract(client, monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "predict_one",
        lambda texto: {
            "prediccion": "Productivo",
            "label_id": 1,
            "probabilidad": 0.91,
            "texto_limpio": "termine informe",
        },
    )
    resp = client.post("/predict", json={"texto": "Hoy terminé el informe"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediccion"] == "Productivo"
    assert body["probabilidad"] == 0.91
    assert "diario_id" not in body


def test_predict_persists_when_usuario_id_present(client, monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "predict_one",
        lambda texto: {"prediccion": "Productivo", "probabilidad": 0.8},
    )
    saved: dict = {}

    def fake_insert(usuario_id, texto, prediccion, probabilidad):
        saved.update(
            usuario_id=usuario_id, texto=texto, prediccion=prediccion, probabilidad=probabilidad
        )
        return {"id": 42}

    monkeypatch.setattr(main, "insert_diario", fake_insert)
    resp = client.post("/predict", json={"texto": "Avancé el pipeline", "usuario_id": 7})
    assert resp.status_code == 200
    assert resp.json()["diario_id"] == 42
    assert saved["usuario_id"] == 7


def test_predict_rejects_blank_text(client) -> None:
    resp = client.post("/predict", json={"texto": ""})
    assert resp.status_code == 422  # pydantic min_length=1


def test_predict_503_when_model_unavailable(client, monkeypatch) -> None:
    def boom(texto):
        raise main.ProductionModelUnavailableError("no production alias")

    monkeypatch.setattr(main, "predict_one", boom)
    resp = client.post("/predict", json={"texto": "algo"})
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# /auth heredado
# ---------------------------------------------------------------------------


def test_legacy_login_ok_and_fail(client, monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "authenticate_user",
        lambda u, p: {"id": 1, "username": u} if p == "good" else None,
    )
    ok = client.post("/auth/login", json={"username": "miya", "password": "good"})
    assert ok.status_code == 200
    bad = client.post("/auth/login", json={"username": "miya", "password": "bad"})
    assert bad.status_code == 401


# ---------------------------------------------------------------------------
# /auth/st/* (streamlit-authenticator)
# ---------------------------------------------------------------------------


def test_st_register_hashes_with_bcrypt(client, monkeypatch) -> None:
    captured: dict = {}

    def fake_registrar(nombre, email, password_hash):
        captured.update(nombre=nombre, email=email, password_hash=password_hash)
        return 5

    monkeypatch.setattr(main, "registrar_usuario", fake_registrar)
    resp = client.post(
        "/auth/st/register",
        json={"username": "miya", "email": "miya@focusai.local", "password": "secret123"},
    )
    assert resp.status_code == 200
    assert resp.json()["id"] == 5
    # El hash almacenado es bcrypt y verifica la contraseña original.
    assert captured["password_hash"].startswith("$2")
    assert bcrypt.checkpw(b"secret123", captured["password_hash"].encode("utf-8"))


def test_st_register_duplicate_returns_400(client, monkeypatch) -> None:
    def boom(nombre, email, password_hash):
        raise ValueError("El usuario ya existe")

    monkeypatch.setattr(main, "registrar_usuario", boom)
    resp = client.post(
        "/auth/st/register",
        json={"username": "dup", "email": "dup@x.com", "password": "secret123"},
    )
    assert resp.status_code == 400


def test_st_credentials_only_includes_bcrypt_users(client, monkeypatch) -> None:
    bcrypt_hash = bcrypt.hashpw(b"x", bcrypt.gensalt()).decode("utf-8")
    monkeypatch.setattr(
        main,
        "listar_usuarios",
        lambda: [
            {"id": 1, "username": "miya", "email": "m@x.com", "password_hash": bcrypt_hash},
            {"id": 2, "username": "legacy", "email": "l@x.com", "password_hash": "sha256hash"},
        ],
    )
    resp = client.get("/auth/st/credentials")
    assert resp.status_code == 200
    usernames = resp.json()["credentials"]["usernames"]
    assert "miya" in usernames
    assert "legacy" not in usernames  # hash heredado (SHA-256) excluido
    assert usernames["miya"]["password"] == bcrypt_hash


def test_st_user_lookup_ok_and_404(client, monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "obtener_usuario_por_username",
        lambda u: {"id": 3, "username": u, "email": "e@x.com", "password_hash": "$2x"},
    )
    ok = client.get("/auth/st/user/miya")
    assert ok.status_code == 200
    assert ok.json() == {"id": 3, "username": "miya", "email": "e@x.com"}

    monkeypatch.setattr(main, "obtener_usuario_por_username", lambda u: None)
    missing = client.get("/auth/st/user/ghost")
    assert missing.status_code == 404
