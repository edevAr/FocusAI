"""CRUD de alto nivel para Usuarios y Diarios — puente para la interfaz gráfica.

Este módulo expone funciones limpias y seguras para que la capa de
presentación (Streamlit / FastAPI de Mireya) interactúe con la base
de datos sin escribir SQL crudo.

Reutiliza la conexión y helpers de ``src.database.db`` sin modificar
migraciones, DAG, ni URLs de conexión existentes.

Funciones principales:
    - registrar_usuario(nombre, email, password_hash) → int
    - obtener_usuario_por_email(email) → dict | None
    - guardar_diario(usuario_id, texto, etiqueta_predicha, probabilidad?) → int
    - obtener_historial_diarios(usuario_id, limit?) → list[dict]
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Asegurar que el proyecto raíz esté en sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.db import get_connection  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow_iso() -> str:
    """Timestamp UTC en formato ISO-8601."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# CRUD — Usuarios
# ---------------------------------------------------------------------------


def registrar_usuario(
    nombre: str,
    email: str,
    password_hash: str,
    db_path: Path | str | None = None,
) -> int:
    """Registra un nuevo usuario y retorna su ID.

    Parámetros
    ----------
    nombre : str
        Nombre de usuario (campo ``username`` en la tabla ``usuarios``).
    email : str
        Correo electrónico (único en la tabla).
    password_hash : str
        Hash de la contraseña (SHA-256 con salt, generado por la UI o API).
    db_path : Path | str | None
        Ruta opcional a la base de datos; usa ``DATABASE_PATH`` por defecto.

    Retorna
    -------
    int
        ID del usuario recién creado.

    Lanza
    -----
    ValueError
        Si los campos están vacíos o el usuario/email ya existe.
    """
    nombre = nombre.strip()
    email = email.strip().lower()
    if not nombre or not email or not password_hash:
        raise ValueError("nombre, email y password_hash son obligatorios")

    try:
        with get_connection(db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO usuarios (username, email, password_hash, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (nombre, email, password_hash, _utcnow_iso()),
            )
            return cur.lastrowid  # type: ignore[return-value]
    except sqlite3.IntegrityError as exc:
        raise ValueError(
            f"El usuario '{nombre}' o email '{email}' ya existe en la base de datos."
        ) from exc


def obtener_usuario_por_email(
    email: str,
    db_path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Busca un usuario por email (útil para login).

    Parámetros
    ----------
    email : str
        Correo electrónico del usuario a buscar.

    Retorna
    -------
    dict | None
        Diccionario con ``{id, username, email, password_hash, created_at}``
        si se encuentra; ``None`` si no existe.
    """
    email = email.strip().lower()
    if not email:
        return None

    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, username, email, password_hash, created_at
            FROM usuarios
            WHERE email = ?
            """,
            (email,),
        ).fetchone()

    return dict(row) if row else None


def obtener_usuario_por_username(
    username: str,
    db_path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Busca un usuario por nombre de usuario (útil para streamlit-authenticator).

    Retorna
    -------
    dict | None
        ``{id, username, email, password_hash, created_at}`` o ``None``.
    """
    username = username.strip()
    if not username:
        return None

    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, username, email, password_hash, created_at
            FROM usuarios
            WHERE username = ?
            """,
            (username,),
        ).fetchone()

    return dict(row) if row else None


def listar_usuarios(
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Lista todos los usuarios registrados.

    Útil para construir el diccionario de credenciales que consume
    ``streamlit-authenticator`` (usuarios + hash de contraseña + email).

    Retorna
    -------
    list[dict]
        Lista de ``{id, username, email, password_hash, created_at}``.
    """
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, username, email, password_hash, created_at
            FROM usuarios
            ORDER BY id ASC
            """
        ).fetchall()

    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# CRUD — Diarios
# ---------------------------------------------------------------------------


def guardar_diario(
    usuario_id: int,
    texto: str,
    etiqueta_predicha: str,
    probabilidad: float | None = None,
    db_path: Path | str | None = None,
) -> int:
    """Guarda una entrada de diario con su predicción y retorna su ID.

    Parámetros
    ----------
    usuario_id : int
        ID del usuario propietario del diario.
    texto : str
        Texto de la entrada del diario.
    etiqueta_predicha : str
        Resultado del modelo: ``"Productivo"`` o ``"Procrastinación"``.
    probabilidad : float | None
        Confianza del modelo (0.0 – 1.0). Opcional.

    Retorna
    -------
    int
        ID de la entrada de diario recién creada.

    Lanza
    -----
    ValueError
        Si el texto está vacío.
    sqlite3.IntegrityError
        Si ``usuario_id`` no existe (violación de FK).
    """
    if not texto or not texto.strip():
        raise ValueError("El texto del diario no puede estar vacío")

    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO diarios (usuario_id, texto, prediccion, probabilidad, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (usuario_id, texto.strip(), etiqueta_predicha, probabilidad, _utcnow_iso()),
        )
        return cur.lastrowid  # type: ignore[return-value]


def obtener_historial_diarios(
    usuario_id: int,
    limit: int = 200,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Retorna todas las entradas de diario de un usuario, ordenadas por fecha.

    Parámetros
    ----------
    usuario_id : int
        ID del usuario.
    limit : int
        Máximo de registros a retornar (default 200).

    Retorna
    -------
    list[dict]
        Lista de diccionarios con
        ``{id, usuario_id, texto, prediccion, probabilidad, created_at}``,
        ordenados de más antiguo a más reciente.
    """
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, usuario_id, texto, prediccion, probabilidad, created_at
            FROM diarios
            WHERE usuario_id = ?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (usuario_id, limit),
        ).fetchall()

    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Punto de entrada para pruebas rápidas
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.database.db import _hash_password, init_db

    db = init_db()
    print(f"Base de datos: {db}\n")

    # --- Probar registro ---
    try:
        uid = registrar_usuario("mireya_test", "mireya@focusai.local", _hash_password("test123"))
        print(f"✅ Usuario registrado con ID: {uid}")
    except ValueError as e:
        print(f"⚠️  {e}")

    # --- Probar búsqueda por email ---
    user = obtener_usuario_por_email("mireya@focusai.local")
    if user:
        print(f"✅ Usuario encontrado: {user['username']} (ID {user['id']})")

        # --- Probar guardar diario ---
        did = guardar_diario(user["id"], "Completé la integración del CRUD.", "Productivo", 0.92)
        print(f"✅ Diario guardado con ID: {did}")

        # --- Probar historial ---
        historial = obtener_historial_diarios(user["id"])
        print(f"✅ Historial: {len(historial)} entradas")
        for entry in historial[-3:]:
            print(f"   [{entry['created_at'][:10]}] {entry['prediccion']}: {entry['texto'][:50]}...")
    else:
        print("❌ Usuario no encontrado")
