"""Capa de acceso a SQLite: Usuarios y Diarios."""
from __future__ import annotations

import hashlib
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATABASE_PATH


def _hash_password(password: str, salt: str = "focusai_salt") -> str:
    payload = f"{salt}:{password}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@contextmanager
def get_connection(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    path = Path(db_path) if db_path else DATABASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path | str | None = None) -> Path:
    path = Path(db_path) if db_path else DATABASE_PATH
    with get_connection(path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS usuarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS diarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usuario_id INTEGER NOT NULL,
                texto TEXT NOT NULL,
                prediccion TEXT,
                probabilidad REAL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (usuario_id) REFERENCES usuarios(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_diarios_usuario
                ON diarios(usuario_id);
            CREATE INDEX IF NOT EXISTS idx_diarios_created
                ON diarios(created_at);
            """
        )
    return path


def create_user(
    username: str,
    email: str,
    password: str,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    username = username.strip()
    email = email.strip().lower()
    if not username or not email or not password:
        raise ValueError("username, email y password son obligatorios")

    password_hash = _hash_password(password)
    with get_connection(db_path) as conn:
        try:
            cur = conn.execute(
                """
                INSERT INTO usuarios (username, email, password_hash, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (username, email, password_hash, _utcnow_iso()),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("El usuario o email ya existe") from exc

        return {
            "id": cur.lastrowid,
            "username": username,
            "email": email,
        }


def authenticate_user(
    username: str,
    password: str,
    db_path: Path | str | None = None,
) -> dict[str, Any] | None:
    password_hash = _hash_password(password)
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, username, email, created_at
            FROM usuarios
            WHERE username = ? AND password_hash = ?
            """,
            (username.strip(), password_hash),
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: int, db_path: Path | str | None = None) -> dict[str, Any] | None:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT id, username, email, created_at FROM usuarios WHERE id = ?",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def insert_diario(
    usuario_id: int,
    texto: str,
    prediccion: str | None = None,
    probabilidad: float | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    if not texto or not texto.strip():
        raise ValueError("El texto del diario no puede estar vacío")

    created_at = _utcnow_iso()
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO diarios (usuario_id, texto, prediccion, probabilidad, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (usuario_id, texto.strip(), prediccion, probabilidad, created_at),
        )
        return {
            "id": cur.lastrowid,
            "usuario_id": usuario_id,
            "texto": texto.strip(),
            "prediccion": prediccion,
            "probabilidad": probabilidad,
            "created_at": created_at,
        }


def get_diarios_by_user(
    usuario_id: int,
    limit: int = 100,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
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


def get_procrastination_series(
    usuario_id: int,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Serie histórica para el gráfico de área de procrastinación."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                date(created_at) AS dia,
                SUM(CASE WHEN prediccion = 'Procrastinación' THEN 1 ELSE 0 END) AS procrastinacion,
                SUM(CASE WHEN prediccion = 'Productivo' THEN 1 ELSE 0 END) AS productivo,
                COUNT(*) AS total
            FROM diarios
            WHERE usuario_id = ?
            GROUP BY date(created_at)
            ORDER BY dia ASC
            """,
            (usuario_id,),
        ).fetchall()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    db = init_db()
    print(f"Base de datos lista: {db}")
    try:
        user = create_user("demo", "demo@focusai.local", "demo123")
        print("Usuario demo:", user)
    except ValueError:
        user = authenticate_user("demo", "demo123")
        print("Usuario demo existente:", user)
