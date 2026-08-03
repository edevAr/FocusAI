"""Capa de acceso a SQLite: Usuarios y Diarios.

Cambios v2 (Alembic + Seed):
  - init_db() ya NO crea tablas ni índices: Alembic es el responsable del
    schema. La función solo garantiza que el directorio del archivo exista y
    devuelve el Path, para que el resto del código (DAG, API, tests) no cambie
    su firma.
  - seed_demo_data() inyecta 30 días de historial de demo con entradas
    productivas y de procrastinación variadas, usando insert_diario().
"""
from __future__ import annotations

import hashlib
import random
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATABASE_PATH  # noqa: E402


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
    """Garantiza que el directorio de la base de datos exista.

    A partir de la versión 2, **Alembic** es el responsable exclusivo de
    crear y evolucionar el schema (tablas, índices, foreign keys). Esta
    función se mantiene para que los callers existentes (DAG, API, tests)
    no rompan su interfaz: devuelve el Path de la DB y asegura que el
    directorio padre exista.

    Para aplicar migraciones pendientes ejecuta:
        alembic upgrade head
    """
    path = Path(db_path) if db_path else DATABASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Usuarios
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Diarios
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Seed de datos de demostración
# ---------------------------------------------------------------------------

# Corpus de entradas productivas (variadas en longitud y estilo)
_TEXTOS_PRODUCTIVOS: list[str] = [
    "Completé el módulo de feature engineering y todas las pruebas pasaron en verde.",
    "Revisé el PR de integración con MLflow; dejé comentarios detallados y lo aprobé.",
    "Redacté la documentación de la API REST: endpoints, ejemplos de request y response.",
    "Implementé la validación de esquema con Pydantic en los modelos de entrada.",
    "Refactoricé el pipeline de limpieza de datos; la cobertura de tests subió al 87 %.",
    "Tuve una sesión de pair programming de 2 h con el equipo para resolver el bug del vectorizador.",
    "Configuré los alertas de Grafana para monitorear la latencia del modelo en producción.",
    "Estudié el capítulo de optimización de hiperparámetros con Optuna durante 90 minutos.",
    "Preparé las slides del demo de MLOps para la reunión con stakeholders del viernes.",
    "Automaticé el proceso de generación del reporte de métricas con un script de Python.",
    "Cerré 5 issues del backlog: 3 bugs y 2 mejoras de rendimiento en la API.",
    "Escribí tests de integración para el flujo de autenticación de usuarios.",
    "Configuré el entorno de CI/CD en GitHub Actions para el repositorio de FocusAI.",
    "Analicé los logs de Airflow e identifiqué el cuello de botella en la tarea de extracción.",
    "Completé el curso de Docker avanzado; practiqué multi-stage builds para la imagen de la API.",
]

# Corpus de entradas de procrastinación (variadas en tono y detalle)
_TEXTOS_PROCRASTINACION: list[str] = [
    "Pasé casi toda la mañana revisando Twitter y Reddit sin avanzar nada concreto.",
    "Abrí el IDE pero acabé viendo videos de YouTube sobre machine learning durante 3 horas.",
    "Reorganicé la carpeta de descargas en lugar de terminar el análisis exploratorio pendiente.",
    "Estuve 2 horas leyendo artículos de Medium sin tomar notas ni aplicar nada.",
    "Pospuse la revisión del código de producción; «mañana lo hago» fue mi mantra del día.",
    "Me distraje con notificaciones del móvil cada 10 minutos; no logré entrar en flujo.",
    "Pasé la tarde ajustando el tema del editor de código en lugar de escribir tests.",
    "Tuve una reunión que pudo haber sido un email; el resto del día lo perdí recuperando el foco.",
    "Revisé el correo electrónico más de 20 veces sin responder ninguno de los importantes.",
    "Empecé a configurar un nuevo plugin de Vim y acabé sin programar nada en 4 horas.",
    "Jugué videojuegos durante la jornada laboral justificándolo como 'descanso mental'.",
    "Estuve planeando el sprint en un tablero Notion sin ejecutar ni una sola tarea.",
    "Busqué referencias de diseño en Dribbble durante horas; el wireframe sigue vacío.",
    "Pospuse la llamada con el cliente porque 'no me sentía preparado' sin haberlo intentado.",
    "Pasé la mañana escuchando podcasts de productividad en vez de ser productivo.",
]


def seed_demo_data(
    usuario_id: int,
    db_path: Path | str | None = None,
    days: int = 30,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Inyecta ``days`` días de historial de demo para el usuario dado.

    Genera entre 1 y 3 entradas por día con predicciones y probabilidades
    realistas, distribuidas a lo largo de los últimos ``days`` días.

    Parámetros
    ----------
    usuario_id:
        ID del usuario al que pertenecerán las entradas.
    db_path:
        Path opcional a la base de datos (usa DATABASE_PATH si es None).
    days:
        Número de días hacia atrás para los que se generan entradas (default 30).
    seed:
        Semilla del generador aleatorio para resultados reproducibles.

    Devuelve
    --------
    Lista de diccionarios con las entradas insertadas.
    """
    rng = random.Random(seed)
    inserted: list[dict[str, Any]] = []

    now = datetime.now(timezone.utc)

    # Probabilidades de que un día dado tenga entradas productivas vs. de procrastinación.
    # Se alterna para simular semanas con altibajos realistas.
    for day_offset in range(days, 0, -1):
        day_dt = now - timedelta(days=day_offset)

        # Entre 1 y 3 entradas por día
        n_entries = rng.randint(1, 3)

        for entry_idx in range(n_entries):
            # Distribuir entradas a lo largo del día (hora entre 8 y 21)
            hour = rng.randint(8, 21)
            minute = rng.randint(0, 59)
            entry_dt = day_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
            created_at_iso = entry_dt.isoformat()

            # Tendencia: días impares más productivos, pares más de procrastinación
            # con variación aleatoria para mayor realismo
            base_productive_prob = 0.65 if day_offset % 2 != 0 else 0.40
            es_productivo = rng.random() < base_productive_prob

            if es_productivo:
                texto = rng.choice(_TEXTOS_PRODUCTIVOS)
                prediccion = "Productivo"
                # Probabilidad alta con algo de ruido (0.72 – 0.97)
                probabilidad = round(rng.uniform(0.72, 0.97), 4)
            else:
                texto = rng.choice(_TEXTOS_PROCRASTINACION)
                prediccion = "Procrastinación"
                # Probabilidad alta con algo de ruido (0.68 – 0.95)
                probabilidad = round(rng.uniform(0.68, 0.95), 4)

            # Inserción directa para poder controlar el campo created_at
            # (insert_diario() siempre usa _utcnow_iso(), por lo que usamos
            # la conexión directamente aquí para simular fechas históricas).
            path = Path(db_path) if db_path else DATABASE_PATH
            with get_connection(path) as conn:
                cur = conn.execute(
                    """
                    INSERT INTO diarios (usuario_id, texto, prediccion, probabilidad, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (usuario_id, texto, prediccion, probabilidad, created_at_iso),
                )
                record = {
                    "id": cur.lastrowid,
                    "usuario_id": usuario_id,
                    "texto": texto,
                    "prediccion": prediccion,
                    "probabilidad": probabilidad,
                    "created_at": created_at_iso,
                }
                inserted.append(record)

    print(
        f"[seed_demo_data] {len(inserted)} entradas inyectadas para usuario_id={usuario_id} "
        f"({days} días de historial)."
    )
    return inserted


# ---------------------------------------------------------------------------
# Punto de entrada para pruebas locales
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    db = init_db()
    print(f"Base de datos lista: {db}")

    try:
        user = create_user("demo", "demo@focusai.local", "demo123")
        print("Usuario demo creado:", user)
    except ValueError:
        user = authenticate_user("demo", "demo123")
        print("Usuario demo existente (autenticado):", user)

    if user:
        print("\nInyectando datos de demostración...")
        entries = seed_demo_data(usuario_id=user["id"])
        series = get_procrastination_series(usuario_id=user["id"])
        print(f"\nSerie de procrastinación ({len(series)} días):")
        for row in series:
            print(
                f"  {row['dia']} | Productivo: {row['productivo']:>2} | "
                f"Procrastinación: {row['procrastinacion']:>2} | Total: {row['total']}"
            )
