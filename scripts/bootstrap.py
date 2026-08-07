"""Idempotently create the demo account and its baseline journal entries."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.db import create_user, get_connection, init_db, seed_demo_data


def bootstrap() -> None:
    """Seed only an empty demo account; migration failures intentionally propagate."""
    init_db()
    with get_connection() as connection:
        row = connection.execute(
            "SELECT id FROM usuarios WHERE username = ?", ("demo",)
        ).fetchone()

    if row is None:
        user_id = create_user("demo", "demo@focusai.local", "demo123")["id"]
        print("Created demo user.")
    else:
        user_id = row["id"]
        print("Demo user already exists.")

    with get_connection() as connection:
        entry_count = connection.execute(
            "SELECT COUNT(*) FROM diarios WHERE usuario_id = ?", (user_id,)
        ).fetchone()[0]
    if entry_count == 0:
        seed_demo_data(user_id, seed=42)
        print("Seeded deterministic demo journal entries.")
    else:
        print(f"Demo journal already contains {entry_count} entries; no duplicates added.")


if __name__ == "__main__":
    bootstrap()
