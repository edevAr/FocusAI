"""Revisión inicial de Alembic: crea las tablas `usuarios` y `diarios`.

Generado manualmente para FocusAI.
    ID de revisión : 0001
    Revisión previa: None  (primera migración)
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# ---------------------------------------------------------------------------
# Metadatos de la revisión (usados por Alembic para encadenar migraciones)
# ---------------------------------------------------------------------------
revision: str = "0001"
down_revision: str | None = None
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


# ---------------------------------------------------------------------------
# upgrade: aplica la migración  →  alembic upgrade head
# ---------------------------------------------------------------------------


def upgrade() -> None:
    # ------------------------------------------------------------------
    # Tabla: usuarios
    # ------------------------------------------------------------------
    op.create_table(
        "usuarios",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("username", sa.Text(), nullable=False),
        sa.Column("email", sa.Text(), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.Text(),
            nullable=False,
            server_default=sa.text("(datetime('now'))"),
        ),
        sa.UniqueConstraint("username", name="uq_usuarios_username"),
        sa.UniqueConstraint("email", name="uq_usuarios_email"),
    )

    # ------------------------------------------------------------------
    # Tabla: diarios
    # ------------------------------------------------------------------
    op.create_table(
        "diarios",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("usuario_id", sa.Integer(), nullable=False),
        sa.Column("texto", sa.Text(), nullable=False),
        sa.Column("prediccion", sa.Text(), nullable=True),
        sa.Column("probabilidad", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.Text(),
            nullable=False,
            server_default=sa.text("(datetime('now'))"),
        ),
        sa.ForeignKeyConstraint(
            ["usuario_id"],
            ["usuarios.id"],
            name="fk_diarios_usuario_id",
            ondelete="CASCADE",
        ),
    )

    # ------------------------------------------------------------------
    # Índices sobre `diarios`
    # ------------------------------------------------------------------
    op.create_index(
        index_name="idx_diarios_usuario",
        table_name="diarios",
        columns=["usuario_id"],
        unique=False,
    )
    op.create_index(
        index_name="idx_diarios_created",
        table_name="diarios",
        columns=["created_at"],
        unique=False,
    )


# ---------------------------------------------------------------------------
# downgrade: revierte la migración  →  alembic downgrade -1
# ---------------------------------------------------------------------------


def downgrade() -> None:
    op.drop_index("idx_diarios_created", table_name="diarios")
    op.drop_index("idx_diarios_usuario", table_name="diarios")
    op.drop_table("diarios")
    op.drop_table("usuarios")
