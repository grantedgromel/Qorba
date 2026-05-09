"""initial schema: users, funds, analyses, benchmarks, peer_groups, ingestion_results

Revision ID: 0001_initial
Revises:
Create Date: 2026-05-09 00:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def _json_type():
    return JSONB().with_variant(sa.JSON(), "sqlite")


def _uuid():
    return UUID(as_uuid=True)


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", _uuid(), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_index("ix_users_email", "users", ["email"])

    op.create_table(
        "funds",
        sa.Column("id", _uuid(), primary_key=True),
        sa.Column(
            "user_id",
            _uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("series", _json_type(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_funds_user_id", "funds", ["user_id"])

    op.create_table(
        "benchmarks",
        sa.Column("id", _uuid(), primary_key=True),
        sa.Column(
            "user_id",
            _uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("code", sa.String(64), nullable=True),
        sa.Column("provider", sa.String(32), nullable=False, server_default="user"),
        sa.Column("series", _json_type(), nullable=False),
        sa.Column("is_user_uploaded", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_benchmarks_user_id", "benchmarks", ["user_id"])
    op.create_index("ix_benchmarks_code", "benchmarks", ["code"])

    op.create_table(
        "peer_groups",
        sa.Column("id", _uuid(), primary_key=True),
        sa.Column(
            "user_id",
            _uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("members", _json_type(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_peer_groups_user_id", "peer_groups", ["user_id"])

    op.create_table(
        "analyses",
        sa.Column("id", _uuid(), primary_key=True),
        sa.Column(
            "user_id",
            _uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "fund_id",
            _uuid(),
            sa.ForeignKey("funds.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("selection", _json_type(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_analyses_user_id", "analyses", ["user_id"])

    op.create_table(
        "ingestion_results",
        sa.Column("id", _uuid(), primary_key=True),
        sa.Column(
            "user_id",
            _uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("payload", _json_type(), nullable=False),
        sa.Column(
            "confirmed_fund_id",
            _uuid(),
            sa.ForeignKey("funds.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_ingestion_results_user_id", "ingestion_results", ["user_id"])
    op.create_index("ix_ingestion_results_created_at", "ingestion_results", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_ingestion_results_created_at", table_name="ingestion_results")
    op.drop_index("ix_ingestion_results_user_id", table_name="ingestion_results")
    op.drop_table("ingestion_results")
    op.drop_index("ix_analyses_user_id", table_name="analyses")
    op.drop_table("analyses")
    op.drop_index("ix_peer_groups_user_id", table_name="peer_groups")
    op.drop_table("peer_groups")
    op.drop_index("ix_benchmarks_code", table_name="benchmarks")
    op.drop_index("ix_benchmarks_user_id", table_name="benchmarks")
    op.drop_table("benchmarks")
    op.drop_index("ix_funds_user_id", table_name="funds")
    op.drop_table("funds")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
