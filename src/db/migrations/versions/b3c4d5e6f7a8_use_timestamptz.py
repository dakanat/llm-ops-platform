"""use timestamptz for all datetime columns

Revision ID: b3c4d5e6f7a8
Revises: a2b3c4d5e6f7
Create Date: 2026-02-19

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b3c4d5e6f7a8"
down_revision: str | None = "a2b3c4d5e6f7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# (table_name, column_name) pairs to alter
_COLUMNS: list[tuple[str, str]] = [
    ("users", "created_at"),
    ("users", "updated_at"),
    ("documents", "created_at"),
    ("documents", "updated_at"),
    ("chunks", "created_at"),
    ("conversations", "created_at"),
    ("conversations", "updated_at"),
    ("messages", "created_at"),
    ("audit_logs", "created_at"),
    ("eval_datasets", "created_at"),
    ("eval_datasets", "updated_at"),
    ("eval_examples", "created_at"),
]


def upgrade() -> None:
    for table, column in _COLUMNS:
        op.alter_column(
            table,
            column,
            type_=sa.DateTime(timezone=True),
            existing_type=sa.DateTime(),
            existing_nullable=False,
        )


def downgrade() -> None:
    for table, column in _COLUMNS:
        op.alter_column(
            table,
            column,
            type_=sa.DateTime(),
            existing_type=sa.DateTime(timezone=True),
            existing_nullable=False,
        )
