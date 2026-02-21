"""drop context and answer from eval_examples

Revision ID: c4d5e6f7a8b9
Revises: b3c4d5e6f7a8
Create Date: 2026-02-21

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c4d5e6f7a8b9"
down_revision: str | None = "b3c4d5e6f7a8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_column("eval_examples", "context")
    op.drop_column("eval_examples", "answer")


def downgrade() -> None:
    op.add_column(
        "eval_examples",
        sa.Column("context", sa.Text(), nullable=False, server_default=""),
    )
    op.add_column(
        "eval_examples",
        sa.Column("answer", sa.Text(), nullable=False, server_default=""),
    )
