"""Add nullable model and provider usage details to existing query logs.

Revision ID: c72d6a91e405
Revises: b6f3d92e1a74
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "c72d6a91e405"
down_revision = "b6f3d92e1a74"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("query_logs", sa.Column("model", sa.String(255), nullable=True))
    op.add_column(
        "query_logs", sa.Column("token_usage", postgresql.JSONB(none_as_null=True), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("query_logs", "token_usage")
    op.drop_column("query_logs", "model")
