"""Shared chat rate limit for serverless replicas.

Revision ID: d18e42c63b90
Revises: c72d6a91e405
"""

import sqlalchemy as sa
from alembic import op

revision = "d18e42c63b90"
down_revision = "c72d6a91e405"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "chat_request_attempts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_chat_attempt_user_created", "chat_request_attempts", ["user_id", "created_at"]
    )
    op.create_index("ix_chat_attempt_created", "chat_request_attempts", ["created_at"])


def downgrade():
    op.drop_table("chat_request_attempts")
