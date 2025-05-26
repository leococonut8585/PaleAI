"""Add user_id to chat_messages

Revision ID: 427af3a72f29
Revises: None
Create Date: 2025-05-16 18:16:29.276345
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '427af3a72f29'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add user_id column to chat_messages if not already present."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = [c["name"] for c in inspector.get_columns("chat_messages")]
    if "user_id" not in columns:
        op.add_column("chat_messages", sa.Column("user_id", sa.Integer(), nullable=True))
        if bind.dialect.name != "sqlite":
            op.create_foreign_key(
                "fk_user_id",
                "chat_messages",
                "users",
                ["user_id"],
                ["id"],
            )


def downgrade() -> None:
    """Remove user_id column from chat_messages if present."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = [c["name"] for c in inspector.get_columns("chat_messages")]
    if "user_id" in columns:
        if bind.dialect.name != "sqlite":
            op.drop_constraint("fk_user_id", "chat_messages", type_="foreignkey")
        op.drop_column("chat_messages", "user_id")
