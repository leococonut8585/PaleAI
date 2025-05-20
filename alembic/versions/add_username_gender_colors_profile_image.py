"""add username and profile fields"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'add_profile_fields'
down_revision: Union[str, None] = 'addtalkmode'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("username", sa.String(length=20), nullable=False),
    )
    op.add_column(
        "users",
        sa.Column("gender", sa.String(), nullable=False, server_default="未回答"),
    )
    op.add_column(
        "users",
        sa.Column("color1", sa.String(length=7), nullable=False, server_default="#000000"),
    )
    op.add_column(
        "users",
        sa.Column("color2", sa.String(length=7), nullable=False, server_default="#ffffff"),
    )
    op.add_column(
        "users",
        sa.Column("profile_image_url", sa.String(), nullable=True),
    )
    op.create_unique_constraint("uq_users_username", "users", ["username"])
    op.create_index(op.f("ix_users_username"), "users", ["username"], unique=True)

def downgrade() -> None:
    op.drop_constraint("uq_users_username", "users", type_="unique")
    op.drop_index("ix_users_username", table_name="users")
    op.drop_column('users', 'profile_image_url')
    op.drop_column('users', 'color2')
    op.drop_column('users', 'color1')
    op.drop_column('users', 'gender')
    op.drop_column('users', 'username')
