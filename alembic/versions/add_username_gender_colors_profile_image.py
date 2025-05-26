"""add username and profile fields"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'add_profile_fields'
down_revision: Union[str, None] = 'addtalkmode'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    """Add profile related columns if they don't exist."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = [c['name'] for c in inspector.get_columns('users')]

    if 'username' not in columns:
        op.add_column(
            "users",
            sa.Column("username", sa.String(length=20), nullable=False),
        )
        op.create_unique_constraint("uq_users_username", "users", ["username"])
        op.create_index(op.f("ix_users_username"), "users", ["username"], unique=True)

    if 'gender' not in columns:
        op.add_column(
            "users",
            sa.Column("gender", sa.String(), nullable=False, server_default="未回答"),
        )

    if 'color1' not in columns:
        op.add_column(
            "users",
            sa.Column("color1", sa.String(length=7), nullable=False, server_default="#000000"),
        )

    if 'color2' not in columns:
        op.add_column(
            "users",
            sa.Column("color2", sa.String(length=7), nullable=False, server_default="#ffffff"),
        )

    if 'profile_image_url' not in columns:
        op.add_column(
            "users",
            sa.Column("profile_image_url", sa.String(), nullable=True),
        )

def downgrade() -> None:
    """Remove profile related columns if present."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = [c['name'] for c in inspector.get_columns('users')]

    if 'username' in columns:
        op.drop_constraint("uq_users_username", "users", type_="unique")
        op.drop_index("ix_users_username", table_name="users")
        op.drop_column('users', 'username')

    if 'gender' in columns:
        op.drop_column('users', 'gender')

    if 'color1' in columns:
        op.drop_column('users', 'color1')

    if 'color2' in columns:
        op.drop_column('users', 'color2')

    if 'profile_image_url' in columns:
        op.drop_column('users', 'profile_image_url')
