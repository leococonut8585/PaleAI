"""add folder table and link to chatsession

Revision ID: d41b5f5727c9
Revises: 427af3a72f29
Create Date: 2025-05-18 10:05:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'd41b5f5727c9'
down_revision: Union[str, None] = '427af3a72f29'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create folders table and link to chat_sessions if not already present."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    # Create folders table only if it doesn't exist
    if 'folders' not in inspector.get_table_names():
        op.create_table(
            'folders',
            sa.Column('id', sa.Integer(), primary_key=True, index=True),
            sa.Column('name', sa.String(), nullable=False, index=True),
            sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id'), nullable=False),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
            sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        )

    # Add folder_id column to chat_sessions if missing
    columns = [c['name'] for c in inspector.get_columns('chat_sessions')]
    if 'folder_id' not in columns:
        with op.batch_alter_table('chat_sessions', schema=None) as batch_op:
            batch_op.add_column(sa.Column('folder_id', sa.Integer(), sa.ForeignKey('folders.id'), nullable=True))


def downgrade() -> None:
    """Drop folder_id column and folders table if they exist."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    columns = [c['name'] for c in inspector.get_columns('chat_sessions')]
    if 'folder_id' in columns:
        with op.batch_alter_table('chat_sessions', schema=None) as batch_op:
            batch_op.drop_column('folder_id')

    if 'folders' in inspector.get_table_names():
        op.drop_table('folders')
