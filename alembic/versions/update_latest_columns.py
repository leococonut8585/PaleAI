"""ensure latest columns added"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'update_latest_columns'
down_revision: Union[str, None] = 'addstarp0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    """Ensure latest columns exist on folders and chat_sessions."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    folder_columns = [c['name'] for c in inspector.get_columns('folders')]
    chat_columns = [c['name'] for c in inspector.get_columns('chat_sessions')]

    if 'position' not in folder_columns:
        op.add_column('folders', sa.Column('position', sa.Integer(), nullable=True))

    if 'starred' not in chat_columns:
        op.add_column('chat_sessions', sa.Column('starred', sa.Boolean(), server_default=sa.text('0'), nullable=False))

    if 'tags' not in chat_columns:
        op.add_column('chat_sessions', sa.Column('tags', sa.String(), nullable=True))

def downgrade() -> None:
    """Remove latest columns if present."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    folder_columns = [c['name'] for c in inspector.get_columns('folders')]
    chat_columns = [c['name'] for c in inspector.get_columns('chat_sessions')]

    if 'tags' in chat_columns:
        op.drop_column('chat_sessions', 'tags')
    if 'starred' in chat_columns:
        op.drop_column('chat_sessions', 'starred')
    if 'position' in folder_columns:
        op.drop_column('folders', 'position')
