"""add star and tags to chat_sessions, position to folders"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'addstarp0'
down_revision: Union[str, None] = 'd41b5f5727c9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    """Add starred, tags and position columns if missing."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    chat_columns = [c['name'] for c in inspector.get_columns('chat_sessions')]
    folder_columns = [c['name'] for c in inspector.get_columns('folders')]

    if 'starred' not in chat_columns or 'tags' not in chat_columns:
        with op.batch_alter_table('chat_sessions') as batch_op:
            if 'starred' not in chat_columns:
                batch_op.add_column(sa.Column('starred', sa.Boolean(), server_default=sa.text('0'), nullable=False))
            if 'tags' not in chat_columns:
                batch_op.add_column(sa.Column('tags', sa.String(), nullable=True))

    if 'position' not in folder_columns:
        with op.batch_alter_table('folders') as batch_op:
            batch_op.add_column(sa.Column('position', sa.Integer(), nullable=True))


def downgrade() -> None:
    """Drop added columns if they exist."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    chat_columns = [c['name'] for c in inspector.get_columns('chat_sessions')]
    folder_columns = [c['name'] for c in inspector.get_columns('folders')]

    if 'position' in folder_columns:
        with op.batch_alter_table('folders') as batch_op:
            batch_op.drop_column('position')

    drops = []
    if 'tags' in chat_columns:
        drops.append('tags')
    if 'starred' in chat_columns:
        drops.append('starred')
    if drops:
        with op.batch_alter_table('chat_sessions') as batch_op:
            for col in drops:
                batch_op.drop_column(col)
