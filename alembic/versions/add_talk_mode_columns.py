"""add mode and completion columns to chat_sessions"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'addtalkmode'
down_revision: Union[str, None] = 'update_latest_columns'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add mode and is_complete columns if they don't exist."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = [c['name'] for c in inspector.get_columns('chat_sessions')]
    to_add = []
    if 'mode' not in columns:
        to_add.append(sa.Column('mode', sa.String(), server_default='chat', nullable=False))
    if 'is_complete' not in columns:
        to_add.append(sa.Column('is_complete', sa.Boolean(), server_default=sa.text('1'), nullable=False))
    if to_add:
        with op.batch_alter_table('chat_sessions') as batch_op:
            for col in to_add:
                batch_op.add_column(col)


def downgrade() -> None:
    """Drop mode and is_complete columns if present."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = [c['name'] for c in inspector.get_columns('chat_sessions')]
    to_drop = []
    if 'is_complete' in columns:
        to_drop.append('is_complete')
    if 'mode' in columns:
        to_drop.append('mode')
    if to_drop:
        with op.batch_alter_table('chat_sessions') as batch_op:
            for col in to_drop:
                batch_op.drop_column(col)
