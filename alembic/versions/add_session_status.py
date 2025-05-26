"""add status column to chat_sessions"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'add_session_status'
down_revision: Union[str, None] = 'add_profile_fields'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add status column to chat_sessions if missing."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = [c['name'] for c in inspector.get_columns('chat_sessions')]
    if 'status' not in columns:
        with op.batch_alter_table('chat_sessions') as batch_op:
            batch_op.add_column(sa.Column('status', sa.String(), server_default='complete', nullable=False))


def downgrade() -> None:
    """Drop status column from chat_sessions if present."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = [c['name'] for c in inspector.get_columns('chat_sessions')]
    if 'status' in columns:
        with op.batch_alter_table('chat_sessions') as batch_op:
            batch_op.drop_column('status')
