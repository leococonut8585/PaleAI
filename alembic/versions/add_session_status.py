"""add status column to chat_sessions"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'add_session_status'
down_revision: Union[str, None] = 'add_profile_fields'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('chat_sessions') as batch_op:
        batch_op.add_column(sa.Column('status', sa.String(), server_default='complete', nullable=False))


def downgrade() -> None:
    with op.batch_alter_table('chat_sessions') as batch_op:
        batch_op.drop_column('status')
