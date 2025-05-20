"""add mode and completion columns to chat_sessions"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'addtalkmode'
down_revision: Union[str, None] = 'update_latest_columns'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('chat_sessions') as batch_op:
        batch_op.add_column(sa.Column('mode', sa.String(), server_default='chat', nullable=False))
        batch_op.add_column(sa.Column('is_complete', sa.Boolean(), server_default=sa.text('1'), nullable=False))


def downgrade() -> None:
    with op.batch_alter_table('chat_sessions') as batch_op:
        batch_op.drop_column('is_complete')
        batch_op.drop_column('mode')
