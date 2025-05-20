"""add star and tags to chat_sessions, position to folders"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'addstarp0'
down_revision: Union[str, None] = 'd41b5f5727c9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    with op.batch_alter_table('chat_sessions') as batch_op:
        batch_op.add_column(sa.Column('starred', sa.Boolean(), server_default=sa.text('0'), nullable=False))
        batch_op.add_column(sa.Column('tags', sa.String(), nullable=True))
    with op.batch_alter_table('folders') as batch_op:
        batch_op.add_column(sa.Column('position', sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('folders') as batch_op:
        batch_op.drop_column('position')
    with op.batch_alter_table('chat_sessions') as batch_op:
        batch_op.drop_column('tags')
        batch_op.drop_column('starred')
