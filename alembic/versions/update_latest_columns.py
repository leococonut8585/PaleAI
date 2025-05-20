"""ensure latest columns added"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'update_latest_columns'
down_revision: Union[str, None] = 'addstarp0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    op.add_column('folders', sa.Column('position', sa.Integer(), nullable=True))
    op.add_column('chat_sessions', sa.Column('starred', sa.Boolean(), server_default=sa.text('0'), nullable=False))
    op.add_column('chat_sessions', sa.Column('tags', sa.String(), nullable=True))

def downgrade() -> None:
    op.drop_column('chat_sessions', 'tags')
    op.drop_column('chat_sessions', 'starred')
    op.drop_column('folders', 'position')
