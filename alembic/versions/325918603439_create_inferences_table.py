from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '325918603439'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'inferences',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('response', sa.Text(), nullable=False),
        sa.Column('image_path', sa.String(length=512), nullable=True),
        sa.Column('image_filename', sa.String(length=255), nullable=True),
        sa.Column('image_mime', sa.String(length=64), nullable=True),
        sa.Column('max_new_tokens', sa.Integer(), nullable=False),
        sa.Column('latency_ms', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('inferences', schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f('ix_inferences_created_at'),
            ['created_at'],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table('inferences', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_inferences_created_at'))

    op.drop_table('inferences')
