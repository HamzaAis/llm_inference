from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'f2bf70de50f2'
down_revision: Union[str, Sequence[str], None] = '325918603439'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('inferences', schema=None) as batch_op:
        batch_op.add_column(sa.Column('query', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('images', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('output', sa.Text(), nullable=False, server_default=''))
        batch_op.add_column(sa.Column('guided_json', sa.Text(), nullable=True))

    op.execute("UPDATE inferences SET query = prompt, output = response")

    with op.batch_alter_table('inferences', schema=None) as batch_op:
        batch_op.drop_column('prompt')
        batch_op.drop_column('response')
        batch_op.drop_column('image_path')
        batch_op.drop_column('image_filename')
        batch_op.drop_column('image_mime')
        batch_op.drop_column('max_new_tokens')


def downgrade() -> None:
    with op.batch_alter_table('inferences', schema=None) as batch_op:
        batch_op.add_column(sa.Column('prompt', sa.Text(), nullable=False, server_default=''))
        batch_op.add_column(sa.Column('response', sa.Text(), nullable=False, server_default=''))
        batch_op.add_column(sa.Column('image_path', sa.String(512), nullable=True))
        batch_op.add_column(sa.Column('image_filename', sa.String(255), nullable=True))
        batch_op.add_column(sa.Column('image_mime', sa.String(64), nullable=True))
        batch_op.add_column(sa.Column('max_new_tokens', sa.Integer(), nullable=False, server_default='512'))

    op.execute("UPDATE inferences SET prompt = query, response = output")

    with op.batch_alter_table('inferences', schema=None) as batch_op:
        batch_op.drop_column('query')
        batch_op.drop_column('images')
        batch_op.drop_column('output')
        batch_op.drop_column('guided_json')
