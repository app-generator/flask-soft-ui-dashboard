"""empty message

Revision ID: d1b52174cf37
Revises: 
Create Date: 2024-12-01 14:26:11.838576

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd1b52174cf37'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('Users',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('username', sa.String(length=64), nullable=True),
    sa.Column('first_name', sa.String(length=100), nullable=True),
    sa.Column('last_name', sa.String(length=100), nullable=True),
    sa.Column('address', sa.String(length=100), nullable=True),
    sa.Column('bio', sa.String(length=200), nullable=True),
    sa.Column('email', sa.String(length=64), nullable=True),
    sa.Column('password', sa.LargeBinary(), nullable=True),
    sa.Column('oauth_github', sa.String(length=100), nullable=True),
    sa.Column('api_token', sa.String(length=100), nullable=True),
    sa.Column('api_token_ts', sa.Integer(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('email'),
    sa.UniqueConstraint('username')
    )
    op.create_table('flask_dance_oauth',
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('provider', sa.String(length=50), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('token', sa.JSON(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['Users.id'], ondelete='cascade'),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('flask_dance_oauth')
    op.drop_table('Users')
    # ### end Alembic commands ###