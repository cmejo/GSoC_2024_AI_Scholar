"""Add enhanced session fields for model management

Revision ID: 003_add_enhanced_session_fields
Revises: 002_add_user_profile_fields
Create Date: 2024-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '003_add_enhanced_session_fields'
down_revision = '002_add_user_profile_fields'
branch_labels = None
depends_on = None


def upgrade():
    # Add new columns to chat_sessions table
    op.add_column('chat_sessions', sa.Column('model_name', sa.String(length=100), nullable=True, default='llama2'))
    op.add_column('chat_sessions', sa.Column('system_prompt_type', sa.String(length=50), nullable=True, default='general'))
    op.add_column('chat_sessions', sa.Column('custom_system_prompt', sa.Text(), nullable=True))
    op.add_column('chat_sessions', sa.Column('model_parameters', sa.JSON(), nullable=True))
    
    # Set default values for existing records
    op.execute("UPDATE chat_sessions SET model_name = 'llama2' WHERE model_name IS NULL")
    op.execute("UPDATE chat_sessions SET system_prompt_type = 'general' WHERE system_prompt_type IS NULL")


def downgrade():
    # Remove the added columns
    op.drop_column('chat_sessions', 'model_parameters')
    op.drop_column('chat_sessions', 'custom_system_prompt')
    op.drop_column('chat_sessions', 'system_prompt_type')
    op.drop_column('chat_sessions', 'model_name')