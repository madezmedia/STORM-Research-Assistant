"""Initial schema

Revision ID: 001
Revises: 
Create Date: 2026-01-16

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create content_briefs table
    op.create_table(
        'content_briefs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('topic', sa.String(500), nullable=False),
        sa.Column('content_type', sa.String(50), nullable=False),
        sa.Column('seo', sa.JSON(), nullable=True),
        sa.Column('geo', sa.JSON(), nullable=True),
        sa.Column('brand_direction', sa.String(50)),
        sa.Column('target_audience', sa.JSON(), nullable=False),
        sa.Column('word_count', sa.Integer(), nullable=False),
        sa.Column('tone', sa.String(50), nullable=False),
        sa.Column('include_examples', sa.Boolean(), default=True),
        sa.Column('include_stats', sa.Boolean(), default=True),
        sa.Column('include_local_data', sa.Boolean(), default=False),
        sa.Column('status', sa.String(50), default='draft'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )
    
    # Create indexes for content_briefs
    op.create_index('idx_content_briefs_user_id', 'content_briefs', ['user_id'])
    op.create_index('idx_content_briefs_status', 'content_briefs', ['status'])
    op.create_index('idx_content_briefs_created_at', 'content_briefs', ['created_at'])
    
    # Create generated_content table
    op.create_table(
        'generated_content',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('brief_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('content_briefs.id'), nullable=False),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('meta_description', sa.Text(), nullable=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('word_count', sa.Integer(), nullable=False),
        sa.Column('sections', sa.JSON(), nullable=False),
        sa.Column('seo_score', sa.JSON(), nullable=False),
        sa.Column('quality_score', sa.JSON(), nullable=False),
        sa.Column('sources', sa.JSON(), nullable=False),
        sa.Column('version', sa.Integer(), default=1),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    
    # Create unique constraint for generated_content
    op.create_unique_constraint('uq_brief_version', 'generated_content', ['brief_id', 'version'])
    
    # Create indexes for generated_content
    op.create_index('idx_generated_content_brief_id', 'generated_content', ['brief_id'])
    op.create_index('idx_generated_content_created_at', 'generated_content', ['created_at'])
    
    # Create research_data table
    op.create_table(
        'research_data',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('brief_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('content_briefs.id'), nullable=False),
        sa.Column('query', sa.String(500), nullable=False),
        sa.Column('source', sa.String(100), nullable=False),
        sa.Column('data', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    
    # Create index for research_data
    op.create_index('idx_research_data_brief_id', 'research_data', ['brief_id'])


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_index('idx_research_data_brief_id', 'research_data')
    op.drop_table('research_data')
    
    op.drop_index('idx_generated_content_created_at', 'generated_content')
    op.drop_index('idx_generated_content_brief_id', 'generated_content')
    op.drop_constraint('uq_brief_version', 'generated_content')
    op.drop_table('generated_content')
    
    op.drop_index('idx_content_briefs_created_at', 'content_briefs')
    op.drop_index('idx_content_briefs_status', 'content_briefs')
    op.drop_index('idx_content_briefs_user_id', 'content_briefs')
    op.drop_table('content_briefs')
