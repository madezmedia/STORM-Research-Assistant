"""GEO Pipeline tables

Revision ID: 002
Revises: 001
Create Date: 2026-01-17

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create geo_analyses table
    op.create_table(
        'geo_analyses',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('brand_name', sa.String(255), nullable=False),
        sa.Column('domain', sa.String(255), nullable=False),
        sa.Column('keywords', sa.JSON(), nullable=True),  # [(keyword, score), ...]
        sa.Column('topics', sa.JSON(), nullable=True),  # topic labels
        sa.Column('questions', sa.JSON(), nullable=True),  # PAA questions
        sa.Column('related_searches', sa.JSON(), nullable=True),
        sa.Column('autocomplete', sa.JSON(), nullable=True),
        sa.Column('prompts', sa.JSON(), nullable=True),  # generated prompts
        sa.Column('competitors', sa.JSON(), nullable=True),  # known competitors
        sa.Column('content_summary', sa.Text(), nullable=True),
        sa.Column('pages_analyzed', sa.Integer(), default=0),
        sa.Column('status', sa.String(50), default='pending'),  # pending, analyzing, complete, failed
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )

    # Create indexes for geo_analyses
    op.create_index('idx_geo_analyses_brand_name', 'geo_analyses', ['brand_name'])
    op.create_index('idx_geo_analyses_domain', 'geo_analyses', ['domain'])
    op.create_index('idx_geo_analyses_status', 'geo_analyses', ['status'])
    op.create_index('idx_geo_analyses_created_at', 'geo_analyses', ['created_at'])

    # Create keyword_tracking table
    op.create_table(
        'keyword_tracking',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('analysis_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('geo_analyses.id'), nullable=False),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('llm_name', sa.String(100), nullable=False),  # e.g., "gpt-4", "claude-3", "gemini"
        sa.Column('response_text', sa.Text(), nullable=True),
        sa.Column('brand_mentioned', sa.Boolean(), default=False),
        sa.Column('mentions', sa.JSON(), nullable=True),  # {keyword: {count, positions}}
        sa.Column('position_score', sa.Float(), default=0.0),  # Higher = mentioned earlier
        sa.Column('total_mentions', sa.Integer(), default=0),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Create indexes for keyword_tracking
    op.create_index('idx_keyword_tracking_analysis_id', 'keyword_tracking', ['analysis_id'])
    op.create_index('idx_keyword_tracking_llm_name', 'keyword_tracking', ['llm_name'])
    op.create_index('idx_keyword_tracking_brand_mentioned', 'keyword_tracking', ['brand_mentioned'])
    op.create_index('idx_keyword_tracking_created_at', 'keyword_tracking', ['created_at'])

    # Create geo_prompts table for storing individual prompts
    op.create_table(
        'geo_prompts',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('analysis_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('geo_analyses.id'), nullable=False),
        sa.Column('prompt_text', sa.Text(), nullable=False),
        sa.Column('category', sa.String(50), nullable=False),  # direct_mention, comparison, recommendation, feature_inquiry
        sa.Column('target_keywords', sa.JSON(), nullable=True),
        sa.Column('expected_mention', sa.String(255), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Create indexes for geo_prompts
    op.create_index('idx_geo_prompts_analysis_id', 'geo_prompts', ['analysis_id'])
    op.create_index('idx_geo_prompts_category', 'geo_prompts', ['category'])


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_index('idx_geo_prompts_category', 'geo_prompts')
    op.drop_index('idx_geo_prompts_analysis_id', 'geo_prompts')
    op.drop_table('geo_prompts')

    op.drop_index('idx_keyword_tracking_created_at', 'keyword_tracking')
    op.drop_index('idx_keyword_tracking_brand_mentioned', 'keyword_tracking')
    op.drop_index('idx_keyword_tracking_llm_name', 'keyword_tracking')
    op.drop_index('idx_keyword_tracking_analysis_id', 'keyword_tracking')
    op.drop_table('keyword_tracking')

    op.drop_index('idx_geo_analyses_created_at', 'geo_analyses')
    op.drop_index('idx_geo_analyses_status', 'geo_analyses')
    op.drop_index('idx_geo_analyses_domain', 'geo_analyses')
    op.drop_index('idx_geo_analyses_brand_name', 'geo_analyses')
    op.drop_table('geo_analyses')
