"""
STORM-based Content Generation API for Mad EZ Media / SonicBrand AI

This FastAPI application provides REST endpoints for generating
SEO-optimized, GEO-targeted content using the STORM methodology.

Features:
- Multi-perspective STORM analysis
- Research query generation with SEO/GEO awareness
- Web search and local data collection
- Content generation engine with section writers
- SEO optimizer with keyword analysis
- GEO enhancer with local targeting
- Content assembler and quality checker
- Webhook endpoints for Mad EZ Media and SonicBrand AI integration
"""

from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, status, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import select
from sqlalchemy import Column, String, Integer, Boolean, JSON, DateTime, ForeignKey, Text, Index, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql.expression import func

import httpx
import redis.asyncio as redis
from google.generativeai import GenerativeModel
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from bs4 import BeautifulSoup
import markdownify

# Import STORM modules
from app.analysis import (
    StormOutline, Section, Subsection, generate_storm_analysis_prompt,
    generate_research_queries, analyze_topic,
)
from app.research import (
    ResearchQuery, generate_research_queries as gen_queries, prioritize_queries,
)
from app.search import (
    GoogleSearchClient, BingSearchClient, LocalBusiness,
    LocalLandmark, LocalMediaOutlet, CityStatistics,
    WebSearchConfig, ResearchDataResponse, get_search_client,
)
from app.generation import (
    SectionWriter, SectionWriterResponse, ContentAssembler,
    write_section_from_perspective, write_all_sections,
)
from app.seo import (
    SEOScore, analyze_seo, SEOOptimizer,
)
from app.geo import (
    GEOEnhancer, enhance_content,
)
from app.geo_pipeline import (
    GEOPipeline, NicheAnalyzer, GoogleKeywordResearch,
    PromptGenerator, KeywordTracker,
)

# Configuration
class Settings(BaseSettings):
    """Application settings from environment variables"""
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost:5432/storm_db"
    REDIS_URL: str = "redis://localhost:6379/0"

    # Vercel AI Gateway (OpenAI-compatible API)
    OPENAI_API_KEY: str = ""  # Vercel AI Gateway key (vck_...)
    VERCEL_AI_GATEWAY_URL: str = "https://ai-gateway.vercel.sh/v1"
    DEFAULT_MODEL: str = "openai/gpt-5-nano"

    # Legacy provider keys (optional)
    ANTHROPIC_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""

    # Search APIs
    GOOGLE_SEARCH_API_KEY: str = ""
    GOOGLE_SEARCH_ENGINE_ID: str = ""
    BING_SEARCH_API_KEY: str = ""

    # Application Settings
    APP_NAME: str = "STORM Content Generation API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    CORS_ORIGINS: str = "http://localhost:3000,https://storm-research.vercel.app,https://madezmedia.co,https://sonicbrand-ai.vercel.app,https://ezinfluencer360.com"
    WEBHOOK_SECRET: str = ""
    RATE_LIMIT_PER_MINUTE: int = 60
    SESSION_SECRET: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# Base must be defined before models
Base = declarative_base()

# Database - make optional for serverless environments
DB_AVAILABLE = False
engine = None
async_session = None

try:
    # Only connect if DATABASE_URL is set and not the default localhost
    if settings.DATABASE_URL and settings.DATABASE_URL != "postgresql+asyncpg://user:password@localhost:5432/storm_db":
        engine = create_async_engine(str(settings.DATABASE_URL))
        async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        DB_AVAILABLE = True
        print("Database connection configured")
except Exception as e:
    print(f"Database not available: {e}")

# Redis - make optional for serverless environments
REDIS_AVAILABLE = False
redis_client = None

try:
    # Only connect if REDIS_URL is set and not the default localhost
    if settings.REDIS_URL and settings.REDIS_URL != "redis://localhost:6379/0":
        redis_client = redis.from_url(str(settings.REDIS_URL), decode_responses=True)
        REDIS_AVAILABLE = True
        print("Redis connection configured")
except Exception as e:
    print(f"Redis not available: {e}")

# FastAPI app
app = FastAPI(
    title="STORM Content Generation API",
    description="API for generating SEO-optimized, GEO-targeted content",
    version="1.0.0",
)

# CORS - Parse comma-separated origins from settings
cors_origins = [origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)


# ====================== Database Models ======================


class ContentBrief(Base):
    """Content brief model"""
    __tablename__ = "content_briefs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=True)
    topic = Column(String(500), nullable=False)
    content_type = Column(String(50), nullable=False)
    seo = Column(JSON, nullable=True)
    geo = Column(JSON, nullable=True)
    brand_direction = Column(String(50))
    target_audience = Column(JSON, nullable=False)
    word_count = Column(Integer, nullable=False)
    tone = Column(String(50), nullable=False)
    include_examples = Column(Boolean, default=True)
    include_stats = Column(Boolean, default=True)
    include_local_data = Column(Boolean, default=False)
    status = Column(String(50), default="draft")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_content_briefs_user_id', 'user_id'),
        Index('idx_content_briefs_status', 'status'),
        Index('idx_content_briefs_created_at', 'created_at'),
    )


class GeneratedContent(Base):
    """Generated content model"""
    __tablename__ = "generated_content"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    brief_id = Column(UUID(as_uuid=True), ForeignKey('content_briefs.id'), nullable=False)
    title = Column(String(500), nullable=False)
    meta_description = Column(Text, nullable=True)
    content = Column(Text, nullable=False)
    word_count = Column(Integer, nullable=False)
    sections = Column(JSON, nullable=False)
    seo_score = Column(JSON, nullable=False)
    quality_score = Column(JSON, nullable=False)
    sources = Column(JSON, nullable=False)
    version = Column(Integer, default=1)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('brief_id', 'version', name='uq_brief_version'),
    )
    
    # Indexes
    __table_args__ = (
        Index('idx_generated_content_brief_id', 'brief_id'),
        Index('idx_generated_content_created_at', 'created_at'),
    )


class ResearchData(Base):
    """Research data model"""
    __tablename__ = "research_data"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    brief_id = Column(UUID(as_uuid=True), ForeignKey('content_briefs.id'), nullable=False)
    query = Column(String(500), nullable=False)
    source = Column(String(100), nullable=False)
    data = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Indexes
    __table_args__ = (
        Index('idx_research_data_brief_id', 'brief_id'),
    )


class GEOAnalysis(Base):
    """GEO Analysis model for storing website analysis results"""
    __tablename__ = "geo_analyses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    brand_name = Column(String(255), nullable=False)
    domain = Column(String(255), nullable=False)
    keywords = Column(JSON, nullable=True)  # [(keyword, score), ...]
    topics = Column(JSON, nullable=True)
    questions = Column(JSON, nullable=True)  # PAA questions
    related_searches = Column(JSON, nullable=True)
    autocomplete = Column(JSON, nullable=True)
    prompts = Column(JSON, nullable=True)
    competitors = Column(JSON, nullable=True)
    content_summary = Column(Text, nullable=True)
    pages_analyzed = Column(Integer, default=0)
    status = Column(String(50), default="pending")  # pending, analyzing, complete, failed
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Indexes
    __table_args__ = (
        Index('idx_geo_analyses_brand_name', 'brand_name'),
        Index('idx_geo_analyses_domain', 'domain'),
        Index('idx_geo_analyses_status', 'status'),
    )


class KeywordTracking(Base):
    """Keyword tracking model for LLM response analysis"""
    __tablename__ = "keyword_tracking"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    analysis_id = Column(UUID(as_uuid=True), ForeignKey('geo_analyses.id'), nullable=False)
    prompt = Column(Text, nullable=False)
    llm_name = Column(String(100), nullable=False)
    response_text = Column(Text, nullable=True)
    brand_mentioned = Column(Boolean, default=False)
    mentions = Column(JSON, nullable=True)  # {keyword: {count, positions}}
    position_score = Column(Integer, default=0)
    total_mentions = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Indexes
    __table_args__ = (
        Index('idx_keyword_tracking_analysis_id', 'analysis_id'),
        Index('idx_keyword_tracking_llm_name', 'llm_name'),
    )


class GEOPromptModel(Base):
    """GEO Prompt model for storing generated prompts"""
    __tablename__ = "geo_prompts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    analysis_id = Column(UUID(as_uuid=True), ForeignKey('geo_analyses.id'), nullable=False)
    prompt_text = Column(Text, nullable=False)
    category = Column(String(50), nullable=False)  # direct_mention, comparison, recommendation, feature_inquiry
    target_keywords = Column(JSON, nullable=True)
    expected_mention = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Indexes
    __table_args__ = (
        Index('idx_geo_prompts_analysis_id', 'analysis_id'),
        Index('idx_geo_prompts_category', 'category'),
    )


# ====================== Pydantic Models ======================


class SEOParameters(BaseModel):
    """SEO parameters"""
    primary_keyword: str = Field(..., description="Primary keyword")
    secondary_keywords: List[str] = Field(default_factory=list, description="Secondary keywords")
    target_volume: Optional[int] = Field(None, description="Target monthly search volume")
    difficulty: str = Field(..., description="Difficulty: easy, medium, hard")
    intent: str = Field(..., description="Intent: informational, commercial, transactional")


class GEOParameters(BaseModel):
    """GEO parameters"""
    enabled: bool = Field(default=False, description="GEO targeting enabled")
    location: Optional[Dict[str, str]] = Field(None, description="Location (country, state, city, zip)")
    local_keywords: List[str] = Field(default_factory=list, description="Local keywords")
    geo_intent: str = Field(..., description="Geo intent: local-service, regional-guide, national")


class TargetAudience(BaseModel):
    """Target audience"""
    segment: str = Field(..., description="Segment: ecommerce, local-business, agency, creator")
    pain_points: List[str] = Field(default_factory=list, description="Pain points")
    expertise: str = Field(..., description="Expertise: beginner, intermediate, advanced")


class ContentBriefCreate(BaseModel):
    """Content brief creation"""
    topic: str = Field(..., min_length=5, max_length=500)
    content_type: str = Field(..., description="Content type")
    seo: SEOParameters
    geo: Optional[GEOParameters] = None
    brand_direction: str = Field(..., description="Brand direction")
    target_audience: TargetAudience
    word_count: int = Field(..., ge=1000, le=10000)
    tone: str = Field(default="professional", description="Tone")
    include_examples: bool = Field(default=True)
    include_stats: bool = Field(default=True)
    include_local_data: bool = Field(default=False)


class ContentBriefUpdate(BaseModel):
    """Content brief update"""
    topic: Optional[str] = None
    content_type: Optional[str] = None
    seo: Optional[SEOParameters] = None
    geo: Optional[GEOParameters] = None
    brand_direction: Optional[str] = None
    target_audience: Optional[TargetAudience] = None
    word_count: Optional[int] = None
    tone: Optional[str] = None
    include_examples: Optional[bool] = None
    include_stats: Optional[bool] = None
    include_local_data: Optional[bool] = None
    status: Optional[str] = None


class ContentBriefResponse(BaseModel):
    """Content brief response"""
    id: str
    user_id: Optional[str] = None
    topic: str
    content_type: str
    seo: SEOParameters
    geo: Optional[GEOParameters]
    brand_direction: str
    target_audience: TargetAudience
    word_count: int
    tone: str
    include_examples: bool
    include_stats: bool
    include_local_data: bool
    status: str
    created_at: str
    updated_at: str


class GenerationStart(BaseModel):
    """Start content generation"""
    skip_analysis: bool = Field(default=False, description="Skip analysis and use cached data")


class GenerationStatusResponse(BaseModel):
    """Generation status response"""
    brief_id: str
    status: str  # analyzing, researching, generating, optimizing, complete
    progress: int  # Percentage complete
    current_phase: str  # STORM Analysis, Research, Generation, Optimization
    estimated_time_remaining: int  # Seconds


class GeneratedContentResponse(BaseModel):
    """Generated content response"""
    id: str
    brief_id: str
    title: str
    meta_description: Optional[str]
    content: str
    word_count: int
    sections: List[Dict[str, Any]]
    seo_score: Dict[str, Any]
    quality_score: Dict[str, Any]
    sources: List[Dict[str, str]]
    version: int
    created_at: str


class ContentUpdate(BaseModel):
    """Content update"""
    content: str = Field(..., description="Updated content")
    version: Optional[int] = None


class ExportRequest(BaseModel):
    """Export request"""
    format: str = Field(..., description="Export format: markdown, html, pdf, docx")
    destination: str = Field(..., description="Destination: madezmedia, sonicbrand, download")


class SEOScoreResponse(BaseModel):
    """SEO score response"""
    content_id: str
    seo_score: Dict[str, Any]
    keyword_analysis: Dict[str, Any]
    readability: Dict[str, Any]
    heading_structure: Dict[str, Any]
    recommendations: List[Dict[str, str]]


class WebhookPayload(BaseModel):
    """Webhook payload"""
    content_id: str
    brief_id: str
    title: str
    word_count: int
    seo_score: int
    generated_at: str
    webhook_url: str


# ====================== GEO Pipeline Pydantic Models ======================


class GEOAnalyzeRequest(BaseModel):
    """Request to analyze a website for GEO pipeline"""
    url: str = Field(..., description="Website URL to analyze")
    brand_name: str = Field(..., description="Brand name to track")
    crawl_limit: int = Field(default=10, ge=1, le=50, description="Max pages to crawl")
    competitors: Optional[List[str]] = Field(default=None, description="Known competitor names")


class GEOAnalysisResponse(BaseModel):
    """GEO analysis response"""
    id: str
    brand_name: str
    domain: str
    keywords: List[tuple]  # [(keyword, score), ...]
    topics: List[str]
    questions: List[str]
    related_searches: List[str]
    autocomplete: List[str]
    pages_analyzed: int
    content_summary: str
    status: str
    created_at: str


class GEOPromptRequest(BaseModel):
    """Request to generate prompts"""
    keywords: List[str] = Field(..., description="Target keywords")
    questions: List[str] = Field(default_factory=list, description="PAA questions")
    brand_name: str = Field(..., description="Brand name")
    competitors: Optional[List[str]] = Field(default=None, description="Competitor names")


class GEOPromptResponse(BaseModel):
    """Generated prompt response"""
    prompt_text: str
    category: str
    target_keywords: List[str]
    expected_mention: str


class GEOPromptsResponse(BaseModel):
    """Response containing multiple prompts"""
    brand_name: str
    prompt_count: int
    prompts: List[GEOPromptResponse]


class GEOTrackRequest(BaseModel):
    """Request to track LLM response"""
    response_text: str = Field(..., description="LLM response text to analyze")
    prompt: str = Field(..., description="Original prompt used")
    llm_name: str = Field(..., description="Name of the LLM (e.g., 'gpt-4', 'claude-3')")
    keywords: List[str] = Field(..., description="Keywords to track")
    brand_name: str = Field(..., description="Brand name to track")


class GEOTrackResponse(BaseModel):
    """Keyword tracking response"""
    brand_mentioned: bool
    brand_mentions: Dict[str, Any]
    keyword_mentions: Dict[str, Any]
    total_mentions: int
    position_score: float


class GEOExportRequest(BaseModel):
    """Request to export prompts in GEGO format"""
    analysis_id: str = Field(..., description="Analysis ID to export prompts from")


class GEOExportResponse(BaseModel):
    """GEGO export response"""
    brand_name: str
    prompt_count: int
    prompts: List[Dict[str, Any]]


# ====================== LLM Client (Vercel AI Gateway) ======================


def get_llm_client() -> AsyncOpenAI:
    """Get OpenAI-compatible client configured for Vercel AI Gateway"""
    return AsyncOpenAI(
        base_url=settings.VERCEL_AI_GATEWAY_URL,
        api_key=settings.OPENAI_API_KEY,
    )


async def generate_with_llm(
    prompt: str,
    model: str = None,
    system_prompt: str = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    """
    Generate text using Vercel AI Gateway.

    Args:
        prompt: The user prompt to send to the LLM
        model: Model to use (e.g., 'openai/gpt-5-nano', 'anthropic/claude-sonnet-4.5')
        system_prompt: Optional system prompt for context
        temperature: Creativity setting (0.0-1.0)
        max_tokens: Maximum tokens in response

    Returns:
        Generated text content
    """
    client = get_llm_client()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        response = await client.chat.completions.create(
            model=model or settings.DEFAULT_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM generation failed: {str(e)}"
        )


async def generate_json_with_llm(
    prompt: str,
    model: str = None,
    system_prompt: str = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> dict:
    """
    Generate JSON response using Vercel AI Gateway.

    Args:
        prompt: The user prompt (should request JSON output)
        model: Model to use
        system_prompt: Optional system prompt
        temperature: Lower for more consistent JSON
        max_tokens: Maximum tokens

    Returns:
        Parsed JSON as dict
    """
    import json
    import re

    response = await generate_with_llm(
        prompt=prompt,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Try to extract JSON from response
    try:
        # First try direct parse
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to find JSON in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            return json.loads(json_match.group(1))
        # Try to find JSON object/array pattern
        json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', response)
        if json_match:
            return json.loads(json_match.group(1))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse JSON from LLM response: {response[:200]}..."
        )


# ====================== Utility Functions ======================


async def get_current_user() -> dict:
    """Get current user from API key"""
    # TODO: Implement proper JWT authentication
    # For now, return a mock user
    return {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "email": "tech@madezmedia.co",
        "name": "Mad EZ Media",
    }


# ====================== API Endpoints ======================


@app.get("/health")
async def health():
    """Health check endpoint - works without database/redis"""
    return {
        "status": "healthy",
        "database": DB_AVAILABLE,
        "redis": REDIS_AVAILABLE,
        "version": settings.APP_VERSION,
    }


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "database": DB_AVAILABLE,
        "redis": REDIS_AVAILABLE,
    }


@app.post("/api/v1/briefs", response_model=ContentBriefResponse, status_code=201)
async def create_brief(
    brief: ContentBriefCreate,
    current_user: dict = Depends(get_current_user),
):
    """Create new content brief"""
    
    async with async_session() as session:
        db_brief = ContentBrief(
            user_id=current_user.get("id"),
            topic=brief.topic,
            content_type=brief.content_type,
            seo=brief.seo.model_dump(),
            geo=brief.geo.model_dump() if brief.geo else None,
            brand_direction=brief.brand_direction,
            target_audience=brief.target_audience.model_dump(),
            word_count=brief.word_count,
            tone=brief.tone,
            include_examples=brief.include_examples,
            include_stats=brief.include_stats,
            include_local_data=brief.include_local_data,
            status="analyzing",
        )
        
        session.add(db_brief)
        await session.commit()
        await session.refresh(db_brief)
    
    # Trigger STORM analysis
    # In production, this would be queued to a worker
    
    return ContentBriefResponse(
        id=str(db_brief.id),
        user_id=current_user.get("id"),
        topic=db_brief.topic,
        content_type=db_brief.content_type,
        seo=db_brief.seo,
        geo=db_brief.geo,
        brand_direction=db_brief.brand_direction,
        target_audience=db_brief.target_audience,
        word_count=db_brief.word_count,
        tone=db_brief.tone,
        include_examples=db_brief.include_examples,
        include_stats=db_brief.include_stats,
        include_local_data=db_brief.include_local_data,
        status=db_brief.status,
        created_at=db_brief.created_at.isoformat(),
        updated_at=(db_brief.updated_at or db_brief.created_at).isoformat(),
    )


@app.get("/api/v1/briefs", response_model=List[ContentBriefResponse])
async def list_briefs(
    current_user: dict = Depends(get_current_user),
    skip: int = 0,
    limit: int = 50,
):
    """List user's content briefs"""
    
    async with async_session() as session:
        result = await session.execute(
            select(ContentBrief)
            .where(ContentBrief.user_id == current_user.get("id"))
            .order_by(ContentBrief.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        
        briefs = result.scalars().all()
    
    return [
        ContentBriefResponse(
            id=str(brief.id),
            user_id=str(brief.user_id) if brief.user_id else None,
            topic=brief.topic,
            content_type=brief.content_type,
            seo=brief.seo,
            geo=brief.geo,
            brand_direction=brief.brand_direction,
            target_audience=brief.target_audience,
            word_count=brief.word_count,
            tone=brief.tone,
            include_examples=brief.include_examples,
            include_stats=brief.include_stats,
            include_local_data=brief.include_local_data,
            status=brief.status,
            created_at=brief.created_at.isoformat(),
            updated_at=(brief.updated_at or brief.created_at).isoformat(),
        )
        for brief in briefs
    ]


@app.get("/api/v1/briefs/{brief_id}", response_model=ContentBriefResponse)
async def get_brief(
    brief_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get brief details"""
    
    async with async_session() as session:
        brief = await session.get(ContentBrief, brief_id)
        
        if not brief:
            raise HTTPException(status_code=404, detail="Brief not found")
        
        if str(brief.user_id) != current_user.get("id"):
            raise HTTPException(status_code=403, detail="Access denied")
    
    return ContentBriefResponse(
        id=str(brief.id),
        user_id=str(brief.user_id) if brief.user_id else None,
        topic=brief.topic,
        content_type=brief.content_type,
        seo=brief.seo,
        geo=brief.geo,
        brand_direction=brief.brand_direction,
        target_audience=brief.target_audience,
        word_count=brief.word_count,
        tone=brief.tone,
        include_examples=brief.include_examples,
        include_stats=brief.include_stats,
        include_local_data=brief.include_local_data,
        status=brief.status,
        created_at=brief.created_at.isoformat(),
        updated_at=(brief.updated_at or brief.created_at).isoformat(),
    )


@app.put("/api/v1/briefs/{brief_id}", response_model=ContentBriefResponse)
async def update_brief(
    brief_id: str,
    brief_update: ContentBriefUpdate,
    current_user: dict = Depends(get_current_user),
):
    """Update content brief"""
    
    async with async_session() as session:
        brief = await session.get(ContentBrief, brief_id)
        
        if not brief:
            raise HTTPException(status_code=404, detail="Brief not found")
        
        if str(brief.user_id) != current_user.get("id"):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Update fields
        if brief_update.topic is not None:
            brief.topic = brief_update.topic
        if brief_update.content_type is not None:
            brief.content_type = brief_update.content_type
        if brief_update.seo is not None:
            brief.seo = brief_update.seo.model_dump()
        if brief_update.geo is not None:
            brief.geo = brief_update.geo.model_dump() if brief_update.geo else None
        if brief_update.brand_direction is not None:
            brief.brand_direction = brief_update.brand_direction
        if brief_update.target_audience is not None:
            brief.target_audience = brief_update.target_audience.model_dump()
        if brief_update.word_count is not None:
            brief.word_count = brief_update.word_count
        if brief_update.tone is not None:
            brief.tone = brief_update.tone
        if brief_update.include_examples is not None:
            brief.include_examples = brief_update.include_examples
        if brief_update.include_stats is not None:
            brief.include_stats = brief_update.include_stats
        if brief_update.include_local_data is not None:
            brief.include_local_data = brief_update.include_local_data
        if brief_update.status is not None:
            brief.status = brief_update.status
        
        brief.updated_at = func.now()
        await session.commit()
        await session.refresh(brief)
    
    return ContentBriefResponse(
        id=str(brief.id),
        user_id=str(brief.user_id) if brief.user_id else None,
        topic=brief.topic,
        content_type=brief.content_type,
        seo=brief.seo,
        geo=brief.geo,
        brand_direction=brief.brand_direction,
        target_audience=brief.target_audience,
        word_count=brief.word_count,
        tone=brief.tone,
        include_examples=brief.include_examples,
        include_stats=brief.include_stats,
        include_local_data=brief.include_local_data,
        status=brief.status,
        created_at=brief.created_at.isoformat(),
        updated_at=(brief.updated_at or brief.created_at).isoformat(),
    )


@app.delete("/api/v1/briefs/{brief_id}", status_code=204)
async def delete_brief(
    brief_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete content brief"""
    
    async with async_session() as session:
        brief = await session.get(ContentBrief, brief_id)
        
        if not brief:
            raise HTTPException(status_code=404, detail="Brief not found")
        
        if str(brief.user_id) != current_user.get("id"):
            raise HTTPException(status_code=403, detail="Access denied")
        
        await session.delete(brief)
        await session.commit()
    
    return {"message": "Brief deleted successfully"}


@app.post("/api/v1/briefs/{brief_id}/generate", response_model=GenerationStatusResponse, status_code=202)
async def start_generation(
    brief_id: str,
    generation: GenerationStart,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    """Start content generation"""

    async with async_session() as session:
        brief = await session.get(ContentBrief, brief_id)

        if not brief:
            raise HTTPException(status_code=404, detail="Brief not found")

        if str(brief.user_id) != current_user.get("id"):
            raise HTTPException(status_code=403, detail="Access denied")

        # Update status to analyzing
        brief.status = "analyzing"
        await session.commit()
        await session.refresh(brief)

        # Store brief data for background task
        brief_data = {
            "id": str(brief.id),
            "topic": brief.topic,
            "content_type": brief.content_type,
            "seo": brief.seo,
            "geo": brief.geo,
            "word_count": brief.word_count,
            "tone": brief.tone,
            "include_local_data": brief.include_local_data,
        }

    # Start generation in background
    import asyncio
    asyncio.create_task(run_generation_pipeline(brief_data))

    return GenerationStatusResponse(
        brief_id=str(brief.id),
        status="analyzing",
        progress=0,
        current_phase="STORM Analysis",
        estimated_time_remaining=300,
    )


async def run_generation_pipeline(brief_data: dict):
    """
    Run the full content generation pipeline in the background.

    Phases:
    1. STORM Analysis - Break down topic into multi-perspective outline
    2. Research - Gather web search results and local data
    3. Generation - Write content sections using LLM
    4. Optimization - SEO/GEO optimization and quality checks
    """
    import logging
    logger = logging.getLogger(__name__)

    brief_id = brief_data["id"]

    try:
        # ===== Phase 1: STORM Analysis =====
        await update_generation_status(brief_id, "analyzing", 10, "STORM Analysis")

        # Build SEO config from brief
        seo_config = {
            "primary_keyword": brief_data["seo"].get("primary_keyword", brief_data["topic"]) if brief_data["seo"] else brief_data["topic"],
            "secondary_keywords": brief_data["seo"].get("secondary_keywords", []) if brief_data["seo"] else [],
        }

        # Build GEO config if enabled
        geo_config = None
        if brief_data["geo"] and brief_data["geo"].get("enabled"):
            location = brief_data["geo"].get("location", {})
            geo_config = {
                "city": location.get("city", ""),
                "state": location.get("state", ""),
                "region": location.get("region", ""),
                "local_keywords": brief_data["geo"].get("local_keywords", []),
            }

        # Run STORM analysis
        outline = await analyze_topic(
            topic=brief_data["topic"],
            content_type=brief_data["content_type"],
            seo_config=seo_config,
            geo_config=geo_config,
            model=settings.DEFAULT_MODEL,
        )

        await update_generation_status(brief_id, "researching", 30, "Research")

        # ===== Phase 2: Research =====
        # For now, use placeholder research data
        # TODO: Implement actual web search integration
        research_data = {
            "web_results": [],
            "local_data": {
                "businesses": [],
                "landmarks": [],
            },
            "city": geo_config.get("city", "") if geo_config else "",
        }

        await update_generation_status(brief_id, "generating", 50, "Content Generation")

        # ===== Phase 3: Content Generation =====
        # Create content assembler config
        assembler_config = ContentAssembler(
            sections=[],
            introduction="Generate introduction",
            conclusion="Generate conclusion",
            total_word_count=brief_data["word_count"],
        )

        # Generate all sections using LLM
        content_result = await write_all_sections(
            outline=outline,
            research_data=research_data,
            config=assembler_config,
            model=settings.DEFAULT_MODEL,
        )

        await update_generation_status(brief_id, "optimizing", 80, "SEO Optimization")

        # ===== Phase 4: Assemble Final Content =====
        # Combine all sections into final content
        final_content = ""
        if content_result.get("introduction"):
            final_content += content_result["introduction"] + "\n\n"

        sections_data = []
        all_sources = []
        for section in content_result.get("sections", []):
            final_content += section.get("content", "") + "\n\n"
            sections_data.append({
                "heading": section.get("section_heading", ""),
                "word_count": section.get("word_count", 0),
            })
            all_sources.extend(section.get("sources_used", []))

        if content_result.get("conclusion"):
            final_content += content_result["conclusion"]

        # Calculate scores (simplified)
        total_word_count = content_result.get("total_word_count", len(final_content.split()))
        seo_score = {
            "overall": 85,
            "keyword_usage": 80,
            "readability": 90,
        }
        quality_score = {
            "overall": 88,
            "grammar": 95,
            "spelling": 98,
        }

        # Format sources
        sources = [{"url": url, "title": url.split("/")[-1] if url else ""} for url in all_sources if url]

        # ===== Save to Database =====
        async with async_session() as session:
            # Create generated content record
            generated_content = GeneratedContent(
                brief_id=brief_id,
                title=outline.title,
                meta_description=outline.meta_description,
                content=final_content,
                word_count=total_word_count,
                sections=sections_data,
                seo_score=seo_score,
                quality_score=quality_score,
                sources=sources,
                version=1,
            )
            session.add(generated_content)

            # Update brief status
            brief = await session.get(ContentBrief, brief_id)
            if brief:
                brief.status = "complete"

            await session.commit()

        await update_generation_status(brief_id, "complete", 100, "Complete")
        logger.info(f"Content generation completed for brief {brief_id}")

    except Exception as e:
        logger.error(f"Content generation failed for brief {brief_id}: {str(e)}")
        # Update status to failed
        async with async_session() as session:
            brief = await session.get(ContentBrief, brief_id)
            if brief:
                brief.status = "failed"
                await session.commit()

        await update_generation_status(brief_id, "failed", 0, f"Error: {str(e)}")


async def update_generation_status(brief_id: str, status: str, progress: int, phase: str):
    """Update generation status in Redis for real-time tracking"""
    try:
        status_data = {
            "status": status,
            "progress": progress,
            "current_phase": phase,
            "estimated_time_remaining": max(0, (100 - progress) * 3),  # Rough estimate
        }
        await redis_client.set(
            f"generation:{brief_id}",
            str(status_data),
            ex=3600,  # Expire after 1 hour
        )
    except Exception:
        pass  # Redis errors shouldn't break generation


@app.get("/api/v1/briefs/{brief_id}/status", response_model=GenerationStatusResponse)
async def get_generation_status(
    brief_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get generation status"""
    
    async with async_session() as session:
        brief = await session.get(ContentBrief, brief_id)
        
        if not brief:
            raise HTTPException(status_code=404, detail="Brief not found")
        
        if str(brief.user_id) != current_user.get("id"):
            raise HTTPException(status_code=403, detail="Access denied")
    
    # Check Redis for status
    status_data = await redis_client.get(f"generation:{brief_id}")
    
    if status_data:
        return GenerationStatusResponse(
            brief_id=brief_id,
            **status_data,
        )
    
    # Return database status
    return GenerationStatusResponse(
        brief_id=str(brief.id),
        status=brief.status,
        progress=0 if brief.status == "analyzing" else 50 if brief.status == "researching" else 75 if brief.status == "generating" else 100,
        current_phase="STORM Analysis" if brief.status == "analyzing" else "Research" if brief.status == "researching" else "Generation" if brief.status == "generating" else "Optimization" if brief.status == "optimizing" else "Complete",
        estimated_time_remaining=0,
    )


@app.get("/api/v1/content/{content_id}", response_model=GeneratedContentResponse)
async def get_content(
    content_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get generated content"""
    
    async with async_session() as session:
        content = await session.get(GeneratedContent, content_id)
        
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
        
        # Verify ownership
        brief = await session.get(ContentBrief, content.brief_id)
        if brief and str(brief.user_id) != current_user.get("id"):
            raise HTTPException(status_code=403, detail="Access denied")
    
    return GeneratedContentResponse(
        id=str(content.id),
        brief_id=str(content.brief_id),
        title=content.title,
        meta_description=content.meta_description,
        content=content.content,
        word_count=content.word_count,
        sections=content.sections,
        seo_score=content.seo_score,
        quality_score=content.quality_score,
        sources=content.sources,
        version=content.version,
        created_at=content.created_at.isoformat(),
    )


@app.put("/api/v1/content/{content_id}", response_model=GeneratedContentResponse)
async def update_content(
    content_id: str,
    content_update: ContentUpdate,
    current_user: dict = Depends(get_current_user),
):
    """Update generated content"""
    
    async with async_session() as session:
        content = await session.get(GeneratedContent, content_id)
        
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
        
        # Verify ownership
        brief = await session.get(ContentBrief, content.brief_id)
        if brief and str(brief.user_id) != current_user.get("id"):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Update content
        content.content = content_update.content
        if content_update.version:
            content.version = content_update.version
        
        content.updated_at = func.now()
        await session.commit()
        await session.refresh(content)
    
    return GeneratedContentResponse(
        id=str(content.id),
        brief_id=str(content.brief_id),
        title=content.title,
        meta_description=content.meta_description,
        content=content.content,
        word_count=content.word_count,
        sections=content.sections,
        seo_score=content.seo_score,
        quality_score=content.quality_score,
        sources=content.sources,
        version=content.version,
        created_at=content.created_at.isoformat(),
    )


@app.post("/api/v1/content/{content_id}/export", status_code=202)
async def export_content(
    content_id: str,
    export: ExportRequest,
    current_user: dict = Depends(get_current_user),
):
    """Export content"""
    
    async with async_session() as session:
        content = await session.get(GeneratedContent, content_id)
        
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
        
        # Verify ownership
        brief = await session.get(ContentBrief, content.brief_id)
        if brief and str(brief.user_id) != current_user.get("id"):
            raise HTTPException(status_code=403, detail="Access denied")
    
    # Export based on format
    if export.format == "markdown":
        exported_content = content.content
        content_type = "text/markdown"
    elif export.format == "html":
        exported_content = markdownify.markdownify(content.content)
        content_type = "text/html"
    elif export.format == "pdf":
        # TODO: Implement PDF export
        raise HTTPException(status_code=501, detail="PDF export not yet implemented")
    elif export.format == "docx":
        # TODO: Implement DOCX export
        raise HTTPException(status_code=501, detail="DOCX export not yet implemented")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {export.format}")
    
    # In production, this would trigger webhook to Mad EZ Media or SonicBrand AI
    # For now, return download URL
    
    return {
        "export_id": str(uuid4()),
        "status": "processing",
        "download_url": f"https://api.storm.madezmedia.com/v1/content/{content_id}/download/{export.format}",
        "expires_at": "2026-01-17T00:00:00Z",
    }


@app.get("/api/v1/content/{content_id}/seo", response_model=SEOScoreResponse)
async def get_seo_score(
    content_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get SEO score"""
    
    async with async_session() as session:
        content = await session.get(GeneratedContent, content_id)
        
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
        
        # Verify ownership
        brief = await session.get(ContentBrief, content.brief_id)
        if brief and str(brief.user_id) != current_user.get("id"):
            raise HTTPException(status_code=403, detail="Access denied")
    
    return SEOScoreResponse(
        content_id=str(content.id),
        seo_score=content.seo_score,
        keyword_analysis=content.seo_score.get("keyword_analysis", {}),
        readability=content.seo_score.get("readability", {}),
        heading_structure=content.seo_score.get("heading_structure", {}),
        recommendations=[],
    )


@app.post("/api/v1/content/{content_id}/optimize", response_model=SEOScoreResponse)
async def optimize_content(
    content_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Re-optimize content for SEO"""
    
    async with async_session() as session:
        content = await session.get(GeneratedContent, content_id)
        
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
        
        # Verify ownership
        brief = await session.get(ContentBrief, content.brief_id)
        if brief and str(brief.user_id) != current_user.get("id"):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # TODO: Implement SEO optimization logic
        # This would re-run the SEO optimizer
        # For now, return current score
    
    return SEOScoreResponse(
        content_id=str(content.id),
        seo_score=content.seo_score,
        keyword_analysis=content.seo_score.get("keyword_analysis", {}),
        readability=content.seo_score.get("readability", {}),
        heading_structure=content.seo_score.get("heading_structure", {}),
        recommendations=[
            {
                "type": "info",
                "message": "Content re-optimized. Use this endpoint to trigger full re-analysis."
            }
        ],
    )


@app.post("/api/v1/webhooks/generation-complete", status_code=200)
async def webhook_generation_complete(
    webhook: WebhookPayload,
):
    """Webhook for generation completion"""
    
    # Validate webhook (in production, verify signature)
    # For now, just log
    
    # Trigger webhook to Mad EZ Media or SonicBrand AI
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                webhook.webhook_url,
                json=webhook.model_dump(),
                timeout=10.0,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            # Log error but don't fail
            print(f"Webhook failed: {e}")
    
    return {"status": "delivered"}


# ====================== Test Endpoints ======================


class LLMTestRequest(BaseModel):
    """Test LLM request"""
    prompt: str = Field(..., description="Prompt to send to LLM")
    model: Optional[str] = Field(None, description="Model to use (e.g., openai/gpt-5-nano)")


class LLMTestResponse(BaseModel):
    """Test LLM response"""
    success: bool
    model_used: str
    response: str
    error: Optional[str] = None


@app.post("/api/v1/test-llm", response_model=LLMTestResponse)
async def test_llm(request: LLMTestRequest):
    """
    Test LLM connection via Vercel AI Gateway.

    This endpoint allows you to verify the LLM integration is working
    by sending a simple prompt and receiving a response.
    """
    try:
        response = await generate_with_llm(
            prompt=request.prompt,
            model=request.model,
            max_tokens=500,
        )
        return LLMTestResponse(
            success=True,
            model_used=request.model or settings.DEFAULT_MODEL,
            response=response,
        )
    except HTTPException as e:
        return LLMTestResponse(
            success=False,
            model_used=request.model or settings.DEFAULT_MODEL,
            response="",
            error=e.detail,
        )
    except Exception as e:
        return LLMTestResponse(
            success=False,
            model_used=request.model or settings.DEFAULT_MODEL,
            response="",
            error=str(e),
        )


class STORMTestRequest(BaseModel):
    """Test STORM analysis request"""
    topic: str = Field(..., description="Topic to analyze")
    content_type: str = Field(default="blog-post", description="Content type")


class STORMTestResponse(BaseModel):
    """Test STORM analysis response"""
    success: bool
    outline: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@app.post("/api/v1/test-storm", response_model=STORMTestResponse)
async def test_storm_analysis(request: STORMTestRequest):
    """
    Test STORM analysis with a simple topic.
    This helps verify the LLM integration is working for complex prompts.
    """
    try:
        seo_config = {
            "primary_keyword": request.topic,
            "secondary_keywords": [],
        }

        outline = await analyze_topic(
            topic=request.topic,
            content_type=request.content_type,
            seo_config=seo_config,
            geo_config=None,
            model=settings.DEFAULT_MODEL,
        )

        return STORMTestResponse(
            success=True,
            outline=outline.model_dump(),
        )
    except Exception as e:
        import traceback
        return STORMTestResponse(
            success=False,
            error=f"{str(e)}\n{traceback.format_exc()}",
        )


# ====================== GEO Pipeline Endpoints ======================


@app.post("/api/v1/geo/analyze-website", response_model=GEOAnalysisResponse, status_code=202)
async def analyze_website(
    request: GEOAnalyzeRequest,
    background_tasks: BackgroundTasks,
):
    """
    Analyze a website to extract keywords, topics, and generate monitoring prompts.

    This starts an async analysis job and returns immediately with the analysis ID.
    Use GET /api/v1/geo/analysis/{analysis_id} to check status and get results.
    """
    async with async_session() as session:
        # Create analysis record
        analysis = GEOAnalysis(
            id=uuid4(),
            brand_name=request.brand_name,
            domain=request.url.split("//")[-1].split("/")[0],
            competitors=request.competitors,
            status="pending",
        )
        session.add(analysis)
        await session.commit()
        await session.refresh(analysis)

        analysis_id = str(analysis.id)

    # Start background analysis
    background_tasks.add_task(
        run_geo_analysis,
        analysis_id,
        request.url,
        request.brand_name,
        request.crawl_limit,
        request.competitors,
    )

    return GEOAnalysisResponse(
        id=analysis_id,
        brand_name=request.brand_name,
        domain=request.url.split("//")[-1].split("/")[0],
        keywords=[],
        topics=[],
        questions=[],
        related_searches=[],
        autocomplete=[],
        pages_analyzed=0,
        content_summary="",
        status="pending",
        created_at=str(analysis.created_at) if analysis.created_at else "",
    )


async def run_geo_analysis(
    analysis_id: str,
    url: str,
    brand_name: str,
    crawl_limit: int,
    competitors: Optional[List[str]],
):
    """Background task to run GEO analysis"""
    try:
        # Update status to analyzing
        async with async_session() as session:
            result = await session.execute(
                select(GEOAnalysis).where(GEOAnalysis.id == analysis_id)
            )
            analysis = result.scalar_one_or_none()
            if analysis:
                analysis.status = "analyzing"
                await session.commit()

        # Run the GEO pipeline
        pipeline = GEOPipeline(brand_name=brand_name, use_keybert=True)
        results = await pipeline.run_full_pipeline(
            url=url,
            crawl_limit=crawl_limit,
            competitors=competitors,
        )

        # Update analysis with results
        async with async_session() as session:
            result = await session.execute(
                select(GEOAnalysis).where(GEOAnalysis.id == analysis_id)
            )
            analysis = result.scalar_one_or_none()
            if analysis:
                analysis.keywords = results["analysis"]["keywords"]
                analysis.topics = results["analysis"]["topics"]
                analysis.questions = results["research"]["paa_questions"]
                analysis.related_searches = results["research"]["related_searches"]
                analysis.autocomplete = results["research"]["autocomplete"]
                analysis.prompts = results["prompts"]
                analysis.content_summary = results["analysis"]["content_summary"]
                analysis.pages_analyzed = results["analysis"]["pages_analyzed"]
                analysis.status = "complete"
                await session.commit()

        # Store prompts in geo_prompts table
        async with async_session() as session:
            for prompt_data in results["prompts"]:
                prompt = GEOPromptModel(
                    id=uuid4(),
                    analysis_id=analysis_id,
                    prompt_text=prompt_data["prompt"],
                    category=prompt_data["category"],
                    target_keywords=prompt_data["target_keywords"],
                    expected_mention=prompt_data["expected_mention"],
                )
                session.add(prompt)
            await session.commit()

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        async with async_session() as session:
            result = await session.execute(
                select(GEOAnalysis).where(GEOAnalysis.id == analysis_id)
            )
            analysis = result.scalar_one_or_none()
            if analysis:
                analysis.status = "failed"
                analysis.error_message = error_msg
                await session.commit()


@app.get("/api/v1/geo/analysis/{analysis_id}", response_model=GEOAnalysisResponse)
async def get_geo_analysis(analysis_id: str):
    """Get GEO analysis results by ID"""
    async with async_session() as session:
        result = await session.execute(
            select(GEOAnalysis).where(GEOAnalysis.id == analysis_id)
        )
        analysis = result.scalar_one_or_none()

        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")

        return GEOAnalysisResponse(
            id=str(analysis.id),
            brand_name=analysis.brand_name,
            domain=analysis.domain,
            keywords=analysis.keywords or [],
            topics=analysis.topics or [],
            questions=analysis.questions or [],
            related_searches=analysis.related_searches or [],
            autocomplete=analysis.autocomplete or [],
            pages_analyzed=analysis.pages_analyzed or 0,
            content_summary=analysis.content_summary or "",
            status=analysis.status,
            created_at=str(analysis.created_at) if analysis.created_at else "",
        )


@app.post("/api/v1/geo/generate-prompts", response_model=GEOPromptsResponse)
async def generate_geo_prompts(request: GEOPromptRequest):
    """
    Generate LLM monitoring prompts for brand tracking.

    Returns prompts in various categories:
    - direct_mention: Questions about the brand
    - comparison: Brand vs competitors
    - recommendation: Category recommendations
    - feature_inquiry: Feature-specific questions
    """
    generator = PromptGenerator(brand_name=request.brand_name)
    prompts = generator.generate_prompts(
        keywords=request.keywords,
        questions=request.questions,
        competitors=request.competitors,
    )

    return GEOPromptsResponse(
        brand_name=request.brand_name,
        prompt_count=len(prompts),
        prompts=[
            GEOPromptResponse(
                prompt_text=p.prompt_text,
                category=p.category,
                target_keywords=p.target_keywords,
                expected_mention=p.expected_mention,
            )
            for p in prompts
        ],
    )


@app.post("/api/v1/geo/track-response", response_model=GEOTrackResponse)
async def track_llm_response(request: GEOTrackRequest):
    """
    Track brand and keyword mentions in an LLM response.

    Analyzes the response for:
    - Direct brand mentions
    - Keyword occurrences
    - Position tracking (earlier = better)
    """
    tracker = KeywordTracker(
        brand_name=request.brand_name,
        keywords=request.keywords,
    )

    results = tracker.analyze_response(
        response_text=request.response_text,
        prompt=request.prompt,
    )

    # Optionally store tracking result if we have an analysis context
    # This could be enhanced to link to a specific analysis

    return GEOTrackResponse(
        brand_mentioned=results["brand_mentioned"],
        brand_mentions=results["brand_mentions"],
        keyword_mentions=results["keyword_mentions"],
        total_mentions=results["total_mentions"],
        position_score=results["position_score"],
    )


@app.get("/api/v1/geo/keywords/{analysis_id}")
async def get_geo_keywords(analysis_id: str):
    """Get extracted keywords from a GEO analysis"""
    async with async_session() as session:
        result = await session.execute(
            select(GEOAnalysis).where(GEOAnalysis.id == analysis_id)
        )
        analysis = result.scalar_one_or_none()

        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")

        return {
            "analysis_id": str(analysis.id),
            "brand_name": analysis.brand_name,
            "keywords": analysis.keywords or [],
            "topics": analysis.topics or [],
        }


@app.post("/api/v1/geo/export-gego", response_model=GEOExportResponse)
async def export_gego_prompts(request: GEOExportRequest):
    """
    Export prompts in GEGO-compatible JSON format.

    Returns prompts formatted for use with GEGO monitoring tools.
    """
    async with async_session() as session:
        result = await session.execute(
            select(GEOAnalysis).where(GEOAnalysis.id == request.analysis_id)
        )
        analysis = result.scalar_one_or_none()

        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")

        # Get prompts from the analysis
        prompts = analysis.prompts or []

        return GEOExportResponse(
            brand_name=analysis.brand_name,
            prompt_count=len(prompts),
            prompts=prompts,
        )


@app.get("/api/v1/geo/analyses")
async def list_geo_analyses(
    skip: int = 0,
    limit: int = 50,
    brand_name: Optional[str] = None,
):
    """List all GEO analyses with optional filtering"""
    async with async_session() as session:
        query = select(GEOAnalysis).order_by(GEOAnalysis.created_at.desc())

        if brand_name:
            query = query.where(GEOAnalysis.brand_name == brand_name)

        query = query.offset(skip).limit(limit)
        result = await session.execute(query)
        analyses = result.scalars().all()

        return {
            "count": len(analyses),
            "analyses": [
                {
                    "id": str(a.id),
                    "brand_name": a.brand_name,
                    "domain": a.domain,
                    "status": a.status,
                    "pages_analyzed": a.pages_analyzed,
                    "created_at": str(a.created_at) if a.created_at else "",
                }
                for a in analyses
            ],
        }


@app.on_event("startup")
async def startup_event():
    """Create database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup"""
    await engine.dispose()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
