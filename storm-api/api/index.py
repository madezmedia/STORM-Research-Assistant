"""
Vercel Serverless Function Handler

FastAPI app for STORM Content Generation API.
Inline implementation for Vercel compatibility.
"""

import os
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx

# Settings
class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    VERCEL_AI_GATEWAY_URL: str = "https://ai-gateway.vercel.sh/v1"
    DEFAULT_MODEL: str = "openai/gpt-4o-mini"
    TAVILY_API_KEY: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()


# =============================================================================
# Tavily Web Search
# =============================================================================

async def search_web(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Execute web search using Tavily API for grounded research"""
    if not settings.TAVILY_API_KEY:
        print("Tavily API key not configured - skipping web search")
        return []

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": settings.TAVILY_API_KEY,
                    "query": query,
                    "max_results": max_results,
                    "include_answer": True,
                    "include_raw_content": False,
                    "search_depth": "basic"
                },
                timeout=30.0
            )

            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                # Format results consistently
                return [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", r.get("snippet", ""))[:500],
                        "score": r.get("score", 0)
                    }
                    for r in results
                ]
            else:
                print(f"Tavily search failed: {response.status_code}")
    except Exception as e:
        print(f"Tavily search error: {e}")

    return []


# =============================================================================
# STORM Outline Generation
# =============================================================================

async def generate_storm_outline(topic: str, content_type: str = "blog_post") -> Dict[str, Any]:
    """Generate a STORM-style outline with 7 perspectives and research queries"""

    outline_prompt = f"""You are an expert content strategist using the STORM methodology.
Generate a comprehensive outline for the topic: "{topic}"
Content type: {content_type}

Analyze this topic from these 7 expert perspectives:
1. Beginner - What foundational concepts, terminology, and prerequisites does a newcomer need?
2. Business Owner - What are the ROI considerations, costs, implementation timelines, and risks?
3. Local Market Expert - What regulations, competitors, and local success stories are relevant?
4. Technical Expert - How does this work technically? What are the requirements and integrations?
5. Competitive Analyst - What are the alternatives, comparisons, and key differentiators?
6. Customer Advocate - What are the pain points, decision factors, and user experience concerns?
7. Industry Expert - What are the best practices, emerging trends, and future outlook?

Return a JSON object with this exact structure:
{{
    "title": "Engaging article title",
    "meta_description": "SEO-friendly description under 160 characters",
    "sections": [
        {{
            "title": "Section Title",
            "perspective": "Which perspective this addresses",
            "subsections": [
                {{"title": "Subsection 1"}},
                {{"title": "Subsection 2"}}
            ],
            "key_questions": ["Question this section should answer"],
            "data_needed": ["Statistics or facts needed"]
        }}
    ],
    "research_queries": [
        "Specific search query 1 for gathering facts",
        "Specific search query 2 for gathering facts",
        "Specific search query 3 for gathering facts",
        "Specific search query 4 for gathering facts",
        "Specific search query 5 for gathering facts"
    ]
}}

Generate 4-6 comprehensive sections and 5 targeted research queries.
Respond ONLY with the JSON object, no additional text."""

    response = await call_llm(outline_prompt, max_tokens=2000)

    if response:
        # Try to parse JSON from response
        try:
            # Handle potential markdown code blocks
            json_str = response.strip()
            if json_str.startswith("```"):
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
            import json
            return json.loads(json_str.strip())
        except Exception as e:
            print(f"Failed to parse outline JSON: {e}")

    # Fallback outline
    return {
        "title": f"Comprehensive Guide to {topic}",
        "meta_description": f"Learn everything about {topic} with expert insights and practical recommendations.",
        "sections": [
            {"title": "Introduction", "perspective": "Beginner", "subsections": [{"title": "Overview"}, {"title": "Why It Matters"}], "key_questions": [f"What is {topic}?"], "data_needed": []},
            {"title": "Key Concepts", "perspective": "Technical", "subsections": [{"title": "Fundamentals"}, {"title": "How It Works"}], "key_questions": [f"How does {topic} work?"], "data_needed": []},
            {"title": "Benefits & Considerations", "perspective": "Business Owner", "subsections": [{"title": "Advantages"}, {"title": "Challenges"}], "key_questions": ["What are the benefits?"], "data_needed": []},
            {"title": "Best Practices", "perspective": "Industry Expert", "subsections": [{"title": "Recommendations"}, {"title": "Common Mistakes"}], "key_questions": ["What are the best practices?"], "data_needed": []},
            {"title": "Conclusion", "perspective": "Customer", "subsections": [{"title": "Summary"}, {"title": "Next Steps"}], "key_questions": ["What should I do next?"], "data_needed": []}
        ],
        "research_queries": [
            f"{topic} statistics 2024",
            f"{topic} benefits and advantages",
            f"{topic} best practices guide",
            f"{topic} comparison alternatives",
            f"{topic} trends future outlook"
        ]
    }


# =============================================================================
# Research-Grounded Content Generation
# =============================================================================

async def generate_content_with_research(
    topic: str,
    outline: Dict[str, Any],
    research_results: List[Dict[str, Any]],
    word_count: int = 1500,
    tone: str = "professional"
) -> str:
    """Generate comprehensive content grounded in research results"""

    # Format research context
    if research_results:
        research_context = "\n\n".join([
            f"**Source: {r.get('title', 'Unknown')}**\nURL: {r.get('url', 'N/A')}\n{r.get('content', '')}"
            for r in research_results[:15]  # Limit to 15 sources
        ])
    else:
        research_context = "No research data available - generate content based on general knowledge."

    # Format outline sections
    import json
    sections_text = json.dumps(outline.get("sections", []), indent=2)

    content_prompt = f"""You are an expert content writer using the STORM methodology.
Write a comprehensive, well-researched article on: "{topic}"

## Research Context (USE THESE SOURCES FOR FACTS AND CITATIONS)
{research_context}

## Article Outline to Follow
{sections_text}

## Requirements
- Target word count: approximately {word_count} words
- Tone: {tone}
- Use markdown formatting with ## for main sections and ### for subsections
- Ground your content with facts from the research context above
- Cite sources using [Source Title] format when referencing specific information
- Include statistics and data points from the research
- Make content engaging with practical examples
- Start with a compelling introduction that hooks the reader
- End with actionable conclusions and recommendations

## Important
- Follow the outline structure provided
- Incorporate insights from ALL perspectives (beginner, business, technical, etc.)
- Make sure each section addresses its key questions
- Do NOT make up statistics - only use what's in the research context or clearly state "studies show" without specific numbers

Write the complete article now:"""

    content = await call_llm(content_prompt, max_tokens=4000)

    if not content:
        # Fallback content
        content = f"""# {outline.get('title', topic)}

## Introduction

{outline.get('meta_description', f'This guide explores {topic} comprehensively.')}

*Note: Configure API keys to enable AI-powered content generation with research.*

## Overview

Understanding {topic} is essential in today's rapidly evolving landscape...

## Key Takeaways

Based on our analysis of {topic}, we recommend:

1. Start with the fundamentals
2. Consider your specific needs and context
3. Stay updated with industry trends

## Conclusion

{topic} continues to evolve, and staying informed is key to success.
"""

    return content

# Create app
app = FastAPI(
    title="STORM Content Generation API",
    description="API for generating SEO-optimized, GEO-targeted content",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class LLMTestRequest(BaseModel):
    prompt: str = Field(default="Say hello!")
    model: Optional[str] = None

class LLMTestResponse(BaseModel):
    success: bool
    response: str
    model_used: str
    provider: str

# Basic endpoints
@app.get("/")
def root():
    return {
        "name": "STORM Content Generation API",
        "version": "1.0.0",
        "status": "running",
        "api_key_configured": bool(settings.OPENAI_API_KEY),
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/api/v1/test")
def test():
    return {"message": "API is working", "endpoint": "/api/v1/test"}

@app.post("/api/v1/test-llm", response_model=LLMTestResponse)
async def test_llm(request: LLMTestRequest):
    """Test LLM connectivity using Vercel AI Gateway"""
    model = request.model or settings.DEFAULT_MODEL

    if not settings.OPENAI_API_KEY:
        return LLMTestResponse(
            success=False,
            response="API key not configured",
            model_used=model,
            provider="none"
        )

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.VERCEL_AI_GATEWAY_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": request.prompt}],
                    "max_tokens": 100
                },
                timeout=30.0
            )

            if response.status_code == 200:
                data = response.json()
                return LLMTestResponse(
                    success=True,
                    response=data["choices"][0]["message"]["content"],
                    model_used=model,
                    provider="vercel-ai-gateway"
                )
            else:
                return LLMTestResponse(
                    success=False,
                    response=f"API error: {response.status_code} - {response.text[:200]}",
                    model_used=model,
                    provider="vercel-ai-gateway"
                )
    except Exception as e:
        return LLMTestResponse(
            success=False,
            response=f"Error: {str(e)}",
            model_used=model,
            provider="vercel-ai-gateway"
        )

# STORM Test endpoint
class STORMTestRequest(BaseModel):
    topic: str = Field(default="AI in healthcare")
    perspectives: int = Field(default=3, ge=1, le=5)

class STORMTestResponse(BaseModel):
    success: bool
    topic: str
    perspectives: List[str]
    outline: Dict[str, Any]
    sample_content: str

async def call_llm(prompt: str, model: str = None, max_tokens: int = 1000) -> str:
    """Helper function to call Vercel AI Gateway"""
    model = model or settings.DEFAULT_MODEL

    if not settings.OPENAI_API_KEY:
        return None

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.VERCEL_AI_GATEWAY_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens
                },
                timeout=60.0
            )

            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"LLM call failed: {e}")

    return None


@app.post("/api/v1/test-storm", response_model=STORMTestResponse)
async def test_storm(request: STORMTestRequest):
    """Test STORM analysis with full research pipeline"""

    # =========================================================================
    # STEP 1: Generate expert perspectives
    # =========================================================================
    llm_perspectives = None
    if settings.OPENAI_API_KEY:
        perspective_prompt = f"""Generate {request.perspectives} expert perspectives on the topic: "{request.topic}"

For each perspective, provide a unique viewpoint from a different domain expert:
1. Industry Analyst - Market trends and business implications
2. Academic Researcher - Scientific and theoretical foundations
3. Practitioner - Real-world implementation and best practices
4. Consumer Advocate - User needs and accessibility concerns
5. Technology Expert - Technical requirements and innovations

Format as a numbered list with perspective title and 1-2 sentence description."""

        llm_perspectives = await call_llm(perspective_prompt, max_tokens=500)

    # Parse LLM perspectives or use fallback
    if llm_perspectives:
        perspectives = [line.strip() for line in llm_perspectives.split('\n') if line.strip() and line.strip()[0].isdigit()][:request.perspectives]
    else:
        perspectives = [
            f"1. Industry Analyst perspective on {request.topic}",
            f"2. Academic Researcher perspective on {request.topic}",
            f"3. Practitioner perspective on {request.topic}"
        ][:request.perspectives]

    # =========================================================================
    # STEP 2: Generate STORM outline with research queries
    # =========================================================================
    outline = await generate_storm_outline(request.topic, "blog_post")

    # Ensure outline has the required frontend format
    if "sections" in outline:
        for section in outline["sections"]:
            if "subsections" not in section:
                section["subsections"] = [{"title": "Overview"}]

    # =========================================================================
    # STEP 3: Execute research queries (if Tavily configured)
    # =========================================================================
    research_results = []
    research_queries = outline.get("research_queries", [])[:3]  # Limit for test endpoint

    for query in research_queries:
        results = await search_web(query, max_results=2)
        research_results.extend(results)

    # =========================================================================
    # STEP 4: Generate content with research context
    # =========================================================================
    sample_content = await generate_content_with_research(
        topic=request.topic,
        outline=outline,
        research_results=research_results,
        word_count=800,
        tone="professional"
    )

    # Add sources section if we have research
    if research_results:
        sources_section = "\n\n---\n\n## Sources\n\n" + "\n".join([
            f"- [{r.get('title', 'Source')}]({r.get('url', '#')})"
            for r in research_results[:5]
        ])
        sample_content += sources_section

    return STORMTestResponse(
        success=True,
        topic=request.topic,
        perspectives=perspectives,
        outline=outline,
        sample_content=sample_content
    )

# In-memory storage for briefs (no database needed for serverless)
import uuid
from datetime import datetime

briefs_store: Dict[str, Any] = {}
content_store: Dict[str, Any] = {}

# Briefs endpoint
class ContentBriefCreate(BaseModel):
    topic: str
    content_type: str = "blog_post"
    word_count: int = 1500
    tone: str = "professional"
    seo: Optional[Dict[str, Any]] = None
    brand_direction: Optional[str] = None
    target_audience: Optional[Dict[str, Any]] = None
    include_examples: bool = True
    include_stats: bool = True
    include_local_data: bool = False

class ContentBriefResponse(BaseModel):
    id: str
    topic: str
    content_type: str
    status: str

class GenerationStatusResponse(BaseModel):
    brief_id: str
    status: str
    progress: int
    current_phase: str
    estimated_time_remaining: int

class GeneratedContentResponse(BaseModel):
    id: str
    brief_id: str
    title: str
    content: str
    word_count: int
    sections: List[Dict[str, Any]]
    seo_score: Dict[str, Any]

@app.post("/api/v1/briefs", response_model=ContentBriefResponse, status_code=201)
async def create_brief(brief: ContentBriefCreate):
    """Create a new content brief"""
    brief_id = str(uuid.uuid4())
    briefs_store[brief_id] = {
        "id": brief_id,
        "topic": brief.topic,
        "content_type": brief.content_type,
        "word_count": brief.word_count,
        "tone": brief.tone,
        "status": "created",
        "progress": 0,
        "current_phase": "Created",
        "created_at": datetime.now().isoformat()
    }
    return ContentBriefResponse(
        id=brief_id,
        topic=brief.topic,
        content_type=brief.content_type,
        status="created"
    )

@app.get("/api/v1/briefs")
async def list_briefs():
    """List content briefs"""
    return list(briefs_store.values())

@app.post("/api/v1/briefs/{brief_id}/generate")
async def generate_content_endpoint(brief_id: str):
    """Start STORM content generation for a brief with full research pipeline"""
    if brief_id not in briefs_store:
        raise HTTPException(status_code=404, detail="Brief not found")

    brief = briefs_store[brief_id]
    topic = brief["topic"]
    word_count = brief.get("word_count", 1500)
    content_type = brief.get("content_type", "blog_post")
    tone = brief.get("tone", "professional")

    # =========================================================================
    # PHASE 1: STORM Analysis - Generate outline with research queries (10%)
    # =========================================================================
    brief["status"] = "generating"
    brief["progress"] = 10
    brief["current_phase"] = "Generating STORM outline"

    outline = await generate_storm_outline(topic, content_type)
    print(f"Generated outline with {len(outline.get('sections', []))} sections")

    # =========================================================================
    # PHASE 2: Research - Execute queries and gather sources (30%)
    # =========================================================================
    brief["progress"] = 30
    brief["current_phase"] = "Researching topic"

    research_results = []
    research_queries = outline.get("research_queries", [])[:5]  # Limit to 5 queries

    for i, query in enumerate(research_queries):
        print(f"Executing research query {i+1}/{len(research_queries)}: {query}")
        results = await search_web(query, max_results=3)
        research_results.extend(results)
        # Update progress during research
        brief["progress"] = 30 + int((i + 1) / len(research_queries) * 20)

    print(f"Gathered {len(research_results)} research sources")

    # =========================================================================
    # PHASE 3: Content Generation - Write with research context (60%)
    # =========================================================================
    brief["progress"] = 60
    brief["current_phase"] = "Writing content"

    content = await generate_content_with_research(
        topic=topic,
        outline=outline,
        research_results=research_results,
        word_count=word_count,
        tone=tone
    )

    # =========================================================================
    # PHASE 4: Finalize - Store and complete (100%)
    # =========================================================================
    brief["progress"] = 90
    brief["current_phase"] = "Finalizing content"

    # Count words
    actual_word_count = len(content.split())

    # Extract sections from outline for response
    sections = [
        {
            "heading": section.get("title", "Section"),
            "perspective": section.get("perspective", "General"),
            "content": "..."
        }
        for section in outline.get("sections", [])
    ]

    # Collect unique sources
    sources = list(set([r.get("url", "") for r in research_results if r.get("url")]))

    # Store generated content
    content_id = str(uuid.uuid4())
    content_store[brief_id] = {
        "id": content_id,
        "brief_id": brief_id,
        "title": outline.get("title", topic),
        "meta_description": outline.get("meta_description", ""),
        "content": content,
        "word_count": actual_word_count,
        "sections": sections,
        "sources": sources[:10],  # Top 10 sources
        "research_queries": research_queries,
        "seo_score": {
            "overall": 85 if research_results else 60,
            "readability": 90,
            "keyword_density": 80,
            "sources_cited": len(sources)
        },
        "created_at": datetime.now().isoformat()
    }

    # Update brief status
    brief["status"] = "complete"
    brief["progress"] = 100
    brief["current_phase"] = "Complete"
    brief["outline"] = outline

    return {
        "status": "started",
        "brief_id": brief_id,
        "research_sources": len(research_results),
        "outline_sections": len(outline.get("sections", []))
    }

@app.get("/api/v1/briefs/{brief_id}/status", response_model=GenerationStatusResponse)
async def get_generation_status(brief_id: str):
    """Get the status of content generation"""
    if brief_id not in briefs_store:
        raise HTTPException(status_code=404, detail="Brief not found")

    brief = briefs_store[brief_id]
    return GenerationStatusResponse(
        brief_id=brief_id,
        status=brief.get("status", "unknown"),
        progress=brief.get("progress", 0),
        current_phase=brief.get("current_phase", "Unknown"),
        estimated_time_remaining=0 if brief.get("status") == "complete" else 5
    )

@app.get("/api/v1/content/{brief_id}", response_model=GeneratedContentResponse)
async def get_generated_content(brief_id: str):
    """Get the generated content for a brief"""
    if brief_id not in content_store:
        raise HTTPException(status_code=404, detail="Content not found")

    return content_store[brief_id]


# =============================================================================
# Slideshow/Video Generation Endpoints
# =============================================================================

# Import slideshow modules
import sys
from pathlib import Path

# Add app directory to path for imports
app_dir = Path(__file__).parent.parent / "app"
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

# Import slideshow modules
try:
    from image_generation import (
        ImageGenerationClient,
        SlideGenerationOptions,
        generate_slideshow_content,
        extract_slides_from_content,
        generate_all_slide_images
    )
    from slideshow_generator import (
        SlideshowOptions,
        generate_html_slideshow,
        generate_embed_code
    )
    from video_generator import (
        VideoOptions,
        VideoGenerator,
        generate_video_from_slides,
        check_ffmpeg,
        is_ffmpeg_available
    )
    SLIDESHOW_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Slideshow modules not available: {e}")
    SLIDESHOW_MODULES_AVAILABLE = False


# Extended Settings for Z.AI
class SlideshowSettings(BaseSettings):
    ZAI_API_KEY: str = ""
    ZAI_API_URL: str = "https://api.z.ai/api/paas/v4"
    GLM_IMAGE_MODEL: str = "glm-image"

    class Config:
        env_file = ".env"
        extra = "ignore"

slideshow_settings = SlideshowSettings()


# Slideshow Request Models
class SlideGenerationRequest(BaseModel):
    """Request to generate slides from content"""
    content: str = Field(..., description="Blog post or article content")
    max_slides: int = Field(default=10, ge=1, le=20)
    style: str = Field(default="professional")  # professional, minimalist, vibrant
    brand_colors: Optional[Dict[str, str]] = None  # {primary: "#hex", secondary: "#hex"}
    include_title_slide: bool = True
    include_conclusion_slide: bool = True
    generate_images: bool = True  # Set False for text-only slides


class SlideshowGenerationRequest(BaseModel):
    """Request to generate HTML slideshow"""
    content: str = Field(..., description="Blog post or article content")
    max_slides: int = Field(default=10, ge=1, le=20)
    style: str = Field(default="professional")
    autoplay: bool = True
    duration: int = Field(default=5, ge=1, le=30)  # seconds per slide
    transition: str = Field(default="fade")  # fade, slide, zoom
    show_controls: bool = True
    loop: bool = True
    theme: str = Field(default="dark")  # dark, light
    title: str = Field(default="Presentation")
    generate_images: bool = True


class VideoGenerationRequest(BaseModel):
    """Request to generate video from content"""
    content: str = Field(..., description="Blog post or article content")
    max_slides: int = Field(default=10, ge=1, le=20)
    style: str = Field(default="professional")
    duration: int = Field(default=5, ge=1, le=30)  # seconds per slide
    resolution: str = Field(default="1080p")  # 720p, 1080p, 4k
    transition: str = Field(default="fade")
    audio_path: Optional[str] = None


# In-memory storage for slideshow jobs
slideshow_jobs: Dict[str, Any] = {}


@app.get("/api/v1/slides/status")
async def get_slideshow_status():
    """Check slideshow generation capabilities"""
    ffmpeg_status = check_ffmpeg() if SLIDESHOW_MODULES_AVAILABLE else {"available": False}

    return {
        "modules_available": SLIDESHOW_MODULES_AVAILABLE,
        "ffmpeg": ffmpeg_status,
        "zai_configured": bool(slideshow_settings.ZAI_API_KEY),
        "gateway_configured": bool(settings.OPENAI_API_KEY),
        "supported_formats": ["slides", "slideshow", "video"] if SLIDESHOW_MODULES_AVAILABLE else []
    }


@app.post("/api/v1/slides/generate")
async def generate_slides(request: SlideGenerationRequest):
    """
    Generate slides from content using AI.

    Extracts key points and generates visual slides with optional images.
    Uses Z.AI GLM-Image for image generation, with DALL-E 3 fallback.
    """
    if not SLIDESHOW_MODULES_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Slideshow modules not available. Check server logs."
        )

    if not settings.OPENAI_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="API key not configured. Set OPENAI_API_KEY in environment."
        )

    try:
        # Create options
        options = SlideGenerationOptions(
            max_slides=request.max_slides,
            style=request.style,
            brand_colors=request.brand_colors,
            include_title_slide=request.include_title_slide,
            include_conclusion_slide=request.include_conclusion_slide
        )

        # Generate slideshow content
        result = await generate_slideshow_content(
            api_key=settings.OPENAI_API_KEY,
            content=request.content,
            options=options,
            gateway_url=settings.VERCEL_AI_GATEWAY_URL,
            generate_images=request.generate_images
        )

        # Store job for retrieval
        job_id = str(uuid.uuid4())
        slideshow_jobs[job_id] = {
            "id": job_id,
            "type": "slides",
            "status": "complete",
            "result": result,
            "created_at": datetime.now().isoformat()
        }

        return {
            "success": True,
            "job_id": job_id,
            "slide_count": result.get("slide_count", 0),
            "style": result.get("style"),
            "slides": result.get("slides", [])
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Slide generation failed: {str(e)}"
        )


@app.post("/api/v1/slides/slideshow")
async def generate_slideshow(request: SlideshowGenerationRequest):
    """
    Generate a self-contained HTML slideshow from content.

    Returns HTML that can be embedded or saved as a standalone file.
    Includes autoplay, transitions, keyboard controls, and responsive design.
    """
    if not SLIDESHOW_MODULES_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Slideshow modules not available. Check server logs."
        )

    if not settings.OPENAI_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="API key not configured. Set OPENAI_API_KEY in environment."
        )

    try:
        # Step 1: Generate slides
        slide_options = SlideGenerationOptions(
            max_slides=request.max_slides,
            style=request.style,
            include_title_slide=True,
            include_conclusion_slide=True
        )

        slides_result = await generate_slideshow_content(
            api_key=settings.OPENAI_API_KEY,
            content=request.content,
            options=slide_options,
            gateway_url=settings.VERCEL_AI_GATEWAY_URL,
            generate_images=request.generate_images
        )

        slides = slides_result.get("slides", [])

        if not slides:
            raise HTTPException(
                status_code=500,
                detail="No slides generated from content"
            )

        # Step 2: Generate HTML slideshow
        slideshow_options = SlideshowOptions(
            autoplay=request.autoplay,
            duration=request.duration,
            transition=request.transition,
            show_controls=request.show_controls,
            show_progress=True,
            loop=request.loop,
            theme=request.theme
        )

        html_content = generate_html_slideshow(
            slides=slides,
            options=slideshow_options,
            title=request.title
        )

        # Store job
        job_id = str(uuid.uuid4())
        slideshow_jobs[job_id] = {
            "id": job_id,
            "type": "slideshow",
            "status": "complete",
            "html": html_content,
            "slide_count": len(slides),
            "created_at": datetime.now().isoformat()
        }

        return {
            "success": True,
            "job_id": job_id,
            "slide_count": len(slides),
            "format": "html",
            "html": html_content,
            "embed_code": f'<iframe src="/api/v1/slides/{job_id}/embed" width="800" height="450" frameborder="0" allowfullscreen></iframe>'
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Slideshow generation failed: {str(e)}"
        )


@app.post("/api/v1/slides/video")
async def generate_video(request: VideoGenerationRequest):
    """
    Generate a video from content slides using FFmpeg.

    NOTE: This endpoint requires FFmpeg to be installed on the server.
    Not available on Vercel serverless - use for local/VPS deployments.
    """
    if not SLIDESHOW_MODULES_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Slideshow modules not available. Check server logs."
        )

    if not is_ffmpeg_available():
        return {
            "success": False,
            "error": "FFmpeg is not installed on this server",
            "help": "Video generation requires FFmpeg. Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)",
            "alternative": "Use /api/v1/slides/slideshow to generate an HTML slideshow instead"
        }

    if not settings.OPENAI_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="API key not configured. Set OPENAI_API_KEY in environment."
        )

    try:
        # Step 1: Generate slides with images
        slide_options = SlideGenerationOptions(
            max_slides=request.max_slides,
            style=request.style,
            include_title_slide=True,
            include_conclusion_slide=True
        )

        slides_result = await generate_slideshow_content(
            api_key=settings.OPENAI_API_KEY,
            content=request.content,
            options=slide_options,
            gateway_url=settings.VERCEL_AI_GATEWAY_URL,
            generate_images=True  # Required for video
        )

        slides = slides_result.get("slides", [])

        if not slides:
            raise HTTPException(
                status_code=500,
                detail="No slides generated from content"
            )

        # Step 2: Generate video
        import tempfile
        output_path = tempfile.mktemp(suffix=".mp4")

        video_result = await generate_video_from_slides(
            slides=slides,
            output_path=output_path,
            duration=request.duration,
            resolution=request.resolution,
            transition=request.transition,
            audio_path=request.audio_path
        )

        if not video_result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=video_result.get("error", "Video generation failed")
            )

        # Read video file and encode as base64
        import base64
        video_data = None
        if os.path.exists(output_path):
            with open(output_path, "rb") as f:
                video_data = base64.b64encode(f.read()).decode("utf-8")
            os.remove(output_path)

        # Store job
        job_id = str(uuid.uuid4())
        slideshow_jobs[job_id] = {
            "id": job_id,
            "type": "video",
            "status": "complete",
            "video_data": video_data,
            "duration_seconds": video_result.get("duration_seconds"),
            "resolution": video_result.get("resolution"),
            "slide_count": len(slides),
            "created_at": datetime.now().isoformat()
        }

        return {
            "success": True,
            "job_id": job_id,
            "slide_count": len(slides),
            "format": "mp4",
            "duration_seconds": video_result.get("duration_seconds"),
            "resolution": video_result.get("resolution"),
            "file_size_bytes": video_result.get("file_size_bytes"),
            "video_base64": video_data
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Video generation failed: {str(e)}"
        )


@app.get("/api/v1/slides/{job_id}")
async def get_slideshow_job(job_id: str):
    """Get the result of a slideshow generation job"""
    if job_id not in slideshow_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = slideshow_jobs[job_id]

    # Return appropriate response based on job type
    if job["type"] == "slides":
        return {
            "id": job["id"],
            "type": job["type"],
            "status": job["status"],
            "slides": job.get("result", {}).get("slides", []),
            "slide_count": job.get("result", {}).get("slide_count", 0),
            "created_at": job["created_at"]
        }
    elif job["type"] == "slideshow":
        return {
            "id": job["id"],
            "type": job["type"],
            "status": job["status"],
            "html": job.get("html"),
            "slide_count": job.get("slide_count", 0),
            "created_at": job["created_at"]
        }
    elif job["type"] == "video":
        return {
            "id": job["id"],
            "type": job["type"],
            "status": job["status"],
            "video_base64": job.get("video_data"),
            "duration_seconds": job.get("duration_seconds"),
            "resolution": job.get("resolution"),
            "slide_count": job.get("slide_count", 0),
            "created_at": job["created_at"]
        }

    return job


@app.get("/api/v1/slides/{job_id}/embed")
async def get_slideshow_embed(job_id: str):
    """Get embeddable HTML for a slideshow job"""
    from fastapi.responses import HTMLResponse

    if job_id not in slideshow_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = slideshow_jobs[job_id]

    if job["type"] != "slideshow":
        raise HTTPException(
            status_code=400,
            detail="This job is not a slideshow. Use /api/v1/slides/{job_id} to get raw data."
        )

    return HTMLResponse(content=job.get("html", ""), media_type="text/html")
