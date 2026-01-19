"""
Vercel Serverless Function Handler

FastAPI app for STORM Content Generation API.
Inline implementation for Vercel compatibility.

Storage: Uses Vercel KV (Redis) for persistent storage across serverless invocations.
"""

import os
import json
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends, Security, status
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse
import asyncio
from fastapi.middleware.cors import CORSMiddleware
import httpx

# Rate limiting (optional - graceful fallback if not available)
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    RATE_LIMITING_AVAILABLE = True
except ImportError:
    print("Warning: slowapi not available - rate limiting disabled")
    RATE_LIMITING_AVAILABLE = False
    # Create dummy limiter that does nothing
    class DummyLimiter:
        def limit(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
    Limiter = None
    RateLimitExceeded = Exception
    def get_remote_address(request):
        return "127.0.0.1"

# Settings
class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    VERCEL_AI_GATEWAY_URL: str = "https://ai-gateway.vercel.sh/v1"
    DEFAULT_MODEL: str = "openai/gpt-4o-mini"
    TAVILY_API_KEY: str = ""

    # Vercel KV (Redis) Settings
    KV_REST_API_URL: str = ""
    KV_REST_API_TOKEN: str = ""

    # API Security
    STORM_API_KEY: str = ""  # Master API key for authentication
    CORS_ORIGINS: str = "http://localhost:3000"  # Comma-separated allowed origins

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()


# =============================================================================
# API Key Authentication
# =============================================================================

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> bool:
    """
    Validate API key from X-API-Key header.

    If STORM_API_KEY is not configured, authentication is disabled (dev mode).
    """
    if not settings.STORM_API_KEY:
        # No key configured = allow all (dev mode)
        return True
    if not api_key or api_key != settings.STORM_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "API key required in X-API-Key header"}
        )
    return True


# =============================================================================
# Rate Limiting
# =============================================================================

if RATE_LIMITING_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address)
else:
    limiter = DummyLimiter()


# =============================================================================
# Storage Clients (Redis + Vercel KV)
# =============================================================================

# Try to import redis for direct Redis connections
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Warning: redis package not available - using Vercel KV or memory fallback")


class RedisStorage:
    """
    Direct Redis storage client using redis-py.

    Supports Redis Labs, Upstash, or any Redis-compatible service.
    """

    def __init__(self, url: str):
        self.url = url
        self.enabled = bool(url) and REDIS_AVAILABLE
        self._client = None

    async def _get_client(self):
        if self._client is None and self.enabled:
            self._client = aioredis.from_url(self.url, decode_responses=True)
        return self._client

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            client = await self._get_client()
            result = await client.get(key)
            if result:
                return json.loads(result) if isinstance(result, str) else result
        except Exception as e:
            print(f"Redis get error: {e}")
        return None

    async def set(self, key: str, value: Dict[str, Any], ex: int = 86400) -> bool:
        if not self.enabled:
            return False
        try:
            client = await self._get_client()
            await client.set(key, json.dumps(value), ex=ex)
            return True
        except Exception as e:
            print(f"Redis set error: {e}")
        return False

    async def delete(self, key: str) -> bool:
        if not self.enabled:
            return False
        try:
            client = await self._get_client()
            await client.delete(key)
            return True
        except Exception as e:
            print(f"Redis delete error: {e}")
        return False

    async def keys(self, pattern: str = "*") -> List[str]:
        if not self.enabled:
            return []
        try:
            client = await self._get_client()
            return await client.keys(pattern)
        except Exception as e:
            print(f"Redis keys error: {e}")
        return []


class KVStorage:
    """
    Vercel KV storage client using REST API.

    Provides persistent storage for briefs, content, and slideshow jobs
    across serverless function invocations.
    """

    def __init__(self, url: str, token: str):
        self.url = url.rstrip('/') if url else ''
        self.token = token
        self.enabled = bool(url and token)

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from KV store"""
        if not self.enabled:
            return None

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.url}/get/{key}",
                    headers={"Authorization": f"Bearer {self.token}"},
                    timeout=10.0
                )
                if response.status_code == 200:
                    data = response.json()
                    result = data.get("result")
                    if result:
                        return json.loads(result) if isinstance(result, str) else result
        except Exception as e:
            print(f"KV get error: {e}")
        return None

    async def set(self, key: str, value: Dict[str, Any], ex: int = 86400) -> bool:
        """Set value in KV store with expiration (default 24 hours)"""
        if not self.enabled:
            return False

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.url}",
                    headers={
                        "Authorization": f"Bearer {self.token}",
                        "Content-Type": "application/json"
                    },
                    json=["SET", key, json.dumps(value), "EX", ex],
                    timeout=10.0
                )
                return response.status_code == 200
        except Exception as e:
            print(f"KV set error: {e}")
        return False

    async def delete(self, key: str) -> bool:
        """Delete key from KV store"""
        if not self.enabled:
            return False

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.url}",
                    headers={
                        "Authorization": f"Bearer {self.token}",
                        "Content-Type": "application/json"
                    },
                    json=["DEL", key],
                    timeout=10.0
                )
                return response.status_code == 200
        except Exception as e:
            print(f"KV delete error: {e}")
        return False

    async def keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching pattern"""
        if not self.enabled:
            return []

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.url}",
                    headers={
                        "Authorization": f"Bearer {self.token}",
                        "Content-Type": "application/json"
                    },
                    json=["KEYS", pattern],
                    timeout=10.0
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("result", [])
        except Exception as e:
            print(f"KV keys error: {e}")
        return []


# Initialize storage (prefer direct Redis, fallback to Vercel KV REST API)
redis_storage = RedisStorage(os.getenv("REDIS_URL", ""))
kv = KVStorage(settings.KV_REST_API_URL, settings.KV_REST_API_TOKEN)

# Determine which storage to use
STORAGE_TYPE = "memory"  # default
if redis_storage.enabled:
    STORAGE_TYPE = "redis"
    print(f"Using direct Redis storage")
elif kv.enabled:
    STORAGE_TYPE = "kv"
    print(f"Using Vercel KV storage")
else:
    print("Warning: No persistent storage configured - using in-memory fallback")

# Fallback in-memory storage (for local development without KV)
_memory_store: Dict[str, Dict[str, Any]] = {
    "briefs": {},
    "content": {},
    "slideshow_jobs": {}
}


async def storage_get(namespace: str, key: str) -> Optional[Dict[str, Any]]:
    """Get from Redis/KV or fallback to memory"""
    full_key = f"{namespace}:{key}"

    # Try Redis first
    if redis_storage.enabled:
        result = await redis_storage.get(full_key)
        if result is not None:
            return result

    # Try Vercel KV
    if kv.enabled:
        result = await kv.get(full_key)
        if result is not None:
            return result

    # Fallback to memory
    return _memory_store.get(namespace, {}).get(key)


async def storage_set(namespace: str, key: str, value: Dict[str, Any], ex: int = 86400) -> bool:
    """Set in Redis/KV and memory fallback"""
    full_key = f"{namespace}:{key}"

    # Always update memory for immediate reads in same invocation
    if namespace not in _memory_store:
        _memory_store[namespace] = {}
    _memory_store[namespace][key] = value

    # Try Redis first
    if redis_storage.enabled:
        return await redis_storage.set(full_key, value, ex)

    # Try Vercel KV
    if kv.enabled:
        return await kv.set(full_key, value, ex)

    return True  # Memory-only mode


async def storage_delete(namespace: str, key: str) -> bool:
    """Delete from Redis/KV and memory"""
    full_key = f"{namespace}:{key}"
    if namespace in _memory_store and key in _memory_store[namespace]:
        del _memory_store[namespace][key]

    # Try Redis first
    if redis_storage.enabled:
        return await redis_storage.delete(full_key)

    # Try Vercel KV
    if kv.enabled:
        return await kv.delete(full_key)

    return True


async def storage_list(namespace: str) -> List[Dict[str, Any]]:
    """List all items in namespace"""
    pattern = f"{namespace}:*"

    # Try Redis first
    if redis_storage.enabled:
        keys = await redis_storage.keys(pattern)
        if keys:
            items = []
            for key in keys:
                item = await redis_storage.get(key)
                if item:
                    items.append(item)
            return items

    # Try Vercel KV
    if kv.enabled:
        keys = await kv.keys(pattern)
        if keys:
            items = []
            for key in keys:
                item = await kv.get(key)
                if item:
                    items.append(item)
            return items

    # Fallback to memory
    return list(_memory_store.get(namespace, {}).values())


async def storage_keys(namespace: str) -> List[str]:
    """Get all keys in namespace (without prefix)"""
    pattern = f"{namespace}:*"
    prefix = f"{namespace}:"

    # Try Redis first
    if redis_storage.enabled:
        keys = await redis_storage.keys(pattern)
        if keys:
            return [k[len(prefix):] if k.startswith(prefix) else k for k in keys]

    # Try Vercel KV
    if kv.enabled:
        keys = await kv.keys(pattern)
        if keys:
            return [k[len(prefix):] if k.startswith(prefix) else k for k in keys]

    # Fallback to memory
    return list(_memory_store.get(namespace, {}).keys())


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

# Rate limiter setup (only if available)
if RATE_LIMITING_AVAILABLE:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS - Hardened configuration
ALLOWED_ORIGINS = [origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()]
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = ["http://localhost:3000"]  # Default fallback

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
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
async def health():
    """Health check with storage status"""
    return {
        "status": "healthy",
        "storage": {
            "type": STORAGE_TYPE,
            "redis_enabled": redis_storage.enabled,
            "kv_enabled": kv.enabled,
            "redis_url_configured": bool(os.getenv("REDIS_URL")),
            "kv_url_configured": bool(settings.KV_REST_API_URL),
        }
    }

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

# Storage uses Vercel KV with in-memory fallback (see KVStorage class above)
import uuid
from datetime import datetime

# Legacy references removed - now using storage_get/storage_set functions

# Briefs endpoint
class ContentBriefCreate(BaseModel):
    topic: str = Field(..., min_length=3, max_length=500, description="Topic to write about")
    content_type: str = Field(default="blog_post", max_length=50)
    word_count: int = Field(default=1500, ge=100, le=10000)
    tone: str = Field(default="professional", max_length=50)
    seo: Optional[Dict[str, Any]] = None
    brand_direction: Optional[str] = Field(default=None, max_length=2000)
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
@limiter.limit("60/hour")
async def create_brief(
    request: Request,
    brief: ContentBriefCreate,
    _: bool = Depends(verify_api_key)
):
    """Create a new content brief (stored in Vercel KV)"""
    brief_id = str(uuid.uuid4())
    brief_data = {
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
    await storage_set("briefs", brief_id, brief_data, ex=604800)  # 7 days
    return ContentBriefResponse(
        id=brief_id,
        topic=brief.topic,
        content_type=brief.content_type,
        status="created"
    )

@app.get("/api/v1/briefs")
@limiter.limit("120/hour")
async def list_briefs(request: Request, _: bool = Depends(verify_api_key)):
    """List content briefs (from Vercel KV)"""
    return await storage_list("briefs")

@app.get("/api/v1/briefs/{brief_id}")
@limiter.limit("120/hour")
async def get_brief(request: Request, brief_id: str, _: bool = Depends(verify_api_key)):
    """Get a single content brief by ID"""
    brief = await storage_get("briefs", brief_id)
    if not brief:
        raise HTTPException(status_code=404, detail="Brief not found")
    return ContentBriefResponse(
        id=brief.get("id", brief_id),
        topic=brief.get("topic", ""),
        content_type=brief.get("content_type", "blog post"),
        status=brief.get("status", "pending"),
        word_count=brief.get("word_count"),
        tone=brief.get("tone"),
        keywords=brief.get("keywords"),
        target_audience=brief.get("target_audience"),
        created_at=brief.get("created_at"),
    )

async def run_generation_pipeline(brief_id: str, brief: dict):
    """Background task to run the full STORM generation pipeline"""
    try:
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
        await storage_set("briefs", brief_id, brief, ex=604800)

        outline = await generate_storm_outline(topic, content_type)
        print(f"Generated outline with {len(outline.get('sections', []))} sections")

        # =========================================================================
        # PHASE 2: Research - Execute queries and gather sources (30%)
        # =========================================================================
        brief["progress"] = 30
        brief["current_phase"] = "Researching topic"
        await storage_set("briefs", brief_id, brief, ex=604800)

        research_results = []
        research_queries = outline.get("research_queries", [])[:5]  # Limit to 5 queries

        for i, query in enumerate(research_queries):
            print(f"Executing research query {i+1}/{len(research_queries)}: {query}")
            results = await search_web(query, max_results=3)
            research_results.extend(results)
            brief["progress"] = 30 + int((i + 1) / len(research_queries) * 20)
            await storage_set("briefs", brief_id, brief, ex=604800)

        print(f"Gathered {len(research_results)} research sources")

        # =========================================================================
        # PHASE 3: Content Generation - Write with research context (60%)
        # =========================================================================
        brief["progress"] = 60
        brief["current_phase"] = "Writing content"
        await storage_set("briefs", brief_id, brief, ex=604800)

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
        await storage_set("briefs", brief_id, brief, ex=604800)

        actual_word_count = len(content.split())
        sections = [
            {
                "heading": section.get("title", "Section"),
                "perspective": section.get("perspective", "General"),
                "content": "..."
            }
            for section in outline.get("sections", [])
        ]
        sources = list(set([r.get("url", "") for r in research_results if r.get("url")]))

        content_id = str(uuid.uuid4())
        content_data = {
            "id": content_id,
            "brief_id": brief_id,
            "title": outline.get("title", topic),
            "meta_description": outline.get("meta_description", ""),
            "content": content,
            "word_count": actual_word_count,
            "sections": sections,
            "sources": sources[:10],
            "research_queries": research_queries,
            "seo_score": {
                "overall": 85 if research_results else 60,
                "readability": 90,
                "keyword_density": 80,
                "sources_cited": len(sources)
            },
            "created_at": datetime.now().isoformat()
        }
        await storage_set("content", brief_id, content_data, ex=604800)

        brief["status"] = "complete"
        brief["progress"] = 100
        brief["current_phase"] = "Complete"
        brief["outline"] = outline
        await storage_set("briefs", brief_id, brief, ex=604800)
        print(f"Generation complete for brief {brief_id}")

    except Exception as e:
        print(f"Generation failed for brief {brief_id}: {e}")
        brief["status"] = "failed"
        brief["current_phase"] = f"Error: {str(e)}"
        await storage_set("briefs", brief_id, brief, ex=604800)


@app.post("/api/v1/briefs/{brief_id}/generate", response_model=GenerationStatusResponse)
@limiter.limit("60/hour")
async def generate_content_endpoint(
    request: Request,
    brief_id: str,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_api_key)
):
    """Start STORM content generation for a brief (runs in background)"""
    brief = await storage_get("briefs", brief_id)
    if not brief:
        raise HTTPException(status_code=404, detail="Brief not found")

    # Check if already generating or complete
    if brief.get("status") in ["generating", "complete"]:
        return GenerationStatusResponse(
            brief_id=brief_id,
            status=brief.get("status", "unknown"),
            progress=brief.get("progress", 0),
            current_phase=brief.get("current_phase", "Unknown"),
            estimated_time_remaining=0 if brief.get("status") == "complete" else 30
        )

    # Set initial status and start background task
    brief["status"] = "generating"
    brief["progress"] = 5
    brief["current_phase"] = "Starting generation"
    await storage_set("briefs", brief_id, brief, ex=604800)

    # Run generation in background
    asyncio.create_task(run_generation_pipeline(brief_id, brief))

    return GenerationStatusResponse(
        brief_id=brief_id,
        status="generating",
        progress=5,
        current_phase="Starting generation",
        estimated_time_remaining=30
    )


@app.get("/api/v1/briefs/{brief_id}/status", response_model=GenerationStatusResponse)
@limiter.limit("120/hour")
async def get_generation_status(request: Request, brief_id: str, _: bool = Depends(verify_api_key)):
    """Get the status of content generation (from Vercel KV)"""
    brief = await storage_get("briefs", brief_id)
    if not brief:
        raise HTTPException(status_code=404, detail="Brief not found")

    return GenerationStatusResponse(
        brief_id=brief_id,
        status=brief.get("status", "unknown"),
        progress=brief.get("progress", 0),
        current_phase=brief.get("current_phase", "Unknown"),
        estimated_time_remaining=0 if brief.get("status") == "complete" else 5
    )


@app.get("/api/v1/briefs/{brief_id}/stream")
@limiter.limit("120/hour")
async def stream_generation_status(request: Request, brief_id: str, _: bool = Depends(verify_api_key)):
    """
    Stream generation progress via Server-Sent Events (SSE).

    Provides real-time updates without polling. Client should use EventSource API.
    """
    async def event_generator():
        last_status = None
        consecutive_errors = 0
        max_errors = 5

        while True:
            try:
                brief = await storage_get("briefs", brief_id)

                if not brief:
                    yield f"data: {json.dumps({'error': 'Brief not found', 'status': 'failed'})}\n\n"
                    break

                current_status = {
                    "brief_id": brief_id,
                    "status": brief.get("status", "unknown"),
                    "progress": brief.get("progress", 0),
                    "current_phase": brief.get("current_phase", "Unknown"),
                    "estimated_time_remaining": 0 if brief.get("status") == "complete" else 5
                }

                # Only send if status changed (or first update)
                if current_status != last_status:
                    yield f"data: {json.dumps(current_status)}\n\n"
                    last_status = current_status

                # Stop streaming when generation completes or fails
                if current_status["status"] in ["complete", "completed", "failed"]:
                    break

                consecutive_errors = 0
                await asyncio.sleep(1)  # Check every second

            except Exception as e:
                consecutive_errors += 1
                yield f"data: {json.dumps({'error': str(e), 'status': 'error'})}\n\n"
                if consecutive_errors >= max_errors:
                    break
                await asyncio.sleep(2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Access-Control-Allow-Origin": "*"
        }
    )


@app.get("/api/v1/content/{brief_id}", response_model=GeneratedContentResponse)
@limiter.limit("120/hour")
async def get_generated_content(request: Request, brief_id: str, _: bool = Depends(verify_api_key)):
    """Get the generated content for a brief (from Vercel KV)"""
    content = await storage_get("content", brief_id)
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")

    return content


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

# =============================================================================
# GEO Pipeline Models & Imports
# =============================================================================

# Import GEO pipeline modules
try:
    from app.geo_pipeline import (
        GEOPipeline, PromptGenerator, KeywordTracker
    )
    GEO_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"GEO modules not available: {e}")
    GEO_MODULES_AVAILABLE = False


# GEO Pydantic Models
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
    keywords: List[tuple] = []  # [(keyword, score), ...]
    topics: List[str] = []
    questions: List[str] = []
    related_searches: List[str] = []
    autocomplete: List[str] = []
    pages_analyzed: int = 0
    content_summary: str = ""
    status: str
    created_at: str = ""


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


# =============================================================================
# Slideshow/Video Generation Models & Imports
# =============================================================================

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
    content: str = Field(..., min_length=10, max_length=50000, description="Blog post or article content (max 50KB)")
    max_slides: int = Field(default=10, ge=1, le=20)
    style: str = Field(default="professional", max_length=50)  # professional, minimalist, vibrant
    brand_colors: Optional[Dict[str, str]] = None  # {primary: "#hex", secondary: "#hex"}
    include_title_slide: bool = True
    include_conclusion_slide: bool = True
    generate_images: bool = True  # Set False for text-only slides


class SlideshowGenerationRequest(BaseModel):
    """Request to generate HTML slideshow"""
    content: str = Field(..., min_length=10, max_length=50000, description="Blog post or article content (max 50KB)")
    max_slides: int = Field(default=10, ge=1, le=20)
    style: str = Field(default="professional", max_length=50)
    autoplay: bool = True
    duration: int = Field(default=5, ge=1, le=30)  # seconds per slide
    transition: str = Field(default="fade", max_length=20)  # fade, slide, zoom
    show_controls: bool = True
    loop: bool = True
    theme: str = Field(default="dark", max_length=20)  # dark, light
    title: str = Field(default="Presentation", max_length=200)
    generate_images: bool = True


class VideoGenerationRequest(BaseModel):
    """Request to generate video from content"""
    content: str = Field(..., min_length=10, max_length=50000, description="Blog post or article content (max 50KB)")
    max_slides: int = Field(default=10, ge=1, le=20)
    style: str = Field(default="professional", max_length=50)
    duration: int = Field(default=5, ge=1, le=30)  # seconds per slide
    resolution: str = Field(default="1080p", max_length=10)  # 720p, 1080p, 4k
    transition: str = Field(default="fade", max_length=20)
    audio_path: Optional[str] = Field(default=None, max_length=500)


# Slideshow jobs storage uses Vercel KV (see storage_get/storage_set above)


class BriefSlidesRequest(BaseModel):
    """Request to generate slides from a brief's generated content"""
    max_slides: int = Field(default=10, ge=1, le=20)
    style: str = Field(default="professional")  # professional, minimalist, vibrant
    generate_images: bool = Field(default=True)
    autoplay: bool = Field(default=True)
    duration: int = Field(default=5, ge=1, le=30)  # seconds per slide
    transition: str = Field(default="fade")  # fade, slide, zoom
    theme: str = Field(default="dark")  # dark, light


@app.post("/api/v1/briefs/{brief_id}/slides")
@limiter.limit("60/hour")
async def generate_slides_from_brief(
    request: Request,
    brief_id: str,
    slides_request: BriefSlidesRequest,
    _: bool = Depends(verify_api_key)
):
    """
    Generate slideshow directly from a brief's generated content.

    This endpoint combines content retrieval and slide generation into one step.
    The brief must have completed content generation first.
    """
    if not SLIDESHOW_MODULES_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Slideshow modules not available. Check server logs."
        )

    # 1. Get the generated content for this brief
    content_data = await storage_get("content", brief_id)
    if not content_data:
        # Check if brief exists but content hasn't been generated
        brief = await storage_get("briefs", brief_id)
        if brief:
            raise HTTPException(
                status_code=400,
                detail=f"Content not yet generated for brief {brief_id}. Status: {brief.get('status', 'unknown')}"
            )
        raise HTTPException(status_code=404, detail="Brief not found")

    content_text = content_data.get("content", "")
    content_title = content_data.get("title", "Presentation")

    if not content_text:
        raise HTTPException(status_code=400, detail="Brief content is empty")

    # 2. Generate slides using the existing pipeline
    try:
        # Create options object for slide generation
        slide_options = SlideGenerationOptions(
            max_slides=slides_request.max_slides,
            style=slides_request.style,
            include_title_slide=True,
            include_conclusion_slide=True
        )

        slides_result = await generate_slideshow_content(
            api_key=settings.OPENAI_API_KEY,
            content=content_text,
            options=slide_options,
            gateway_url=settings.VERCEL_AI_GATEWAY_URL,
            generate_images=slides_request.generate_images,
            zai_api_key=slideshow_settings.ZAI_API_KEY
        )

        # 3. Generate HTML slideshow
        slideshow_options = SlideshowOptions(
            autoplay=slides_request.autoplay,
            duration=slides_request.duration,
            transition=slides_request.transition,
            theme=slides_request.theme,
            show_controls=True,
            loop=True
        )

        html_content = generate_html_slideshow(
            slides=slides_result.get("slides", []),
            options=slideshow_options,
            title=content_title
        )

        embed_code = generate_embed_code(html_content, width=800, height=450)

        # 4. Store job result
        job_id = str(uuid.uuid4())
        job_data = {
            "id": job_id,
            "brief_id": brief_id,
            "type": "slideshow",
            "status": "complete",
            "slides": slides_result.get("slides", []),
            "html": html_content,
            "embed_code": embed_code,
            "title": content_title,
            "created_at": datetime.utcnow().isoformat()
        }
        await storage_set("slideshow_jobs", job_id, job_data, ex=604800)  # 7 days

        return {
            "job_id": job_id,
            "brief_id": brief_id,
            "status": "complete",
            "slide_count": len(slides_result.get("slides", [])),
            "title": content_title,
            "html": html_content,
            "embed_code": embed_code,
            "slides": slides_result.get("slides", [])
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Slide generation failed: {str(e)}"
        )


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
@limiter.limit("60/hour")
async def generate_slides(
    request: Request,
    slide_request: SlideGenerationRequest,
    _: bool = Depends(verify_api_key)
):
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
            max_slides=slide_request.max_slides,
            style=slide_request.style,
            brand_colors=slide_request.brand_colors,
            include_title_slide=slide_request.include_title_slide,
            include_conclusion_slide=slide_request.include_conclusion_slide
        )

        # Generate slideshow content
        result = await generate_slideshow_content(
            api_key=settings.OPENAI_API_KEY,
            content=slide_request.content,
            options=options,
            gateway_url=settings.VERCEL_AI_GATEWAY_URL,
            generate_images=slide_request.generate_images,
            zai_api_key=slideshow_settings.ZAI_API_KEY
        )

        # Store job for retrieval in KV
        job_id = str(uuid.uuid4())
        job_data = {
            "id": job_id,
            "type": "slides",
            "status": "complete",
            "result": result,
            "created_at": datetime.now().isoformat()
        }
        await storage_set("slideshow_jobs", job_id, job_data, ex=86400)  # 24 hours

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
@limiter.limit("60/hour")
async def generate_slideshow(
    request: Request,
    slideshow_request: SlideshowGenerationRequest,
    _: bool = Depends(verify_api_key)
):
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
            max_slides=slideshow_request.max_slides,
            style=slideshow_request.style,
            include_title_slide=True,
            include_conclusion_slide=True
        )

        slides_result = await generate_slideshow_content(
            api_key=settings.OPENAI_API_KEY,
            content=slideshow_request.content,
            options=slide_options,
            gateway_url=settings.VERCEL_AI_GATEWAY_URL,
            generate_images=slideshow_request.generate_images,
            zai_api_key=slideshow_settings.ZAI_API_KEY
        )

        slides = slides_result.get("slides", [])

        if not slides:
            raise HTTPException(
                status_code=500,
                detail="No slides generated from content"
            )

        # Step 2: Generate HTML slideshow
        slideshow_options = SlideshowOptions(
            autoplay=slideshow_request.autoplay,
            duration=slideshow_request.duration,
            transition=slideshow_request.transition,
            show_controls=slideshow_request.show_controls,
            show_progress=True,
            loop=slideshow_request.loop,
            theme=slideshow_request.theme
        )

        html_content = generate_html_slideshow(
            slides=slides,
            options=slideshow_options,
            title=slideshow_request.title
        )

        # Generate embed code
        embed_code = generate_embed_code(html_content, width=800, height=450)

        # Store job in KV with complete data
        job_id = str(uuid.uuid4())
        job_data = {
            "id": job_id,
            "type": "slideshow",
            "status": "complete",
            "title": slideshow_request.title,
            "html": html_content,
            "embed_code": embed_code,
            "slides": slides,
            "slide_count": len(slides),
            "style": slideshow_request.style,
            "theme": slideshow_request.theme,
            "created_at": datetime.now().isoformat()
        }
        await storage_set("slideshow_jobs", job_id, job_data, ex=604800)  # 7 days

        return {
            "success": True,
            "job_id": job_id,
            "slide_count": len(slides),
            "format": "html",
            "title": request.title,
            "html": html_content,
            "embed_code": embed_code,
            "slides": slides
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Slideshow generation failed: {str(e)}"
        )


@app.post("/api/v1/slides/video")
@limiter.limit("10/hour")
async def generate_video(
    request: Request,
    video_request: VideoGenerationRequest,
    _: bool = Depends(verify_api_key)
):
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
            max_slides=video_request.max_slides,
            style=video_request.style,
            include_title_slide=True,
            include_conclusion_slide=True
        )

        slides_result = await generate_slideshow_content(
            api_key=settings.OPENAI_API_KEY,
            content=video_request.content,
            options=slide_options,
            gateway_url=settings.VERCEL_AI_GATEWAY_URL,
            generate_images=True,  # Required for video
            zai_api_key=slideshow_settings.ZAI_API_KEY
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
            duration=video_request.duration,
            resolution=video_request.resolution,
            transition=video_request.transition,
            audio_path=video_request.audio_path
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

        # Store job in KV (video data may be large, shorter TTL)
        job_id = str(uuid.uuid4())
        job_data = {
            "id": job_id,
            "type": "video",
            "status": "complete",
            "video_data": video_data,
            "duration_seconds": video_result.get("duration_seconds"),
            "resolution": video_result.get("resolution"),
            "slide_count": len(slides),
            "created_at": datetime.now().isoformat()
        }
        await storage_set("slideshow_jobs", job_id, job_data, ex=3600)  # 1 hour (videos are large)

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


@app.get("/api/v1/slides")
@limiter.limit("120/hour")
async def list_slideshows(request: Request, _: bool = Depends(verify_api_key)):
    """List all stored slideshows"""
    try:
        jobs = await storage_keys("slideshow_jobs")
        slideshows = []

        for job_id in jobs:
            job = await storage_get("slideshow_jobs", job_id)
            if job and job.get("type") == "slideshow":
                # Return summary without full HTML to keep response small
                slideshows.append({
                    "id": job.get("id"),
                    "title": job.get("title", "Untitled"),
                    "slide_count": job.get("slide_count", 0),
                    "style": job.get("style", "professional"),
                    "theme": job.get("theme", "dark"),
                    "brief_id": job.get("brief_id"),
                    "created_at": job.get("created_at")
                })

        # Sort by created_at descending (newest first)
        slideshows.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return {
            "slideshows": slideshows,
            "total": len(slideshows)
        }
    except Exception as e:
        print(f"Error listing slideshows: {e}")
        return {"slideshows": [], "total": 0}


@app.get("/api/v1/slides/{job_id}")
@limiter.limit("120/hour")
async def get_slideshow_job(request: Request, job_id: str, _: bool = Depends(verify_api_key)):
    """Get the result of a slideshow generation job (from Vercel KV)"""
    job = await storage_get("slideshow_jobs", job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

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
            "title": job.get("title", "Untitled"),
            "html": job.get("html"),
            "embed_code": job.get("embed_code"),
            "slides": job.get("slides", []),
            "slide_count": job.get("slide_count", 0),
            "style": job.get("style"),
            "theme": job.get("theme"),
            "brief_id": job.get("brief_id"),
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
@limiter.limit("120/hour")
async def get_slideshow_embed(request: Request, job_id: str, _: bool = Depends(verify_api_key)):
    """Get embeddable HTML for a slideshow job (from Vercel KV)"""
    from fastapi.responses import HTMLResponse

    job = await storage_get("slideshow_jobs", job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["type"] != "slideshow":
        raise HTTPException(
            status_code=400,
            detail="This job is not a slideshow. Use /api/v1/slides/{job_id} to get raw data."
        )

    return HTMLResponse(content=job.get("html", ""), media_type="text/html")


# =============================================================================
# GEO Pipeline Endpoints
# =============================================================================

@app.post("/api/v1/geo/analyze-website", response_model=GEOAnalysisResponse, status_code=202)
@limiter.limit("60/hour")
async def analyze_website(
    request: Request,
    geo_request: GEOAnalyzeRequest,
    _: bool = Depends(verify_api_key)
):
    """
    Analyze a website to extract keywords, topics, and generate monitoring prompts.

    Uses Vercel KV for storage. Analysis runs synchronously (no background tasks in serverless).
    """
    import uuid
    from datetime import datetime

    if not GEO_MODULES_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="GEO pipeline modules not available"
        )

    # Create analysis ID
    analysis_id = str(uuid.uuid4())
    domain = geo_request.url.split("//")[-1].split("/")[0]

    # Create initial analysis record
    analysis_data = {
        "id": analysis_id,
        "brand_name": geo_request.brand_name,
        "domain": domain,
        "competitors": geo_request.competitors,
        "status": "analyzing",
        "keywords": [],
        "topics": [],
        "questions": [],
        "related_searches": [],
        "autocomplete": [],
        "pages_analyzed": 0,
        "content_summary": "",
        "prompts": [],
        "created_at": datetime.now().isoformat()
    }

    # Store initial state
    await storage_set("geo_analyses", analysis_id, analysis_data, ex=604800)  # 7 days

    try:
        # Run the GEO pipeline (synchronous in serverless)
        pipeline = GEOPipeline(brand_name=geo_request.brand_name, use_keybert=True)
        results = await pipeline.run_full_pipeline(
            url=geo_request.url,
            crawl_limit=geo_request.crawl_limit,
            competitors=geo_request.competitors,
        )

        # Update analysis with results
        analysis_data.update({
            "keywords": results["analysis"]["keywords"],
            "topics": results["analysis"]["topics"],
            "questions": results["research"]["paa_questions"],
            "related_searches": results["research"]["related_searches"],
            "autocomplete": results["research"]["autocomplete"],
            "prompts": results["prompts"],
            "content_summary": results["analysis"]["content_summary"],
            "pages_analyzed": results["analysis"]["pages_analyzed"],
            "status": "complete"
        })
        await storage_set("geo_analyses", analysis_id, analysis_data, ex=604800)

    except Exception as e:
        import traceback
        analysis_data["status"] = "failed"
        analysis_data["error_message"] = f"{str(e)}\n{traceback.format_exc()}"
        await storage_set("geo_analyses", analysis_id, analysis_data, ex=604800)

    return GEOAnalysisResponse(
        id=analysis_id,
        brand_name=geo_request.brand_name,
        domain=domain,
        keywords=analysis_data.get("keywords", []),
        topics=analysis_data.get("topics", []),
        questions=analysis_data.get("questions", []),
        related_searches=analysis_data.get("related_searches", []),
        autocomplete=analysis_data.get("autocomplete", []),
        pages_analyzed=analysis_data.get("pages_analyzed", 0),
        content_summary=analysis_data.get("content_summary", ""),
        status=analysis_data["status"],
        created_at=analysis_data["created_at"]
    )


@app.get("/api/v1/geo/analysis/{analysis_id}", response_model=GEOAnalysisResponse)
@limiter.limit("120/hour")
async def get_geo_analysis(request: Request, analysis_id: str, _: bool = Depends(verify_api_key)):
    """Get GEO analysis results by ID (from Vercel KV)"""
    analysis = await storage_get("geo_analyses", analysis_id)

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return GEOAnalysisResponse(
        id=analysis.get("id", analysis_id),
        brand_name=analysis.get("brand_name", ""),
        domain=analysis.get("domain", ""),
        keywords=analysis.get("keywords", []),
        topics=analysis.get("topics", []),
        questions=analysis.get("questions", []),
        related_searches=analysis.get("related_searches", []),
        autocomplete=analysis.get("autocomplete", []),
        pages_analyzed=analysis.get("pages_analyzed", 0),
        content_summary=analysis.get("content_summary", ""),
        status=analysis.get("status", "unknown"),
        created_at=analysis.get("created_at", "")
    )


@app.post("/api/v1/geo/generate-prompts", response_model=GEOPromptsResponse)
@limiter.limit("120/hour")
async def generate_geo_prompts(
    request: Request,
    prompt_request: GEOPromptRequest,
    _: bool = Depends(verify_api_key)
):
    """
    Generate LLM monitoring prompts for brand tracking.

    Returns prompts in various categories:
    - direct_mention: Questions about the brand
    - comparison: Brand vs competitors
    - recommendation: Category recommendations
    - feature_inquiry: Feature-specific questions
    """
    if not GEO_MODULES_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="GEO pipeline modules not available"
        )

    generator = PromptGenerator(brand_name=prompt_request.brand_name)
    prompts = generator.generate_prompts(
        keywords=prompt_request.keywords,
        questions=prompt_request.questions,
        competitors=prompt_request.competitors,
    )

    return GEOPromptsResponse(
        brand_name=prompt_request.brand_name,
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
@limiter.limit("120/hour")
async def track_llm_response(
    request: Request,
    track_request: GEOTrackRequest,
    _: bool = Depends(verify_api_key)
):
    """
    Track brand and keyword mentions in an LLM response.

    Analyzes the response for:
    - Direct brand mentions
    - Keyword occurrences
    - Position tracking (earlier = better)
    """
    if not GEO_MODULES_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="GEO pipeline modules not available"
        )

    tracker = KeywordTracker(
        brand_name=track_request.brand_name,
        keywords=track_request.keywords,
    )

    results = tracker.analyze_response(
        response_text=track_request.response_text,
        prompt=track_request.prompt,
    )

    return GEOTrackResponse(
        brand_mentioned=results["brand_mentioned"],
        brand_mentions=results["brand_mentions"],
        keyword_mentions=results["keyword_mentions"],
        total_mentions=results["total_mentions"],
        position_score=results["position_score"],
    )


@app.get("/api/v1/geo/keywords/{analysis_id}")
@limiter.limit("120/hour")
async def get_geo_keywords(request: Request, analysis_id: str, _: bool = Depends(verify_api_key)):
    """Get extracted keywords from a GEO analysis (from Vercel KV)"""
    analysis = await storage_get("geo_analyses", analysis_id)

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return {
        "analysis_id": analysis.get("id", analysis_id),
        "brand_name": analysis.get("brand_name", ""),
        "keywords": analysis.get("keywords", []),
        "topics": analysis.get("topics", []),
    }


@app.post("/api/v1/geo/export-gego", response_model=GEOExportResponse)
@limiter.limit("120/hour")
async def export_gego_prompts(
    request: Request,
    export_request: GEOExportRequest,
    _: bool = Depends(verify_api_key)
):
    """
    Export prompts in GEGO-compatible JSON format (from Vercel KV).
    """
    analysis = await storage_get("geo_analyses", export_request.analysis_id)

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    prompts = analysis.get("prompts", [])

    return GEOExportResponse(
        brand_name=analysis.get("brand_name", ""),
        prompt_count=len(prompts),
        prompts=prompts,
    )


@app.get("/api/v1/geo/analyses")
@limiter.limit("120/hour")
async def list_geo_analyses(
    request: Request,
    skip: int = 0,
    limit: int = 50,
    brand_name: Optional[str] = None,
    _: bool = Depends(verify_api_key)
):
    """List all GEO analyses (from Vercel KV)"""
    # Get all analyses from KV
    all_analyses = await storage_list("geo_analyses")

    # Filter by brand_name if specified
    if brand_name:
        all_analyses = [a for a in all_analyses if a.get("brand_name") == brand_name]

    # Sort by created_at descending
    all_analyses.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    # Apply pagination
    paginated = all_analyses[skip:skip + limit]

    return {
        "count": len(paginated),
        "total": len(all_analyses),
        "analyses": [
            {
                "id": a.get("id"),
                "brand_name": a.get("brand_name"),
                "domain": a.get("domain"),
                "status": a.get("status"),
                "pages_analyzed": a.get("pages_analyzed", 0),
                "keyword_count": len(a.get("keywords", [])),
                "created_at": a.get("created_at", ""),
            }
            for a in paginated
        ],
    }
