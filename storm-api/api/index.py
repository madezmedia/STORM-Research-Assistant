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

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

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

@app.post("/api/v1/test-storm", response_model=STORMTestResponse)
async def test_storm(request: STORMTestRequest):
    """Test STORM analysis with a simple topic"""

    # Generate mock perspectives
    perspectives = [
        f"Expert {i+1} perspective on {request.topic}"
        for i in range(request.perspectives)
    ]

    # Generate mock outline
    outline = {
        "title": f"Comprehensive Guide to {request.topic}",
        "sections": [
            {"heading": "Introduction", "subsections": ["Overview", "Background"]},
            {"heading": "Key Concepts", "subsections": ["Fundamentals", "Applications"]},
            {"heading": "Analysis", "subsections": ["Current State", "Future Trends"]},
            {"heading": "Conclusion", "subsections": ["Summary", "Recommendations"]},
        ]
    }

    # Generate sample content
    sample_content = f"""# Comprehensive Guide to {request.topic}

## Introduction

This guide provides an in-depth analysis of {request.topic}, examining it from {request.perspectives} different perspectives.

## Key Concepts

Understanding {request.topic} requires familiarity with several fundamental concepts...

## Analysis

Our multi-perspective analysis reveals several important insights about {request.topic}...

## Conclusion

Based on our comprehensive examination of {request.topic}, we recommend...
"""

    return STORMTestResponse(
        success=True,
        topic=request.topic,
        perspectives=perspectives,
        outline=outline,
        sample_content=sample_content
    )

# Briefs endpoint (placeholder)
class ContentBriefCreate(BaseModel):
    topic: str
    content_type: str = "blog_post"
    word_count: int = 1500
    tone: str = "professional"

class ContentBriefResponse(BaseModel):
    id: str
    topic: str
    content_type: str
    status: str

@app.post("/api/v1/briefs", response_model=ContentBriefResponse, status_code=201)
async def create_brief(brief: ContentBriefCreate):
    """Create a new content brief (placeholder - no database)"""
    import uuid
    return ContentBriefResponse(
        id=str(uuid.uuid4()),
        topic=brief.topic,
        content_type=brief.content_type,
        status="created"
    )

@app.get("/api/v1/briefs")
async def list_briefs():
    """List content briefs (placeholder - no database)"""
    return []
