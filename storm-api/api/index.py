"""
Vercel Serverless Function Handler

Simple inline FastAPI app for Vercel Python.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

@app.get("/")
def root():
    return {
        "name": "STORM Content Generation API",
        "version": "1.0.0",
        "status": "running",
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/api/v1/test")
def test():
    return {"message": "API is working", "endpoint": "/api/v1/test"}
