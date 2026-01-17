"""
Vercel Serverless Function Handler

For Vercel Python, just export the FastAPI app directly.
Vercel's runtime handles ASGI natively without Mangum.
"""

from fastapi import FastAPI

# Create minimal FastAPI app
app = FastAPI(title="STORM API")

@app.get("/")
def root():
    return {"status": "running", "mode": "minimal"}

@app.get("/health")
def health():
    return {"status": "healthy", "mode": "minimal"}
