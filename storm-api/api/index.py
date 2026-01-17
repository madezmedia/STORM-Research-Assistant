"""
Vercel Serverless Function Handler - Minimal Test
"""

from fastapi import FastAPI
from mangum import Mangum

# Create minimal FastAPI app
app = FastAPI(title="STORM API Debug")

@app.get("/")
def root():
    return {"status": "running", "mode": "minimal_debug"}

@app.get("/health")
def health():
    return {"status": "healthy", "mode": "minimal_debug"}

# Mangum handler for Vercel
handler = Mangum(app, lifespan="off")
