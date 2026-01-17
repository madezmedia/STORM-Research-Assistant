"""
Vercel Serverless Function Handler

For Vercel Python, just export the FastAPI app directly.
Vercel's runtime handles ASGI natively without Mangum.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import the full app
try:
    from main import app
    print("Full STORM app loaded successfully")
except Exception as e:
    # Fallback to minimal app if main fails
    print(f"Failed to load main app: {e}")
    from fastapi import FastAPI
    app = FastAPI(title="STORM API (Fallback)")

    @app.get("/")
    def root():
        return {"status": "fallback", "error": str(e)}

    @app.get("/health")
    def health():
        return {"status": "healthy", "mode": "fallback", "import_error": str(e)}
