"""
Vercel Serverless Function Handler

Minimal version to debug import issues.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)

# Try to import the full app, fallback to minimal if it fails
try:
    from main import app
    print("Main app imported successfully")
except Exception as e:
    print(f"Failed to import main app: {e}")
    # Create minimal fallback app
    from fastapi import FastAPI
    app = FastAPI(title="STORM API (Minimal)")

    @app.get("/")
    def root():
        return {"status": "minimal", "error": str(e)}

    @app.get("/health")
    def health():
        return {"status": "healthy", "mode": "minimal", "import_error": str(e)}

# Import Mangum
try:
    from mangum import Mangum
    handler = Mangum(app, lifespan="off")
    print("Mangum handler created")
except Exception as e:
    print(f"Failed to create Mangum handler: {e}")
    # Try without Mangum
    handler = app
