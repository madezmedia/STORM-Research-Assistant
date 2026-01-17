"""
Vercel Serverless Function Handler

This module wraps the FastAPI application for deployment on Vercel serverless functions.
Uses Mangum to adapt ASGI to AWS Lambda/Vercel serverless format.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mangum import Mangum
from main import app

# Create the handler for Vercel serverless
# lifespan="off" is used because serverless functions don't support ASGI lifespan events
handler = Mangum(app, lifespan="off")
