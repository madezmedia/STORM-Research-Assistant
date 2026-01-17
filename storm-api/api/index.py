"""
Vercel Serverless Function Handler - Ultra Minimal Debug Version
"""

# First, try the absolute minimum - just respond with JSON
def handler(event, context):
    """AWS Lambda/Vercel compatible handler"""
    import json

    # Track what imports work
    import_status = {}

    # Test basic imports
    try:
        import sys
        import_status["sys"] = "ok"
    except Exception as e:
        import_status["sys"] = str(e)

    try:
        from pathlib import Path
        import_status["pathlib"] = "ok"
    except Exception as e:
        import_status["pathlib"] = str(e)

    # Test FastAPI
    try:
        from fastapi import FastAPI
        import_status["fastapi"] = "ok"
    except Exception as e:
        import_status["fastapi"] = str(e)

    # Test Mangum
    try:
        from mangum import Mangum
        import_status["mangum"] = "ok"
    except Exception as e:
        import_status["mangum"] = str(e)

    # Test main app
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from main import app
        import_status["main"] = "ok"
    except Exception as e:
        import_status["main"] = str(e)

    # Return JSON response
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "status": "debug",
            "imports": import_status,
            "path": event.get("path", "unknown")
        })
    }
