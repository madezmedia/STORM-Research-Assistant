# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

STORM Content Generation API - A FastAPI backend for generating SEO-optimized, GEO-targeted content using the STORM methodology (Synthesis of Topic Outlines through Retrieval and Multi-perspective Question Asking). Built for Mad EZ Media / SonicBrand AI integration.

## Common Commands

### Development Server
```bash
cd storm-api
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# OR
uv run python main.py
```

### Testing
```bash
uv run pytest                                    # Run all tests
uv run pytest tests/unit_tests/test_utils.py    # Run specific test file
uv run pytest --cov=app --cov-report=html       # Run with coverage
```

### Code Quality
```bash
uv run black app/ main.py    # Format code
uv run flake8 app/ main.py   # Lint
uv run mypy app/ main.py     # Type check
```

### Database Migrations (Alembic)
```bash
uv run alembic upgrade head                           # Apply all migrations
uv run alembic downgrade -1                           # Rollback one migration
uv run alembic revision --autogenerate -m "message"   # Create new migration
```

### Parent Project (from root)
```bash
make test                      # Run unit tests
make test TEST_FILE=<path>     # Run specific test
make lint                      # Run ruff + mypy
make format                    # Format with ruff
```

## Architecture

### Content Generation Pipeline

The system follows a sequential pipeline for content generation:

1. **Brief Creation** (`POST /api/v1/briefs`) → Stores topic, SEO/GEO config, audience
2. **STORM Analysis** (`app/analysis.py`) → Generates 7-perspective outline with research queries
3. **Research** (`app/research.py`) → Creates prioritized search queries
4. **Web Search** (`app/search.py`) → Fetches results from Google/Bing, collects local data
5. **Content Generation** (`app/generation.py`) → Writes sections using LLM, runs in parallel
6. **SEO Optimization** (`app/seo.py`) → Analyzes keyword density, placement, readability
7. **GEO Enhancement** (`app/geo.py`) → Adds location mentions, local keywords
8. **Storage** → Saves to `generated_content` table, sends webhook

### Key Modules

| Module | Purpose |
|--------|---------|
| `main.py` | FastAPI app, routes, DB models, background tasks |
| `app/llm.py` | Vercel AI Gateway client (OpenAI-compatible API) |
| `app/analysis.py` | STORM 7-perspective outline generation |
| `app/generation.py` | Section writers, content assembly |
| `app/search.py` | Google/Bing search clients, local data models |
| `app/seo.py` | Keyword density, placement, readability scoring |
| `app/geo.py` | Location mentions, local keyword tracking |

### LLM Integration

Uses Vercel AI Gateway at `https://ai-gateway.vercel.sh/v1` with OpenAI-compatible API:
- Primary model: `xai/grok-4.1-fast-reasoning`
- Fallback models: `openai/gpt-4o`, `anthropic/claude-sonnet-4.5`
- Key functions in `app/llm.py`: `generate_text()`, `generate_json()`, `generate_with_messages()`

### Database Schema

Three main tables (PostgreSQL via SQLAlchemy async):
- **content_briefs**: Topic, SEO/GEO config, audience, status
- **generated_content**: Title, content, sections, scores, sources (FK to brief)
- **research_data**: Query results cached per brief

### Background Processing

Content generation runs via `asyncio.create_task()` in `run_generation_pipeline()`. Status tracked in Redis with key pattern `generation:{brief_id}`.

## Environment Variables

Required in `.env`:
```
DATABASE_URL=postgresql+asyncpg://...    # Supabase/Vercel Postgres
REDIS_URL=redis://...                     # Redis Labs/Vercel KV
OPENAI_API_KEY=vck_...                    # Vercel AI Gateway key
VERCEL_AI_GATEWAY_URL=https://ai-gateway.vercel.sh/v1
DEFAULT_MODEL=xai/grok-4.1-fast-reasoning
```

## Testing Endpoints

```bash
# Test LLM connection
curl -X POST http://localhost:8000/api/v1/test-llm \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello"}'

# Test STORM analysis
curl -X POST http://localhost:8000/api/v1/test-storm \
  -H "Content-Type: application/json" \
  -d '{"topic": "Coffee Tips", "content_type": "blog-post"}'
```

## STORM 7 Perspectives

The analysis engine generates content outlines from these perspectives:
1. Beginner - Foundational concepts, terminology
2. Business Owner - ROI, costs, implementation
3. Local Market - Regulations, competitors, success stories
4. Technical - How it works, requirements, integrations
5. Competitive - Alternatives, comparisons, differentiators
6. Customer - Pain points, decision factors, UX
7. Industry Expert - Best practices, trends, future outlook
