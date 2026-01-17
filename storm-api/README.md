# STORM Content Generation API

A FastAPI-based REST API for generating SEO-optimized, GEO-targeted content using the STORM (Synthesis of Topic Outlines through Retrieval and Multi-perspective Question Asking) methodology.

## Features

- **Multi-perspective STORM Analysis**: 7 distinct perspectives (Beginner, Business Owner, Local Market, Technical, Competitive, Customer, Industry Expert)
- **Research Query Generation**: SEO and GEO-aware query generation with content type modifiers
- **Web Search & Local Data Collection**: Integration with Google and Bing search APIs, plus local business/landmark/media data
- **Content Generation Engine**: Section-based writing with perspective-specific prompts
- **SEO Optimizer**: Keyword density analysis, placement tracking, heading structure validation, readability scoring
- **GEO Enhancer**: Location mention counting, local keyword usage, enhancement suggestions
- **Content Assembler**: Combines sections into a cohesive article with quality checking
- **Webhook Integration**: Support for Mad EZ Media and SonicBrand AI platforms

## Technology Stack

- **Python 3.11+**
- **FastAPI**: Modern, fast web framework
- **SQLAlchemy**: Async ORM for PostgreSQL
- **Redis**: Caching and task queue
- **Vercel AI Gateway**: Primary LLM provider (OpenAI-compatible API)
- **OpenAI**: Fallback LLM provider
- **Anthropic Claude**: Fallback LLM provider
- **BeautifulSoup**: HTML parsing
- **markdownify**: HTML to Markdown conversion

## Project Structure

```
storm-api/
├── app/
│   ├── __init__.py          # App package initialization
│   ├── llm.py              # LLM client (Vercel AI Gateway)
│   ├── analysis.py          # STORM analysis engine
│   ├── research.py          # Research query generator
│   ├── search.py            # Web search and local data collection
│   ├── generation.py        # Content generation engine
│   ├── seo.py              # SEO optimizer
│   └── geo.py              # GEO enhancer
├── migrations/
│   ├── env.py              # Alembic environment configuration
│   └── versions/
│       └── 001_initial_schema.py  # Initial database schema
├── main.py                # FastAPI application and API endpoints
├── pyproject.toml          # Project configuration and dependencies
├── CLAUDE.md              # Claude Code guidance file
└── alembic.ini            # Alembic configuration
```

## Installation

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager
- **Vercel Postgres** - Managed PostgreSQL database (set up in Vercel dashboard)
- **Vercel KV** - Managed Redis-compatible store (set up in Vercel dashboard)

### Setting Up Vercel Services

1. Go to your [Vercel Dashboard](https://vercel.com/dashboard)
2. Select your project (or create one)
3. Go to **Storage** tab
4. Click **Create Database** and select **Postgres**
5. Click **Create Database** again and select **KV**
6. Go to **Settings** > **Environment Variables** and copy the database credentials

### Setup

1. **Navigate to the API directory**:
   ```bash
   cd storm-api
   ```

2. **Create a virtual environment and install dependencies**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e .
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your Vercel database credentials and API keys
   ```

4. **Run database migrations**:
   ```bash
   uv run alembic upgrade head
   ```

5. **Start the development server**:
   ```bash
   uv run python main.py
   ```

## Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|-----------|---------|
| `DATABASE_URL` | PostgreSQL connection URL | Yes | - |
| `REDIS_URL` | Redis connection URL | Yes | - |
| `OPENAI_API_KEY` | Vercel AI Gateway key (vck_...) | Yes | - |
| `VERCEL_AI_GATEWAY_URL` | Vercel AI Gateway endpoint | No | `https://ai-gateway.vercel.sh/v1` |
| `DEFAULT_MODEL` | Default LLM model | No | `xai/grok-4.1-fast-reasoning` |
| `ANTHROPIC_API_KEY` | Anthropic API key (fallback) | No | - |
| `GOOGLE_API_KEY` | Google Generative AI API key (fallback) | No | - |
| `GOOGLE_SEARCH_API_KEY` | Google Search API key | No | - |
| `GOOGLE_SEARCH_ENGINE_ID` | Google Search Engine ID | No | - |
| `BING_SEARCH_API_KEY` | Bing Search API key | No | - |

The primary LLM provider is Vercel AI Gateway which supports multiple models including `openai/gpt-4o`, `anthropic/claude-sonnet-4.5`, `xai/grok-4.1-fast-reasoning`, and more.

## API Endpoints

### Content Briefs

#### Create Brief
```http
POST /api/v1/briefs
Content-Type: application/json
Authorization: Bearer <token>
```

#### List Briefs
```http
GET /api/v1/briefs?skip=0&limit=50
Authorization: Bearer <token>
```

#### Get Brief
```http
GET /api/v1/briefs/{brief_id}
Authorization: Bearer <token>
```

#### Update Brief
```http
PUT /api/v1/briefs/{brief_id}
Content-Type: application/json
Authorization: Bearer <token>
```

#### Delete Brief
```http
DELETE /api/v1/briefs/{brief_id}
Authorization: Bearer <token>
```

### Content Generation

#### Start Generation
```http
POST /api/v1/briefs/{brief_id}/generate
Content-Type: application/json
Authorization: Bearer <token>
```

#### Get Generation Status
```http
GET /api/v1/briefs/{brief_id}/status
Authorization: Bearer <token>
```

#### Get Generated Content
```http
GET /api/v1/content/{content_id}
Authorization: Bearer <token>
```

#### Update Content
```http
PUT /api/v1/content/{content_id}
Content-Type: application/json
Authorization: Bearer <token>
```

#### Export Content
```http
POST /api/v1/content/{content_id}/export
Content-Type: application/json
Authorization: Bearer <token>
```

### SEO Analysis

#### Get SEO Score
```http
GET /api/v1/content/{content_id}/seo
Authorization: Bearer <token>
```

#### Optimize Content
```http
POST /api/v1/content/{content_id}/optimize
Authorization: Bearer <token>
```

### Webhooks

#### Generation Complete Webhook
```http
POST /api/v1/webhooks/generation-complete
Content-Type: application/json
```

### Testing Endpoints

#### Test LLM Connection
```http
POST /api/v1/test-llm
Content-Type: application/json

{"prompt": "Hello", "model": "xai/grok-4.1-fast-reasoning"}
```

#### Test STORM Analysis
```http
POST /api/v1/test-storm
Content-Type: application/json

{"topic": "Coffee Tips", "content_type": "blog-post"}
```

## Database Schema

### ContentBriefs
- `id`: UUID (primary key)
- `user_id`: UUID (nullable)
- `topic`: String(500)
- `content_type`: String(50)
- `seo`: JSON
- `geo`: JSON
- `brand_direction`: String(50)
- `target_audience`: JSON
- `word_count`: Integer
- `tone`: String(50)
- `include_examples`: Boolean
- `include_stats`: Boolean
- `include_local_data`: Boolean
- `status`: String(50)
- `created_at`: DateTime
- `updated_at`: DateTime

### GeneratedContent
- `id`: UUID (primary key)
- `brief_id`: UUID (foreign key)
- `title`: String(500)
- `meta_description`: Text
- `content`: Text
- `word_count`: Integer
- `sections`: JSON
- `seo_score`: JSON
- `quality_score`: JSON
- `sources`: JSON
- `version`: Integer
- `created_at`: DateTime

### ResearchData
- `id`: UUID (primary key)
- `brief_id`: UUID (foreign key)
- `query`: String(500)
- `source`: String(100)
- `data`: JSON
- `created_at`: DateTime

## STORM Methodology

The STORM methodology involves:

1. **Multi-perspective Analysis**: Generate questions from 7 different perspectives
2. **Research Query Generation**: Create SEO and GEO-aware search queries
3. **Information Retrieval**: Search web and local data sources
4. **Content Synthesis**: Write sections from each perspective
5. **Content Assembly**: Combine sections into a cohesive article
6. **SEO Optimization**: Optimize for search engines
7. **GEO Enhancement**: Add local targeting elements

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=app --cov-report=html
```

### Code Style

```bash
# Format code
uv run black app/ main.py

# Check linting
uv run flake8 app/ main.py

# Type checking
uv run mypy app/ main.py
```

### Creating Migrations

```bash
# Create a new migration
uv run alembic revision --autogenerate -m "description"

# Apply migrations
uv run alembic upgrade head

# Rollback migrations
uv run alembic downgrade -1
```

## Production Deployment

### Using Docker

```bash
# Build the image
docker build -t storm-api .

# Run the container
docker run -p 8000:8000 --env-file .env storm-api
```

### Using Systemd

Create a systemd service file at `/etc/systemd/system/storm-api.service`:

```ini
[Unit]
Description=STORM Content Generation API
After=network.target

[Service]
Type=simple
User=storm
WorkingDirectory=/opt/storm-api
Environment="PATH=/opt/storm-api/venv/bin"
ExecStart=/opt/storm-api/venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Then enable and start the service:

```bash
sudo systemctl enable storm-api
sudo systemctl start storm-api
```

## Monitoring

### Health Check

```http
GET /health
```

Returns:
```json
{
  "status": "healthy",
  "timestamp": "2026-01-16T20:00:00Z"
}
```

### Metrics

```http
GET /metrics
```

Returns Prometheus-compatible metrics.

## Security

- All endpoints require authentication via Bearer token
- CORS is configured for production domains
- Rate limiting is enforced via Redis
- SQL injection protection via SQLAlchemy
- XSS protection via proper input validation

## License

MIT License - See LICENSE file for details.

## Support

For support, contact tech@madezmedia.co
