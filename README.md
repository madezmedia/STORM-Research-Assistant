# 🌪️ STORM Research Assistant

<!-- Project badges -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Vercel AI Gateway](https://img.shields.io/badge/Vercel%20AI%20Gateway-black.svg)](https://vercel.com/docs/ai-gateway)
[![Next.js](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **STORM** (Synthesis of Topic Outlines through Retrieval and Multi-perspective Question Asking) - A writing system for generating grounded and organized long-form articles from scratch, with comparable breadth and depth to Wikipedia pages

## 📖 Overview

STORM Research Assistant is a LangGraph-based implementation of the STORM methodology from Stanford, designed to write grounded and organized long-form articles from scratch. The system models the pre-writing stage by (1) discovering diverse perspectives for researching the given topic, (2) simulating conversations where writers with different perspectives pose questions to a topic expert grounded on trusted Internet sources, and (3) curating the collected information to create an outline before generating the final article.

### 🎯 Key Features

- **🔍 Pre-writing Stage Modeling**: Comprehensive research and outline preparation before article generation
- **🤖 Diverse Perspective Discovery**: Automatic generation of multiple expert perspectives for comprehensive topic coverage
- **💬 Simulated Expert Conversations**: Multi-perspective question asking with grounded answers from trusted sources
- **📚 Grounded Information**: All content backed by reliable Internet sources (Tavily web search and ArXiv papers)
- **📊 Structured Outline Creation**: Systematic curation of collected information into organized outlines
- **✏️ Long-form Article Generation**: Wikipedia-quality articles with introduction, detailed sections, and conclusion
- **🔄 User Feedback Integration**: Human-in-the-loop capability for refining analyst perspectives
- **⚡ Parallel Processing**: Simultaneous execution of multiple perspective interviews for efficiency
- **🎨 LangGraph Studio Support**: Full integration with LangGraph Studio for visual debugging
- **🌐 Web Frontend**: Next.js + Vercel AI SDK frontend for interactive research experience

## 🏗️ Architecture

### System Structure

```
📁 storm-api/                  # FastAPI Content Generation API
├── 📄 main.py              # FastAPI application & endpoints
├── 📁 app/                 # Core modules
│   ├── 🧠 llm.py           # Vercel AI Gateway client
│   ├── 📊 analysis.py      # STORM 7-perspective analysis
│   ├── ✏️ generation.py    # Content generation engine
│   ├── 🔍 research.py      # Research query generator
│   ├── 🌐 search.py        # Web search & local data
│   ├── 📈 seo.py           # SEO optimizer
│   └── 📍 geo.py           # GEO enhancer
├── 📁 migrations/          # Alembic database migrations
└── 📄 pyproject.toml       # Dependencies & config

📁 frontend/                   # Next.js + Vercel AI SDK frontend
├── 📄 package.json         # Frontend dependencies
├── 📁 app/                 # Next.js app directory
│   ├── 📄 layout.tsx       # Root layout
│   ├── 📄 page.tsx         # Main interface
│   └── 📄 globals.css      # Global styles
├── 📁 components/          # UI components
└── 📄 .env.local.example   # Environment variables template

📁 src/storm_research/         # LangGraph backend (optional)
├── 🧠 graph.py             # LangGraph graph definition
├── 📊 state.py             # State definitions
├── 💬 prompts.py           # Prompt templates
└── 🔧 tools.py             # Search tools
```

### Backend Workflow

```mermaid
graph TD
    A[Start] --> B[Discover Diverse Perspectives]
    B --> C[Generate Expert Analysts]
    C --> D{User Feedback?}
    D -->|Has Feedback| C
    D -->|No Feedback| E[Simulate Expert Conversations]
    E --> F1[Perspective 1: Q&A with Expert]
    E --> F2[Perspective 2: Q&A with Expert]
    E --> F3[Perspective 3: Q&A with Expert]
    F1 --> G1[Ground Answers in Sources]
    F2 --> G2[Ground Answers in Sources]
    F3 --> G3[Ground Answers in Sources]
    G1 --> H[Curate Information]
    G2 --> H
    G3 --> H
    H --> I[Create Structured Outline]
    I --> J[Generate Article Sections]
    J --> K[Write Introduction]
    J --> L[Write Conclusion]
    K --> M[Final Wikipedia-style Article]
    L --> M
    M --> N[End]
```

### API Architecture

```mermaid
graph LR
    A[Next.js Frontend] --> B[FastAPI storm-api]
    B --> C[Vercel AI Gateway]
    C --> D[xai/grok-4.1]
    C --> E[openai/gpt-4o]
    C --> F[anthropic/claude]
    B --> G[PostgreSQL]
    B --> H[Redis]
    B --> I[Web Search APIs]
```

### Content Generation Pipeline

```mermaid
graph TD
    A[Create Brief] --> B[STORM Analysis]
    B --> C[7-Perspective Outline]
    C --> D[Research Query Generation]
    D --> E[Web Search & Local Data]
    E --> F[Parallel Section Writing]
    F --> G[SEO Optimization]
    G --> H[GEO Enhancement]
    H --> I[Final Content Assembly]
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- PostgreSQL database (Supabase or Vercel Postgres)
- Redis (Redis Labs or Vercel KV)
- Vercel AI Gateway API key

### 1. Clone the Repository

```bash
git clone https://github.com/madezmedia/STORM-Research-Assistant.git
cd STORM-Research-Assistant
```

### 2. Setup Storm API (Backend)

```bash
cd storm-api

# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with your credentials

# Run database migrations
uv run alembic upgrade head

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Environment Variables (storm-api/.env)

```env
# Database (Supabase/Vercel Postgres)
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db

# Redis (Redis Labs/Vercel KV)
REDIS_URL=redis://default:pass@host:port

# Vercel AI Gateway (required)
OPENAI_API_KEY=vck_your_vercel_ai_gateway_key
VERCEL_AI_GATEWAY_URL=https://ai-gateway.vercel.sh/v1
DEFAULT_MODEL=xai/grok-4.1-fast-reasoning

# Search APIs (optional)
GOOGLE_SEARCH_API_KEY=your_google_search_key
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id
BING_SEARCH_API_KEY=your_bing_key
```

### 4. Setup Frontend (Optional)

```bash
cd frontend

# Install dependencies
npm install

# Create environment file
cp .env.local.example .env.local

# Start development server
npm run dev
```

Access the frontend at `http://localhost:3000`

### 5. Verify Installation

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

## 📝 Usage

### API Usage

#### 1. Create a Content Brief

```bash
curl -X POST http://localhost:8000/api/v1/briefs \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Best Coffee Shops in Austin, Texas",
    "content_type": "local-guide",
    "seo": {
      "primary_keyword": "coffee shops austin",
      "secondary_keywords": ["austin coffee", "best coffee austin"],
      "difficulty": "medium",
      "intent": "informational"
    },
    "geo": {
      "enabled": true,
      "location": {"city": "Austin", "state": "Texas"},
      "local_keywords": ["south congress", "downtown austin"],
      "geo_intent": "local-service"
    },
    "target_audience": {
      "segment": "local-business",
      "expertise": "beginner"
    },
    "word_count": 2000,
    "tone": "friendly"
  }'
```

#### 2. Start Content Generation

```bash
curl -X POST http://localhost:8000/api/v1/briefs/{brief_id}/generate \
  -H "Content-Type: application/json" \
  -d '{"skip_analysis": false}'
```

#### 3. Check Generation Status

```bash
curl http://localhost:8000/api/v1/briefs/{brief_id}/status
```

#### 4. Retrieve Generated Content

```bash
curl http://localhost:8000/api/v1/content/{content_id}
```

### Supported Models (via Vercel AI Gateway)

| Provider | Models |
|----------|--------|
| **xAI** | `xai/grok-4.1-fast-reasoning` (default) |
| **OpenAI** | `openai/gpt-4o`, `openai/gpt-4o-mini` |
| **Anthropic** | `anthropic/claude-sonnet-4.5`, `anthropic/claude-4-opus` |
| **Google** | `google/gemini-pro` |

### STORM 7 Perspectives

The analysis engine generates outlines from these perspectives:
1. **Beginner** - Foundational concepts, terminology, prerequisites
2. **Business Owner** - ROI, costs, implementation timeline, risks
3. **Local Market** - Regulations, competitors, local success stories
4. **Technical** - How it works, requirements, integrations
5. **Competitive** - Alternatives, comparisons, differentiators
6. **Customer** - Pain points, decision factors, user experience
7. **Industry Expert** - Best practices, trends, future outlook

## 📚 Content Types

### Supported Content Types

| Type | Description | Best For |
|------|-------------|----------|
| `blog-post` | Standard blog article | General topics |
| `local-guide` | Location-specific guide | Local SEO, business directories |
| `comparison` | Product/service comparison | Commercial intent |
| `how-to` | Step-by-step tutorial | Informational content |
| `case-study` | In-depth analysis | B2B, professional services |

### Example Topics

**Local Business Content:**
- "Best Coffee Shops in Austin, Texas"
- "Top Digital Marketing Agencies in Miami"

**Technology Content:**
- "AI-Powered Content Generation Tools Comparison"
- "How to Implement RAG in Enterprise Applications"

**Business Content:**
- "ROI of SEO for Small Businesses in 2024"
- "Local Radio Advertising vs Digital Marketing"


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on Stanford's STORM paper: [Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models](https://arxiv.org/abs/2402.14207)
  - STORM achieves 25% better article organization and 10% broader topic coverage compared to baseline methods
  - The methodology addresses challenges in pre-writing stages including topic research and outline preparation
- Built with [LangGraph](https://langchain-ai.github.io/langgraph/) and [LangChain](https://python.langchain.com/)
- LangChain YouTube Channel: [Building STORM from scratch with LangGraph](https://youtu.be/1uUORSZwTz4?si=4RrM3UIuwwdWKFET)

## 🌐 Web Frontend

The project includes a Next.js frontend for creating content briefs and viewing generated content.

### Frontend Features

- **🎨 Modern UI**: Clean, responsive interface built with Tailwind CSS
- **📝 Content Brief Form**: Create and configure content briefs
- **📊 Generation Status**: Real-time progress tracking
- **📄 Content Viewer**: View and export generated content

### Frontend Tech Stack

- **Next.js 15**: React framework with App Router
- **Tailwind CSS**: Utility-first CSS framework
- **TypeScript**: Type-safe development
- **shadcn/ui**: UI component library

### Frontend Setup

See [`frontend/README.md`](frontend/README.md) for detailed setup instructions.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/madezmedia/STORM-Research-Assistant/issues)
- **Email**: tech@madezmedia.co
- **API Docs**: http://localhost:8000/docs (when running locally)