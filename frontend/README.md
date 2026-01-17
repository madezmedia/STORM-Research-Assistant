# STORM Research Assistant - Frontend

Next.js frontend for the STORM Research Assistant, using Vercel AI SDK to connect to the Python LangGraph backend.

## 🚀 Getting Started

### Prerequisites

- Node.js 18+ and npm/yarn/pnpm
- Python LangGraph backend running on `http://localhost:2024` (or a deployed LangSmith instance)

### Installation

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Create environment file:
```bash
cp .env.local.example .env.local
```

3. Configure `.env.local`:
```env
# Local development server:
NEXT_PUBLIC_LANGGRAPH_URL=http://localhost:2024

# Or for a LangSmith deployment:
# NEXT_PUBLIC_LANGGRAPH_URL=https://your-deployment.langsmith.app
# NEXT_PUBLIC_LANGSMITH_API_KEY=your_langsmith_api_key
```

4. Start the development server:
```bash
npm run dev
```

5. Open [http://localhost:3000](http://localhost:3000) in your browser

## 🏗️ Architecture

### Components

- **Layout**: Root layout with global styles
- **Page**: Main research interface with:
  - Configuration panel for initializing research
  - Chat interface for interacting with the research process
  - Real-time streaming responses

### Integration with LangGraph

The frontend uses `LangSmithDeploymentTransport` from `@ai-sdk/langchain` to connect to the LangGraph backend:

```typescript
import { LangSmithDeploymentTransport } from '@ai-sdk/langchain';
import { useChat } from '@ai-sdk/react';

const transport = new LangSmithDeploymentTransport({
  url: 'http://localhost:2024',
});

const { messages, sendMessage, status } = useChat({ transport });
```

## 📝 Usage

### Initialize Research

1. Enter a research topic (e.g., "The Future of Quantum Computing in Cryptography")
2. Configure options:
   - Max Analysts: Number of expert perspectives (1-10)
   - Max Interview Turns: Conversation depth per analyst (1-10)
   - Model: Choose LLM provider (OpenAI, Anthropic, Azure)
3. Click "Initialize Research"

### Interact with Research Process

- View real-time progress as the system:
  - Generates expert analysts
  - Conducts interviews
  - Writes report sections
  - Generates final article

- Provide feedback to refine perspectives (optional)

## 🎨 Tech Stack

- **Next.js 15** - React framework
- **Vercel AI SDK** - AI integration
- **LangSmithDeploymentTransport** - LangGraph connection
- **Tailwind CSS** - Styling
- **TypeScript** - Type safety

## 🔧 Configuration

### Model Providers

The frontend supports the following model providers:

| Provider | Models |
|----------|---------|
| OpenAI | gpt-4.1, gpt-4.1-mini |
| Anthropic | claude-opus-4-20250514, claude-3-7-sonnet-latest |
| Azure OpenAI | gpt-4.1, gpt-4.1-mini |

Note: Actual model availability depends on the backend configuration and API keys.

## 📚 Documentation

- [Vercel AI SDK Docs](https://sdk.vercel.ai/docs)
- [LangSmithDeploymentTransport](https://sdk.vercel.ai/docs/ai-sdk-core/providers-and-models/langsmith-deployment-transport)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)

## 🤝 Contributing

This frontend is part of the STORM Research Assistant project. See the main [README](../README.md) for contribution guidelines.
