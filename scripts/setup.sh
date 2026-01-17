#!/bin/bash

# STORM Research Assistant Setup Script
# This script helps set up both the Python backend and Next.js frontend

set -e  # Exit on error

echo "🌪️  STORM Research Assistant - Setup Script"
echo "=========================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed."
    echo "Please install uv from: https://github.com/astral-sh/uv"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ Error: npm is not installed."
    echo "Please install Node.js and npm from: https://nodejs.org/"
    exit 1
fi

echo "✅ Prerequisites check passed!"
echo ""

# Setup Python Backend
echo "📦 Setting up Python Backend..."
echo "-----------------------------------"

# Create virtual environment
echo "Creating virtual environment..."
uv venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
uv pip install -e .
uv pip install -e ".[dev]"

echo "✅ Python backend setup complete!"
echo ""

# Setup Frontend
echo "🌐 Setting up Next.js Frontend..."
echo "-----------------------------------"

cd frontend

# Install Node dependencies
echo "Installing Node dependencies..."
npm install

# Create environment file
if [ ! -f .env.local ]; then
    echo "Creating .env.local from template..."
    cp .env.local.example .env.local
    echo "✅ .env.local created. Please edit it with your configuration."
else
    echo "⚠️  .env.local already exists. Skipping creation."
fi

cd ..

echo "✅ Frontend setup complete!"
echo ""

# Create .env file for backend
if [ ! -f .env ]; then
    echo "📝 Creating .env file for backend..."
    cp .env.example .env
    echo "✅ .env created. Please edit it with your API keys."
    echo ""
    echo "Required API keys:"
    echo "  - TAVILY_API_KEY (required for web search)"
    echo "  - OPENAI_API_KEY or ANTHROPIC_API_KEY or AZURE_OPENAI_API_KEY"
    echo "  - LANGSMITH_API_KEY (optional, for tracing)"
else
    echo "⚠️  .env already exists. Skipping creation."
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo ""
echo "1. Configure your API keys in .env file:"
echo "   nano .env"
echo ""
echo "2. Start the Python backend:"
echo "   source .venv/bin/activate"
echo "   uv run langgraph dev"
echo ""
echo "3. In a new terminal, start the frontend:"
echo "   cd frontend"
echo "   npm run dev"
echo ""
echo "4. Access the application:"
echo "   - Frontend: http://localhost:3000"
echo "   - LangGraph Studio: http://localhost:2024"
echo ""
