'use client';

import { useState } from 'react';
import { useChat } from '@ai-sdk/react';
import { LangSmithDeploymentTransport } from '@ai-sdk/langchain';
import { useMemo } from 'react';

export default function Home() {
  const [topic, setTopic] = useState('');
  const [maxAnalysts, setMaxAnalysts] = useState(3);
  const [maxInterviewTurns, setMaxInterviewTurns] = useState(3);
  const [model, setModel] = useState('openai/gpt-4.1');
  const [isInitialized, setIsInitialized] = useState(false);

  // Create transport for LangGraph backend connection
  const transport = useMemo(
    () =>
      new LangSmithDeploymentTransport({
        // Local development server:
        url: process.env.NEXT_PUBLIC_LANGGRAPH_URL || 'http://localhost:2024',
        // Optional: For LangSmith deployment
        // apiKey: process.env.NEXT_PUBLIC_LANGSMITH_API_KEY,
      }),
    [],
  );

  // Initialize chat with transport
  const { messages, sendMessage, status } = useChat({
    transport,
  });

  const handleInitialize = async () => {
    if (!topic.trim()) {
      alert('Please enter a research topic');
      return;
    }

    // Send initialization message with configuration
    await sendMessage({
      text: `Initialize research for topic: "${topic}" with max_analysts: ${maxAnalysts}, max_interview_turns: ${maxInterviewTurns}, model: ${model}`,
    });

    setIsInitialized(true);
  };

  const handleSendMessage = async (text: string) => {
    await sendMessage({ text });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Header */}
        <header className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-slate-900 dark:text-white mb-2">
            🌪️ STORM Research Assistant
          </h1>
          <p className="text-slate-600 dark:text-slate-400">
            Multi-perspective AI research system using LangGraph
          </p>
        </header>

        {/* Configuration Panel */}
        {!isInitialized && (
          <div className="bg-white dark:bg-slate-800 rounded-lg shadow-lg p-6 mb-6">
            <h2 className="text-xl font-semibold mb-4 text-slate-900 dark:text-white">
              Initialize Knowledge Context
            </h2>

            <div className="space-y-4">
              {/* Topic Input */}
              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                  Research Topic
                </label>
                <input
                  type="text"
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                  placeholder="e.g., The Future of Quantum Computing in Cryptography"
                  className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-slate-700 dark:text-white"
                />
              </div>

              {/* Configuration Options */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                    Max Analysts
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="10"
                    value={maxAnalysts}
                    onChange={(e) => setMaxAnalysts(parseInt(e.target.value))}
                    className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-slate-700 dark:text-white"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                    Max Interview Turns
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="10"
                    value={maxInterviewTurns}
                    onChange={(e) => setMaxInterviewTurns(parseInt(e.target.value))}
                    className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-slate-700 dark:text-white"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                    Model
                  </label>
                  <select
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-slate-700 dark:text-white"
                  >
                    <option value="openai/gpt-4.1">OpenAI GPT-4.1</option>
                    <option value="openai/gpt-4.1-mini">OpenAI GPT-4.1 Mini</option>
                    <option value="anthropic/claude-opus-4-20250514">Anthropic Claude Opus</option>
                    <option value="anthropic/claude-3-7-sonnet-latest">Anthropic Claude Sonnet</option>
                    <option value="azure/gpt-4.1">Azure GPT-4.1</option>
                  </select>
                </div>
              </div>

              {/* Initialize Button */}
              <button
                onClick={handleInitialize}
                disabled={status === 'streaming' || !topic.trim()}
                className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-slate-400 text-white font-semibold py-3 px-6 rounded-lg transition-colors"
              >
                {status === 'streaming' ? 'Initializing...' : 'Initialize Research'}
              </button>
            </div>
          </div>
        )}

        {/* Chat Interface */}
        <div className="bg-white dark:bg-slate-800 rounded-lg shadow-lg p-6">
          <div className="mb-4 flex justify-between items-center">
            <h2 className="text-xl font-semibold text-slate-900 dark:text-white">
              Research Progress
            </h2>
            {isInitialized && (
              <span className="text-sm text-slate-600 dark:text-slate-400">
                {status === 'streaming' ? 'Processing...' : 'Ready'}
              </span>
            )}
          </div>

          {/* Messages */}
          <div className="space-y-4 mb-4 max-h-96 overflow-y-auto">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`p-4 rounded-lg ${
                  message.role === 'user'
                    ? 'bg-blue-50 dark:bg-blue-900/20 ml-8'
                    : 'bg-slate-50 dark:bg-slate-700/50 mr-8'
                }`}
              >
                <div className="font-semibold text-sm mb-2 text-slate-700 dark:text-slate-300">
                  {message.role === 'user' ? '👤 You' : '🤖 STORM Assistant'}
                </div>
                {message.parts.map((part, i) => (
                  <div key={i}>
                    {part.type === 'text' && (
                      <div className="text-slate-900 dark:text-white whitespace-pre-wrap">
                        {part.text}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ))}
          </div>

          {/* Input */}
          {isInitialized && (
            <form
              onSubmit={(e) => {
                e.preventDefault();
                const input = e.currentTarget.elements.namedItem(
                  'message',
                ) as HTMLInputElement;
                if (input.value.trim()) {
                  handleSendMessage(input.value);
                  input.value = '';
                }
              }}
              className="flex gap-2"
            >
              <input
                name="message"
                placeholder="Type a message or feedback..."
                disabled={status === 'streaming'}
                className="flex-1 px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-slate-700 dark:text-white"
              />
              <button
                type="submit"
                disabled={status === 'streaming'}
                className="bg-blue-600 hover:bg-blue-700 disabled:bg-slate-400 text-white font-semibold py-2 px-6 rounded-lg transition-colors"
              >
                Send
              </button>
            </form>
          )}
        </div>

        {/* Footer */}
        <footer className="mt-8 text-center text-sm text-slate-600 dark:text-slate-400">
          <p>Powered by LangGraph & Vercel AI SDK</p>
        </footer>
      </div>
    </div>
  );
}
