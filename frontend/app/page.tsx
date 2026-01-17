'use client';

import { useState } from 'react';

interface GenerationStatus {
  brief_id: string;
  status: string;
  progress: number;
  current_phase: string;
  estimated_time_remaining: number;
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || '/api/v1';

export default function Home() {
  const [topic, setTopic] = useState('');
  const [contentType, setContentType] = useState('blog-post');
  const [wordCount, setWordCount] = useState(2000);
  const [model, setModel] = useState('xai/grok-4.1-fast-reasoning');
  const [isGenerating, setIsGenerating] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [generationStatus, setGenerationStatus] = useState<GenerationStatus | null>(null);
  const [briefId, setBriefId] = useState<string | null>(null);

  const addMessage = (role: 'user' | 'assistant', content: string) => {
    setMessages(prev => [...prev, {
      id: Date.now().toString(),
      role,
      content,
      timestamp: new Date()
    }]);
  };

  const handleGenerate = async () => {
    if (!topic.trim()) {
      alert('Please enter a research topic');
      return;
    }

    setIsGenerating(true);
    addMessage('user', `Generate content for: "${topic}"`);

    try {
      // Step 1: Create content brief
      addMessage('assistant', 'Creating content brief...');

      const briefResponse = await fetch(`${API_BASE}/briefs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic,
          content_type: contentType,
          seo: {
            primary_keyword: topic.split(' ').slice(0, 3).join(' '),
            secondary_keywords: [],
            difficulty: 'medium',
            intent: 'informational'
          },
          brand_direction: 'neutral',
          target_audience: {
            segment: 'general',
            pain_points: [],
            expertise: 'intermediate'
          },
          word_count: wordCount,
          tone: 'professional',
          include_examples: true,
          include_stats: true,
          include_local_data: false
        })
      });

      if (!briefResponse.ok) {
        throw new Error('Failed to create content brief');
      }

      const brief = await briefResponse.json();
      setBriefId(brief.id);
      addMessage('assistant', `Content brief created (ID: ${brief.id})`);

      // Step 2: Start generation
      addMessage('assistant', 'Starting STORM analysis and content generation...');

      const genResponse = await fetch(`${API_BASE}/briefs/${brief.id}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ skip_analysis: false })
      });

      if (!genResponse.ok) {
        throw new Error('Failed to start generation');
      }

      // Step 3: Poll for status
      let status: GenerationStatus;
      do {
        await new Promise(resolve => setTimeout(resolve, 3000));

        const statusResponse = await fetch(`${API_BASE}/briefs/${brief.id}/status`);
        status = await statusResponse.json();
        setGenerationStatus(status);

        addMessage('assistant', `${status.current_phase}: ${status.progress}% complete`);
      } while (status.status !== 'complete' && status.status !== 'failed');

      if (status.status === 'complete') {
        addMessage('assistant', 'Content generation complete! Fetching results...');

        // Get the generated content
        const contentResponse = await fetch(`${API_BASE}/content/${brief.id}`);
        if (contentResponse.ok) {
          const content = await contentResponse.json();
          addMessage('assistant', `**${content.title}**\n\n${content.content.substring(0, 500)}...\n\n[Full content: ${content.word_count} words]`);
        }
      } else {
        addMessage('assistant', 'Generation failed. Please try again.');
      }

    } catch (error) {
      addMessage('assistant', `Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleTestSTORM = async () => {
    addMessage('user', `Testing STORM analysis for: "${topic}"`);

    try {
      const response = await fetch(`${API_BASE}/test-storm`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic, content_type: contentType })
      });

      const result = await response.json();

      if (result.success && result.outline) {
        const outline = result.outline;
        let outlineText = `**STORM Analysis Complete**\n\n`;
        outlineText += `**Topic:** ${outline.topic}\n`;
        outlineText += `**Type:** ${outline.content_type}\n\n`;
        outlineText += `**Sections:**\n`;
        outline.sections?.forEach((section: any, i: number) => {
          outlineText += `${i + 1}. ${section.title}\n`;
          section.subsections?.forEach((sub: any) => {
            outlineText += `   - ${sub.title}\n`;
          });
        });
        addMessage('assistant', outlineText);
      } else {
        addMessage('assistant', `STORM test failed: ${result.error || 'Unknown error'}`);
      }
    } catch (error) {
      addMessage('assistant', `Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Header */}
        <header className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-slate-900 dark:text-white mb-2">
            STORM Research Assistant
          </h1>
          <p className="text-slate-600 dark:text-slate-400">
            AI-powered content generation with multi-perspective STORM analysis
          </p>
        </header>

        {/* Configuration Panel */}
        <div className="bg-white dark:bg-slate-800 rounded-lg shadow-lg p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4 text-slate-900 dark:text-white">
            Content Configuration
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
                placeholder="e.g., Best Coffee Shops in Austin, Texas"
                className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-slate-700 dark:text-white"
              />
            </div>

            {/* Configuration Options */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                  Content Type
                </label>
                <select
                  value={contentType}
                  onChange={(e) => setContentType(e.target.value)}
                  className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-slate-700 dark:text-white"
                >
                  <option value="blog-post">Blog Post</option>
                  <option value="local-guide">Local Guide</option>
                  <option value="comparison">Comparison</option>
                  <option value="how-to">How-To Guide</option>
                  <option value="case-study">Case Study</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                  Word Count
                </label>
                <input
                  type="number"
                  min="1000"
                  max="10000"
                  step="500"
                  value={wordCount}
                  onChange={(e) => setWordCount(parseInt(e.target.value))}
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
                  <option value="xai/grok-4.1-fast-reasoning">xAI Grok 4.1</option>
                  <option value="openai/gpt-4o">OpenAI GPT-4o</option>
                  <option value="anthropic/claude-sonnet-4.5">Claude Sonnet 4.5</option>
                </select>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-4">
              <button
                onClick={handleTestSTORM}
                disabled={isGenerating || !topic.trim()}
                className="flex-1 bg-slate-600 hover:bg-slate-700 disabled:bg-slate-400 text-white font-semibold py-3 px-6 rounded-lg transition-colors"
              >
                Test STORM Analysis
              </button>
              <button
                onClick={handleGenerate}
                disabled={isGenerating || !topic.trim()}
                className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-400 text-white font-semibold py-3 px-6 rounded-lg transition-colors"
              >
                {isGenerating ? 'Generating...' : 'Generate Content'}
              </button>
            </div>
          </div>
        </div>

        {/* Progress Status */}
        {generationStatus && (
          <div className="bg-white dark:bg-slate-800 rounded-lg shadow-lg p-6 mb-6">
            <h3 className="text-lg font-semibold mb-2 text-slate-900 dark:text-white">
              Generation Progress
            </h3>
            <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-4 mb-2">
              <div
                className="bg-blue-600 h-4 rounded-full transition-all duration-500"
                style={{ width: `${generationStatus.progress}%` }}
              />
            </div>
            <p className="text-sm text-slate-600 dark:text-slate-400">
              {generationStatus.current_phase} - {generationStatus.progress}%
            </p>
          </div>
        )}

        {/* Messages */}
        <div className="bg-white dark:bg-slate-800 rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4 text-slate-900 dark:text-white">
            Activity Log
          </h2>

          <div className="space-y-4 max-h-96 overflow-y-auto">
            {messages.length === 0 ? (
              <p className="text-slate-500 dark:text-slate-400 text-center py-8">
                Enter a topic and click &quot;Test STORM Analysis&quot; or &quot;Generate Content&quot; to begin
              </p>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`p-4 rounded-lg ${
                    message.role === 'user'
                      ? 'bg-blue-50 dark:bg-blue-900/20 ml-8'
                      : 'bg-slate-50 dark:bg-slate-700/50 mr-8'
                  }`}
                >
                  <div className="font-semibold text-sm mb-2 text-slate-700 dark:text-slate-300">
                    {message.role === 'user' ? 'You' : 'STORM Assistant'}
                  </div>
                  <div className="text-slate-900 dark:text-white whitespace-pre-wrap">
                    {message.content}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-8 text-center text-sm text-slate-600 dark:text-slate-400">
          <p>Powered by STORM API & Vercel AI Gateway</p>
        </footer>
      </div>
    </div>
  );
}
