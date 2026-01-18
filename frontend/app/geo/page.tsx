'use client';

import { useState } from 'react';
import { PageHeader } from '@/components/layout';
import { geoApi } from '@/lib/api';
import { GeoAnalysis, GeneratedPrompt } from '@/types/api';
import {
  Globe,
  Search,
  Loader2,
  Copy,
  CheckCircle2,
  Sparkles,
  Target,
  Tags,
  MessageSquare
} from 'lucide-react';
import { cn } from '@/lib/utils';

export default function GeoPage() {
  const [url, setUrl] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState<GeoAnalysis | null>(null);
  const [prompts, setPrompts] = useState<GeneratedPrompt[]>([]);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    if (!url.trim()) return;

    setIsAnalyzing(true);
    setError(null);
    setAnalysis(null);
    setPrompts([]);

    try {
      // Extract domain from URL for brand name
      const urlObj = new URL(url.startsWith('http') ? url : `https://${url}`);
      const brandName = urlObj.hostname.replace('www.', '').split('.')[0];

      const result = await geoApi.analyzeWebsite({
        url: url.startsWith('http') ? url : `https://${url}`,
        brand_name: brandName,
        crawl_limit: 10,
      });
      setAnalysis(result);

      // Auto-generate prompts after analysis
      if (result.keywords && result.keywords.length > 0) {
        // Extract just the keyword strings from the [string, number][] tuples
        const keywordStrings = result.keywords.slice(0, 5).map(k => Array.isArray(k) ? k[0] : k);
        const promptResult = await geoApi.generatePrompts({
          keywords: keywordStrings,
          brand_name: result.brand_name || brandName,
        });
        setPrompts(promptResult.prompts || []);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleCopy = async (text: string, id: string) => {
    await navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div className="animate-fade-in">
      <PageHeader
        title="GEO Intelligence"
        description="Analyze websites for AI search optimization"
      />

      {/* URL Input */}
      <div className="glass rounded-xl p-6 mb-8">
        <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
          <Globe className="h-5 w-5 text-primary" />
          Website Analysis
        </h2>

        <div className="flex gap-4">
          <input
            type="url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://example.com"
            className="flex-1 px-4 py-3 rounded-lg bg-secondary/50 border border-border text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary"
          />
          <button
            onClick={handleAnalyze}
            disabled={isAnalyzing || !url.trim()}
            className={cn(
              'flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all',
              isAnalyzing || !url.trim()
                ? 'bg-secondary text-muted-foreground cursor-not-allowed'
                : 'bg-primary text-primary-foreground hover:bg-primary/90 glow-sm'
            )}
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="h-5 w-5 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Search className="h-5 w-5" />
                Analyze
              </>
            )}
          </button>
        </div>

        {error && (
          <div className="mt-4 p-4 rounded-lg bg-destructive/10 text-destructive text-sm">
            {error}
          </div>
        )}
      </div>

      {/* Analysis Results */}
      {analysis && (
        <div className="grid gap-6 lg:grid-cols-2">
          {/* Keywords */}
          <div className="glass rounded-xl p-6">
            <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
              <Tags className="h-5 w-5 text-accent" />
              Keywords & Topics
            </h3>

            {analysis.keywords && analysis.keywords.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {analysis.keywords.map((keyword, i) => {
                  // Handle both [string, number] tuples and plain strings
                  const keywordText = Array.isArray(keyword) ? keyword[0] : keyword;
                  const score = Array.isArray(keyword) ? keyword[1] : null;
                  return (
                    <span
                      key={i}
                      className="px-3 py-1.5 rounded-full bg-accent/20 text-accent text-sm font-medium"
                      title={score ? `Score: ${score}` : undefined}
                    >
                      {keywordText}
                    </span>
                  );
                })}
              </div>
            ) : (
              <p className="text-muted-foreground">No keywords extracted</p>
            )}
          </div>

          {/* Analysis Details */}
          <div className="glass rounded-xl p-6">
            <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
              <Target className="h-5 w-5 text-success" />
              Analysis Details
            </h3>

            <dl className="space-y-3">
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Domain</dt>
                <dd className="text-foreground truncate max-w-[200px]">{analysis.domain}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Status</dt>
                <dd className="flex items-center gap-1 text-success">
                  <CheckCircle2 className="h-4 w-4" />
                  Complete
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Keywords Found</dt>
                <dd className="text-foreground">{analysis.keywords?.length || 0}</dd>
              </div>
            </dl>
          </div>

          {/* Generated Prompts */}
          <div className="lg:col-span-2 glass rounded-xl p-6">
            <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
              <MessageSquare className="h-5 w-5 text-primary" />
              AI Search Prompts
            </h3>

            {prompts.length > 0 ? (
              <div className="space-y-4">
                {prompts.map((prompt, i) => (
                  <div
                    key={i}
                    className="group p-4 rounded-lg bg-secondary/30 hover:bg-secondary/50 transition-colors"
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1">
                        <p className="text-foreground">{prompt.prompt_text}</p>
                        {prompt.category && (
                          <span className="mt-2 inline-block px-2 py-0.5 rounded text-xs bg-primary/20 text-primary">
                            {prompt.category}
                          </span>
                        )}
                      </div>
                      <button
                        onClick={() => handleCopy(prompt.prompt_text, `prompt-${i}`)}
                        className="p-2 rounded-lg bg-secondary/50 hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors opacity-0 group-hover:opacity-100"
                      >
                        {copiedId === `prompt-${i}` ? (
                          <CheckCircle2 className="h-4 w-4 text-success" />
                        ) : (
                          <Copy className="h-4 w-4" />
                        )}
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <Sparkles className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Prompts will be generated after website analysis</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Empty State */}
      {!analysis && !isAnalyzing && (
        <div className="glass rounded-xl p-12 text-center">
          <Globe className="h-16 w-16 text-muted-foreground/50 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-foreground mb-2">
            Analyze a Website
          </h3>
          <p className="text-muted-foreground max-w-md mx-auto">
            Enter a URL above to analyze the website for keywords, topics, and generate
            AI search prompts to improve visibility in AI-powered search engines.
          </p>
        </div>
      )}
    </div>
  );
}
