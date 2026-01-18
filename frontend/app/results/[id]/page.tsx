'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { PageHeader } from '@/components/layout';
import { useBriefStore } from '@/stores';
import { contentApi } from '@/lib/api';
import {
  ArrowLeft,
  Loader2,
  Copy,
  Download,
  BarChart3,
  Link as LinkIcon,
  CheckCircle2
} from 'lucide-react';
import { cn } from '@/lib/utils';
import ReactMarkdown from 'react-markdown';

interface SeoAnalysis {
  content_id?: string;
  score?: number;
  seo_score?: Record<string, unknown>;
  recommendations?: string[] | unknown[];
  keyword_density?: Record<string, number>;
}

export default function ResultDetailPage() {
  const params = useParams();
  const briefId = params.id as string;

  const { currentBrief, fetchBrief, isLoading } = useBriefStore();
  const [content, setContent] = useState<any>(null);
  const [seoAnalysis, setSeoAnalysis] = useState<SeoAnalysis | null>(null);
  const [copied, setCopied] = useState(false);
  const [loadingContent, setLoadingContent] = useState(true);

  useEffect(() => {
    if (briefId) {
      fetchBrief(briefId);
      loadContent();
    }
  }, [briefId, fetchBrief]);

  const loadContent = async () => {
    setLoadingContent(true);
    try {
      const [contentResult, seoResult] = await Promise.all([
        contentApi.getContent(briefId),
        contentApi.getSeoAnalysis(briefId).catch(() => null),
      ]);
      setContent(contentResult);
      setSeoAnalysis(seoResult);
    } catch (error) {
      console.error('Failed to load content:', error);
    } finally {
      setLoadingContent(false);
    }
  };

  const handleCopy = async () => {
    if (content?.content) {
      await navigator.clipboard.writeText(content.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleDownload = () => {
    if (content?.content) {
      const blob = new Blob([content.content], { type: 'text/markdown' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${currentBrief?.topic || 'content'}.md`;
      a.click();
      URL.revokeObjectURL(url);
    }
  };

  if (isLoading || loadingContent) {
    return (
      <div className="animate-fade-in flex items-center justify-center min-h-[400px]">
        <div className="flex flex-col items-center text-muted-foreground">
          <Loader2 className="h-12 w-12 animate-spin mb-4" />
          <p>Loading content...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="animate-fade-in">
      <PageHeader
        title={content?.title || currentBrief?.topic || 'Result'}
        description={`${currentBrief?.content_type?.replace('-', ' ')} • ${content?.word_count?.toLocaleString() || currentBrief?.word_count?.toLocaleString()} words`}
        actions={
          <Link
            href="/results"
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-secondary text-foreground font-medium hover:bg-secondary/80 transition-colors"
          >
            <ArrowLeft className="h-4 w-4" />
            All Results
          </Link>
        }
      />

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Main Content */}
        <div className="lg:col-span-2 space-y-6">
          {/* Content Display */}
          <div className="glass rounded-xl p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-bold text-foreground">Content</h2>
              <div className="flex items-center gap-2">
                <button
                  onClick={handleCopy}
                  className="flex items-center gap-2 px-3 py-2 rounded-lg bg-secondary hover:bg-secondary/80 text-foreground text-sm transition-colors"
                >
                  <Copy className="h-4 w-4" />
                  {copied ? 'Copied!' : 'Copy'}
                </button>
                <button
                  onClick={handleDownload}
                  className="flex items-center gap-2 px-3 py-2 rounded-lg bg-primary hover:bg-primary/90 text-primary-foreground text-sm transition-colors"
                >
                  <Download className="h-4 w-4" />
                  Download
                </button>
              </div>
            </div>

            {content?.content ? (
              <div className="prose prose-invert max-w-none">
                <ReactMarkdown>{content.content}</ReactMarkdown>
              </div>
            ) : (
              <p className="text-muted-foreground">No content available</p>
            )}
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* SEO Score */}
          {seoAnalysis && (
            <div className="glass rounded-xl p-6">
              <h3 className="font-semibold text-foreground mb-4 flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-primary" />
                SEO Analysis
              </h3>

              {seoAnalysis.score !== undefined && (
                <div className="relative mb-6">
                  <svg className="w-32 h-32 mx-auto" viewBox="0 0 100 100">
                    <circle
                      cx="50"
                      cy="50"
                      r="45"
                      fill="none"
                      stroke="hsl(var(--secondary))"
                      strokeWidth="8"
                    />
                    <circle
                      cx="50"
                      cy="50"
                      r="45"
                      fill="none"
                      stroke="hsl(var(--primary))"
                      strokeWidth="8"
                      strokeLinecap="round"
                      strokeDasharray={`${(seoAnalysis.score / 100) * 283} 283`}
                      transform="rotate(-90 50 50)"
                    />
                  </svg>
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <span className="text-3xl font-bold text-foreground">{seoAnalysis.score}</span>
                    <span className="text-sm text-muted-foreground">SEO Score</span>
                  </div>
                </div>
              )}

              {seoAnalysis.recommendations && seoAnalysis.recommendations.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-foreground mb-2">Recommendations</h4>
                  <ul className="space-y-2">
                    {seoAnalysis.recommendations.slice(0, 5).map((rec, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm text-muted-foreground">
                        <CheckCircle2 className="h-4 w-4 text-success mt-0.5 shrink-0" />
                        {String(rec)}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* Sources */}
          {content?.sources && content.sources.length > 0 && (
            <div className="glass rounded-xl p-6">
              <h3 className="font-semibold text-foreground mb-4 flex items-center gap-2">
                <LinkIcon className="h-5 w-5 text-accent" />
                Sources ({content.sources.length})
              </h3>

              <ul className="space-y-3">
                {content.sources.slice(0, 10).map((source: any, i: number) => (
                  <li key={i}>
                    <a
                      href={source.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="block p-3 rounded-lg bg-secondary/30 hover:bg-secondary/50 transition-colors"
                    >
                      <span className="font-medium text-foreground text-sm line-clamp-1">
                        {source.title || source.url}
                      </span>
                      <span className="text-xs text-muted-foreground line-clamp-1">
                        {source.url}
                      </span>
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Content Stats */}
          <div className="glass rounded-xl p-6">
            <h3 className="font-semibold text-foreground mb-4">Stats</h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Words</span>
                <span className="text-foreground font-medium">
                  {content?.word_count?.toLocaleString() || '—'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Sources</span>
                <span className="text-foreground font-medium">
                  {content?.sources?.length || 0}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Type</span>
                <span className="text-foreground font-medium capitalize">
                  {currentBrief?.content_type?.replace('-', ' ') || 'Article'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
