'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { PageHeader } from '@/components/layout';
import { useBriefStore, useGenerationStore } from '@/stores';
import {
  ArrowLeft,
  Loader2,
  CheckCircle2,
  XCircle,
  Download,
  Copy,
  FileText,
  Clock,
  Globe,
  BarChart3
} from 'lucide-react';
import { cn } from '@/lib/utils';
import ReactMarkdown from 'react-markdown';

export default function StudioDetailPage() {
  const params = useParams();
  const briefId = params.id as string;

  const { currentBrief, fetchBrief, isLoading: briefLoading } = useBriefStore();
  const { status, progress, currentPhase, content, startPolling, stopPolling } = useGenerationStore();
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (briefId) {
      fetchBrief(briefId);
      startPolling(briefId);
    }

    return () => {
      stopPolling();
    };
  }, [briefId, fetchBrief, startPolling, stopPolling]);

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

  if (briefLoading && !currentBrief) {
    return (
      <div className="animate-fade-in flex items-center justify-center min-h-[400px]">
        <div className="flex flex-col items-center text-muted-foreground">
          <Loader2 className="h-12 w-12 animate-spin mb-4" />
          <p>Loading brief...</p>
        </div>
      </div>
    );
  }

  const isGenerating = status === 'generating' || status === 'processing';
  const isComplete = status === 'complete' || status === 'completed';
  const isFailed = status === 'failed';

  return (
    <div className="animate-fade-in">
      <PageHeader
        title={currentBrief?.topic || 'Content Brief'}
        description={`${currentBrief?.content_type?.replace('-', ' ')} • ${currentBrief?.word_count?.toLocaleString()} words`}
        actions={
          <Link
            href="/briefs"
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-secondary text-foreground font-medium hover:bg-secondary/80 transition-colors"
          >
            <ArrowLeft className="h-4 w-4" />
            All Briefs
          </Link>
        }
      />

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Main Content Area */}
        <div className="lg:col-span-2 space-y-6">
          {/* Generation Progress */}
          {isGenerating && (
            <div className="glass rounded-xl p-6 animate-pulse-glow">
              <div className="flex items-center gap-4 mb-4">
                <Loader2 className="h-6 w-6 text-primary animate-spin" />
                <div>
                  <h3 className="font-semibold text-foreground">Generating Content</h3>
                  <p className="text-sm text-muted-foreground">{currentPhase}</p>
                </div>
              </div>

              <div className="relative">
                <div className="w-full h-3 bg-secondary rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-primary to-accent transition-all duration-500"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <span className="absolute right-0 top-4 text-sm text-muted-foreground">
                  {progress}%
                </span>
              </div>
            </div>
          )}

          {/* Status Banner */}
          {isComplete && (
            <div className="glass rounded-xl p-6 bg-success/10 border-success/30">
              <div className="flex items-center gap-3">
                <CheckCircle2 className="h-6 w-6 text-success" />
                <div>
                  <h3 className="font-semibold text-success">Generation Complete</h3>
                  <p className="text-sm text-success/80">Your content is ready to view and export.</p>
                </div>
              </div>
            </div>
          )}

          {isFailed && (
            <div className="glass rounded-xl p-6 bg-destructive/10 border-destructive/30">
              <div className="flex items-center gap-3">
                <XCircle className="h-6 w-6 text-destructive" />
                <div>
                  <h3 className="font-semibold text-destructive">Generation Failed</h3>
                  <p className="text-sm text-destructive/80">Something went wrong. Please try again.</p>
                </div>
              </div>
            </div>
          )}

          {/* Content Preview */}
          {content?.content && (
            <div className="glass rounded-xl p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-bold text-foreground">{content.title}</h2>
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

              <div className="prose prose-invert max-w-none">
                <ReactMarkdown>{content.content}</ReactMarkdown>
              </div>
            </div>
          )}

          {/* Empty State */}
          {!isGenerating && !content?.content && !isFailed && (
            <div className="glass rounded-xl p-12 text-center">
              <FileText className="h-16 w-16 text-muted-foreground/50 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-foreground mb-2">Content Not Generated</h3>
              <p className="text-muted-foreground mb-4">This brief hasn't been generated yet.</p>
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Brief Details */}
          <div className="glass rounded-xl p-6">
            <h3 className="font-semibold text-foreground mb-4">Brief Details</h3>
            <dl className="space-y-4">
              <div>
                <dt className="text-sm text-muted-foreground">Type</dt>
                <dd className="text-foreground capitalize">{currentBrief?.content_type?.replace('-', ' ')}</dd>
              </div>
              <div>
                <dt className="text-sm text-muted-foreground">Word Count</dt>
                <dd className="text-foreground">{currentBrief?.word_count?.toLocaleString()}</dd>
              </div>
              <div>
                <dt className="text-sm text-muted-foreground">Tone</dt>
                <dd className="text-foreground capitalize">{currentBrief?.tone || 'Professional'}</dd>
              </div>
              <div>
                <dt className="text-sm text-muted-foreground">Status</dt>
                <dd className={cn(
                  'font-medium',
                  isComplete && 'text-success',
                  isGenerating && 'text-primary',
                  isFailed && 'text-destructive'
                )}>
                  {status || currentBrief?.status || 'Pending'}
                </dd>
              </div>
            </dl>
          </div>

          {/* SEO Info */}
          {currentBrief?.seo && (
            <div className="glass rounded-xl p-6">
              <h3 className="font-semibold text-foreground mb-4 flex items-center gap-2">
                <BarChart3 className="h-4 w-4" />
                SEO Keywords
              </h3>
              <div className="space-y-3">
                <div>
                  <span className="text-sm text-muted-foreground">Primary:</span>
                  <span className="ml-2 px-2 py-1 rounded bg-primary/20 text-primary text-sm">
                    {currentBrief.seo.primary_keyword}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Content Stats */}
          {content && (
            <div className="glass rounded-xl p-6">
              <h3 className="font-semibold text-foreground mb-4">Content Stats</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-3 rounded-lg bg-secondary/50 text-center">
                  <div className="text-2xl font-bold text-foreground">
                    {content.word_count?.toLocaleString() || '—'}
                  </div>
                  <div className="text-xs text-muted-foreground">Words</div>
                </div>
                <div className="p-3 rounded-lg bg-secondary/50 text-center">
                  <div className="text-2xl font-bold text-foreground">
                    {content.sources?.length || 0}
                  </div>
                  <div className="text-xs text-muted-foreground">Sources</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
