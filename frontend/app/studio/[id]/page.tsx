'use client';

import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { PageHeader } from '@/components/layout';
import { useBriefStore, useGenerationStore } from '@/stores';
import { slidesApi } from '@/lib/api';
import type { BriefSlidesResponse } from '@/lib/api/slides';
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
  BarChart3,
  Presentation,
  X,
  ExternalLink
} from 'lucide-react';
import { cn } from '@/lib/utils';
import ReactMarkdown from 'react-markdown';

export default function StudioDetailPage() {
  const params = useParams();
  const router = useRouter();
  const briefId = params.id as string;

  const { currentBrief, currentContent, fetchBrief, fetchContent, isLoading: briefLoading } = useBriefStore();
  const {
    status,
    progress,
    currentPhase,
    startStreaming,
    startPolling,
    stopTracking,
    useSSE
  } = useGenerationStore();

  const [copied, setCopied] = useState(false);
  const [isGeneratingSlides, setIsGeneratingSlides] = useState(false);
  const [slideshowData, setSlideshowData] = useState<BriefSlidesResponse | null>(null);
  const [showSlideshowModal, setShowSlideshowModal] = useState(false);
  const [slidesError, setSlidesError] = useState<string | null>(null);

  // Use content from store or generation
  const content = currentContent;

  useEffect(() => {
    if (briefId) {
      fetchBrief(briefId);
      fetchContent(briefId);

      // Use SSE streaming by default, with polling fallback
      if (useSSE) {
        startStreaming(briefId, () => {
          // Refresh content when generation completes
          fetchContent(briefId);
        });
      } else {
        startPolling(briefId, () => {
          fetchContent(briefId);
        });
      }
    }

    return () => {
      stopTracking();
    };
  }, [briefId, fetchBrief, fetchContent, startStreaming, startPolling, stopTracking, useSSE]);

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

  const handleGenerateSlides = async () => {
    if (!briefId) return;

    setIsGeneratingSlides(true);
    setSlidesError(null);

    try {
      const response = await slidesApi.generateFromBrief(briefId, {
        max_slides: 10,
        style: 'professional',
        generate_images: false,  // Disabled - image generation APIs have rate limits
        theme: 'dark',
        transition: 'fade',
        duration: 5
      });

      setSlideshowData(response);
      setShowSlideshowModal(true);
    } catch (err) {
      console.error('Failed to generate slides:', err);
      setSlidesError(err instanceof Error ? err.message : 'Failed to generate slides');
    } finally {
      setIsGeneratingSlides(false);
    }
  };

  const handleDownloadSlideshow = () => {
    if (slideshowData?.html) {
      const blob = new Blob([slideshowData.html], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${currentBrief?.topic || 'slideshow'}.html`;
      a.click();
      URL.revokeObjectURL(url);
    }
  };

  const handleCopyEmbed = async () => {
    if (slideshowData?.embed_code) {
      await navigator.clipboard.writeText(slideshowData.embed_code);
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

  const isGenerating = ['generating', 'processing', 'analyzing', 'researching', 'optimizing'].includes(status || '');
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
                  <p className="text-sm text-muted-foreground">{currentPhase || 'Processing...'}</p>
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

              <p className="mt-3 text-xs text-muted-foreground">
                {useSSE ? 'Streaming updates in real-time' : 'Checking progress every 2 seconds'}
              </p>
            </div>
          )}

          {/* Status Banner */}
          {isComplete && (
            <div className="glass rounded-xl p-6 bg-success/10 border-success/30">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <CheckCircle2 className="h-6 w-6 text-success" />
                  <div>
                    <h3 className="font-semibold text-success">Generation Complete</h3>
                    <p className="text-sm text-success/80">Your content is ready to view and export.</p>
                  </div>
                </div>

                {/* Generate Slides Button */}
                <button
                  onClick={handleGenerateSlides}
                  disabled={isGeneratingSlides}
                  className={cn(
                    "flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors",
                    isGeneratingSlides
                      ? "bg-secondary text-muted-foreground cursor-not-allowed"
                      : "bg-accent hover:bg-accent/90 text-accent-foreground"
                  )}
                >
                  {isGeneratingSlides ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Generating...
                    </>
                  ) : (
                    <>
                      <Presentation className="h-4 w-4" />
                      Generate Slides
                    </>
                  )}
                </button>
              </div>

              {slidesError && (
                <p className="mt-3 text-sm text-destructive">{slidesError}</p>
              )}
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

          {/* Slideshow Quick Access */}
          {slideshowData && (
            <div className="glass rounded-xl p-6">
              <h3 className="font-semibold text-foreground mb-4 flex items-center gap-2">
                <Presentation className="h-4 w-4" />
                Generated Slideshow
              </h3>
              <div className="space-y-3">
                <button
                  onClick={() => setShowSlideshowModal(true)}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-accent hover:bg-accent/90 text-accent-foreground font-medium transition-colors"
                >
                  <ExternalLink className="h-4 w-4" />
                  View Slideshow
                </button>
                <button
                  onClick={handleDownloadSlideshow}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-secondary hover:bg-secondary/80 text-foreground font-medium transition-colors"
                >
                  <Download className="h-4 w-4" />
                  Download HTML
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Slideshow Modal */}
      {showSlideshowModal && slideshowData && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80">
          <div className="relative w-full max-w-5xl h-[80vh] mx-4">
            {/* Modal Header */}
            <div className="absolute top-0 left-0 right-0 flex items-center justify-between p-4 bg-background/90 backdrop-blur rounded-t-xl z-10">
              <h3 className="font-semibold text-foreground">{slideshowData.title}</h3>
              <div className="flex items-center gap-2">
                <button
                  onClick={handleCopyEmbed}
                  className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-secondary hover:bg-secondary/80 text-foreground text-sm transition-colors"
                >
                  <Copy className="h-4 w-4" />
                  Embed Code
                </button>
                <button
                  onClick={handleDownloadSlideshow}
                  className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-primary hover:bg-primary/90 text-primary-foreground text-sm transition-colors"
                >
                  <Download className="h-4 w-4" />
                  Download
                </button>
                <button
                  onClick={() => setShowSlideshowModal(false)}
                  className="p-2 rounded-lg hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
            </div>

            {/* Slideshow Preview */}
            <div className="w-full h-full pt-16 rounded-xl overflow-hidden">
              <iframe
                srcDoc={slideshowData.html}
                className="w-full h-full border-0 rounded-b-xl"
                title="Slideshow Preview"
                sandbox="allow-scripts"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
