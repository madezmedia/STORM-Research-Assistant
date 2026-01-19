'use client';

import { useState, useEffect } from 'react';
import { PageHeader } from '@/components/layout';
import { slidesApi } from '@/lib/api';
import type { SlideshowSummary, SlideshowDetail } from '@/lib/api/slides';
import { SlidesCapabilities, SlideshowResponse } from '@/types/api';
import {
  Presentation,
  Loader2,
  Download,
  Play,
  Image as ImageIcon,
  Video,
  Clock,
  Palette,
  CheckCircle2,
  AlertCircle,
  History,
  ExternalLink,
  X
} from 'lucide-react';
import { cn } from '@/lib/utils';

const styles = [
  { value: 'professional', label: 'Professional', description: 'Clean and business-focused' },
  { value: 'minimalist', label: 'Minimalist', description: 'Simple and elegant' },
  { value: 'vibrant', label: 'Vibrant', description: 'Bold colors and energy' },
  { value: 'dark', label: 'Dark Mode', description: 'Modern dark theme' },
];

const themes = [
  { value: 'business', label: 'Business' },
  { value: 'creative', label: 'Creative' },
  { value: 'tech', label: 'Technology' },
  { value: 'education', label: 'Education' },
];

export default function SlidesPage() {
  const [capabilities, setCapabilities] = useState<SlidesCapabilities | null>(null);
  const [content, setContent] = useState('');
  const [title, setTitle] = useState('');
  const [style, setStyle] = useState('professional');
  const [theme, setTheme] = useState('business');
  const [duration, setDuration] = useState(5);
  const [generateImages, setGenerateImages] = useState(false);  // Disabled by default - API rate limits

  const [isGenerating, setIsGenerating] = useState(false);
  const [slideshow, setSlideshow] = useState<SlideshowResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Saved slideshows state
  const [savedSlideshows, setSavedSlideshows] = useState<SlideshowSummary[]>([]);
  const [loadingSaved, setLoadingSaved] = useState(true);
  const [selectedSlideshow, setSelectedSlideshow] = useState<SlideshowDetail | null>(null);
  const [loadingSlideshow, setLoadingSlideshow] = useState(false);

  useEffect(() => {
    // Check capabilities on mount
    slidesApi.getCapabilities()
      .then((caps) => {
        // Compute convenience properties
        setCapabilities({
          ...caps,
          image_generation: caps.zai_configured || caps.gateway_configured,
          video_generation: caps.ffmpeg?.available || false,
        });
      })
      .catch(console.error);

    // Load saved slideshows
    loadSavedSlideshows();
  }, []);

  const loadSavedSlideshows = async () => {
    setLoadingSaved(true);
    try {
      const result = await slidesApi.list();
      setSavedSlideshows(result.slideshows || []);
    } catch (err) {
      console.error('Failed to load saved slideshows:', err);
    } finally {
      setLoadingSaved(false);
    }
  };

  const viewSlideshow = async (id: string) => {
    setLoadingSlideshow(true);
    try {
      const detail = await slidesApi.getSlideshow(id);
      setSelectedSlideshow(detail);
    } catch (err) {
      console.error('Failed to load slideshow:', err);
    } finally {
      setLoadingSlideshow(false);
    }
  };

  const handleGenerate = async () => {
    if (!content.trim()) return;

    setIsGenerating(true);
    setError(null);
    setSlideshow(null);

    try {
      const result = await slidesApi.generateSlideshow({
        content,
        title: title || 'Untitled Presentation',
        style,
        theme,
        duration,  // seconds per slide
        generate_images: generateImages,
      });
      setSlideshow(result);
      // Refresh saved slideshows list
      loadSavedSlideshows();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Generation failed');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDownloadHtml = () => {
    if (slideshow?.html) {
      const blob = new Blob([slideshow.html], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${title || 'slideshow'}.html`;
      a.click();
      URL.revokeObjectURL(url);
    }
  };

  return (
    <div className="animate-fade-in">
      <PageHeader
        title="Slideshow Studio"
        description="Transform content into beautiful presentations"
      />

      {/* Capabilities Status */}
      {capabilities && (
        <div className="glass rounded-xl p-4 mb-6 flex items-center gap-4">
          <div className="flex items-center gap-2">
            <ImageIcon className={cn('h-5 w-5', capabilities.image_generation ? 'text-success' : 'text-muted-foreground')} />
            <span className="text-sm text-muted-foreground">
              Image Generation: {capabilities.image_generation ? 'Available' : 'Unavailable'}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Video className={cn('h-5 w-5', capabilities.video_generation ? 'text-success' : 'text-muted-foreground')} />
            <span className="text-sm text-muted-foreground">
              Video Export: {capabilities.video_generation ? 'Available' : 'Unavailable'}
            </span>
          </div>
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Input Panel */}
        <div className="space-y-6">
          {/* Content Input */}
          <div className="glass rounded-xl p-6">
            <h2 className="text-lg font-semibold text-foreground mb-4">Content</h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Presentation Title
                </label>
                <input
                  type="text"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="My Presentation"
                  className="w-full px-4 py-3 rounded-lg bg-secondary/50 border border-border text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Content (Markdown or plain text)
                </label>
                <textarea
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  placeholder="Paste your article content here or enter key points for each slide..."
                  rows={10}
                  className="w-full px-4 py-3 rounded-lg bg-secondary/50 border border-border text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary resize-none"
                />
              </div>
            </div>
          </div>

          {/* Style Options */}
          <div className="glass rounded-xl p-6">
            <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
              <Palette className="h-5 w-5 text-accent" />
              Style & Theme
            </h2>

            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                {styles.map((s) => (
                  <button
                    key={s.value}
                    onClick={() => setStyle(s.value)}
                    className={cn(
                      'p-3 rounded-lg border-2 text-left transition-all',
                      style === s.value
                        ? 'border-primary bg-primary/10'
                        : 'border-border hover:border-primary/50'
                    )}
                  >
                    <span className="font-medium text-foreground text-sm">{s.label}</span>
                    <p className="text-xs text-muted-foreground mt-0.5">{s.description}</p>
                  </button>
                ))}
              </div>

              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Theme
                </label>
                <select
                  value={theme}
                  onChange={(e) => setTheme(e.target.value)}
                  className="w-full px-4 py-3 rounded-lg bg-secondary/50 border border-border text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
                >
                  {themes.map((t) => (
                    <option key={t.value} value={t.value}>{t.label}</option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {/* Generation Options */}
          <div className="glass rounded-xl p-6">
            <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
              <Clock className="h-5 w-5 text-primary" />
              Options
            </h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Slide Duration (seconds)
                </label>
                <input
                  type="number"
                  min="3"
                  max="30"
                  value={duration}
                  onChange={(e) => setDuration(parseInt(e.target.value))}
                  className="w-full px-4 py-3 rounded-lg bg-secondary/50 border border-border text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
                />
              </div>

              <label className="flex items-center gap-3 p-4 rounded-lg bg-secondary/30 cursor-pointer hover:bg-secondary/50 transition-colors">
                <input
                  type="checkbox"
                  checked={generateImages}
                  onChange={(e) => setGenerateImages(e.target.checked)}
                  className="h-5 w-5 rounded border-border text-primary focus:ring-primary"
                />
                <div>
                  <span className="font-medium text-foreground">Generate AI Images</span>
                  <p className="text-sm text-muted-foreground">Create visuals for each slide using AI</p>
                </div>
              </label>
            </div>

            <button
              onClick={handleGenerate}
              disabled={isGenerating || !content.trim()}
              className={cn(
                'w-full mt-6 flex items-center justify-center gap-2 px-6 py-3 rounded-lg font-medium transition-all',
                isGenerating || !content.trim()
                  ? 'bg-secondary text-muted-foreground cursor-not-allowed'
                  : 'bg-gradient-to-r from-primary to-accent text-white hover:opacity-90 glow'
              )}
            >
              {isGenerating ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  Generating Slideshow...
                </>
              ) : (
                <>
                  <Presentation className="h-5 w-5" />
                  Generate Slideshow
                </>
              )}
            </button>
          </div>
        </div>

        {/* Preview Panel */}
        <div className="space-y-6">
          <div className="glass rounded-xl p-6 min-h-[600px]">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-foreground">Preview</h2>
              {slideshow && (
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleDownloadHtml}
                    className="flex items-center gap-2 px-3 py-2 rounded-lg bg-secondary hover:bg-secondary/80 text-foreground text-sm transition-colors"
                  >
                    <Download className="h-4 w-4" />
                    Download HTML
                  </button>
                </div>
              )}
            </div>

            {error && (
              <div className="p-4 rounded-lg bg-destructive/10 text-destructive flex items-center gap-3">
                <AlertCircle className="h-5 w-5" />
                {error}
              </div>
            )}

            {isGenerating && (
              <div className="flex flex-col items-center justify-center h-[500px] text-muted-foreground">
                <Loader2 className="h-12 w-12 animate-spin mb-4" />
                <p>Generating your slideshow...</p>
                <p className="text-sm mt-2">This may take a minute if generating images.</p>
              </div>
            )}

            {slideshow?.html && !isGenerating && (
              <div className="space-y-4">
                <div className="flex items-center gap-2 text-success">
                  <CheckCircle2 className="h-5 w-5" />
                  <span>Slideshow generated successfully!</span>
                </div>

                <div className="aspect-video rounded-lg overflow-hidden border border-border bg-black">
                  <iframe
                    srcDoc={slideshow.html}
                    className="w-full h-full"
                    title="Slideshow Preview"
                    sandbox="allow-scripts"
                  />
                </div>

                {slideshow.slides && (
                  <div className="grid grid-cols-4 gap-2">
                    {slideshow.slides.slice(0, 8).map((slide, i) => (
                      <div
                        key={i}
                        className="aspect-video rounded bg-secondary/50 flex items-center justify-center text-xs text-muted-foreground"
                      >
                        Slide {i + 1}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {!slideshow && !isGenerating && !error && (
              <div className="flex flex-col items-center justify-center h-[500px] text-muted-foreground">
                <Presentation className="h-16 w-16 opacity-50 mb-4" />
                <p>Enter content and generate a slideshow</p>
                <p className="text-sm mt-2">Preview will appear here</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Saved Slideshows Section */}
      <div className="mt-8">
        <div className="glass rounded-xl p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-foreground flex items-center gap-2">
              <History className="h-5 w-5 text-accent" />
              Previous Slideshows
            </h2>
            <span className="text-sm text-muted-foreground">
              {savedSlideshows.length} saved
            </span>
          </div>

          {loadingSaved ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : savedSlideshows.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <Presentation className="h-12 w-12 mx-auto mb-3 opacity-50" />
              <p>No saved slideshows yet</p>
              <p className="text-sm mt-1">Generated slideshows will appear here</p>
            </div>
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {savedSlideshows.map((ss) => (
                <div
                  key={ss.id}
                  className="p-4 rounded-lg bg-secondary/30 border border-border hover:border-primary/50 transition-colors"
                >
                  <div className="flex items-start justify-between mb-2">
                    <h3 className="font-medium text-foreground truncate flex-1">
                      {ss.title || 'Untitled'}
                    </h3>
                    <span className="text-xs text-muted-foreground ml-2">
                      {ss.slide_count} slides
                    </span>
                  </div>

                  <div className="flex items-center gap-2 text-xs text-muted-foreground mb-3">
                    <span className="capitalize">{ss.style}</span>
                    <span>•</span>
                    <span className="capitalize">{ss.theme}</span>
                    {ss.created_at && (
                      <>
                        <span>•</span>
                        <span>{new Date(ss.created_at).toLocaleDateString()}</span>
                      </>
                    )}
                  </div>

                  <button
                    onClick={() => viewSlideshow(ss.id)}
                    disabled={loadingSlideshow}
                    className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-primary/20 hover:bg-primary/30 text-primary text-sm font-medium transition-colors"
                  >
                    <ExternalLink className="h-4 w-4" />
                    View Slideshow
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Slideshow View Modal */}
      {selectedSlideshow && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80">
          <div className="relative w-full max-w-5xl h-[85vh] mx-4">
            {/* Modal Header */}
            <div className="absolute top-0 left-0 right-0 flex items-center justify-between p-4 bg-background/90 backdrop-blur rounded-t-xl z-10">
              <h3 className="font-semibold text-foreground">{selectedSlideshow.title}</h3>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => {
                    if (selectedSlideshow.html) {
                      const blob = new Blob([selectedSlideshow.html], { type: 'text/html' });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = `${selectedSlideshow.title || 'slideshow'}.html`;
                      a.click();
                      URL.revokeObjectURL(url);
                    }
                  }}
                  className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-primary hover:bg-primary/90 text-primary-foreground text-sm transition-colors"
                >
                  <Download className="h-4 w-4" />
                  Download
                </button>
                <button
                  onClick={() => setSelectedSlideshow(null)}
                  className="p-2 rounded-lg hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
            </div>

            {/* Slideshow Content */}
            <div className="w-full h-full pt-16 rounded-xl overflow-hidden">
              {loadingSlideshow ? (
                <div className="flex items-center justify-center h-full bg-background">
                  <Loader2 className="h-12 w-12 animate-spin text-muted-foreground" />
                </div>
              ) : selectedSlideshow.html ? (
                <iframe
                  srcDoc={selectedSlideshow.html}
                  className="w-full h-full border-0 rounded-b-xl"
                  title="Slideshow"
                  sandbox="allow-scripts"
                />
              ) : (
                <div className="flex items-center justify-center h-full bg-background text-muted-foreground">
                  <p>No slideshow content available</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
