'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { PageHeader } from '@/components/layout';
import { useBriefStore, useGenerationStore } from '@/stores';
import { Sparkles, ChevronRight, ChevronLeft, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { ContentType } from '@/types/api';

const contentTypes = [
  { value: 'blog-post', label: 'Blog Post', description: 'Long-form educational content' },
  { value: 'local-guide', label: 'Local Guide', description: 'Location-based content with local insights' },
  { value: 'comparison', label: 'Comparison', description: 'Compare products, services, or options' },
  { value: 'how-to', label: 'How-To Guide', description: 'Step-by-step tutorial content' },
  { value: 'case-study', label: 'Case Study', description: 'In-depth analysis and results' },
];

const tones = [
  { value: 'professional', label: 'Professional' },
  { value: 'casual', label: 'Casual' },
  { value: 'friendly', label: 'Friendly' },
  { value: 'authoritative', label: 'Authoritative' },
];

export default function StudioPage() {
  const router = useRouter();
  const { createBrief, isLoading: isBriefLoading } = useBriefStore();
  const { startGeneration } = useGenerationStore();

  const [step, setStep] = useState(1);
  const [formData, setFormData] = useState({
    topic: '',
    content_type: 'blog-post',
    word_count: 2000,
    tone: 'professional',
    primary_keyword: '',
    include_examples: true,
    include_stats: true,
  });
  const [isGenerating, setIsGenerating] = useState(false);

  const updateForm = (field: string, value: unknown) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
    // Auto-fill primary keyword from topic if empty
    if (field === 'topic' && !formData.primary_keyword) {
      const keyword = (value as string).split(' ').slice(0, 4).join(' ');
      setFormData((prev) => ({ ...prev, primary_keyword: keyword }));
    }
  };

  const handleCreate = async () => {
    if (!formData.topic.trim()) return;

    setIsGenerating(true);
    try {
      const brief = await createBrief({
        topic: formData.topic,
        content_type: formData.content_type as ContentType,
        word_count: formData.word_count,
        tone: formData.tone,
        seo: {
          primary_keyword: formData.primary_keyword || formData.topic,
          secondary_keywords: [],
          difficulty: 'medium',
          intent: 'informational',
        },
        brand_direction: 'neutral',
        target_audience: {
          segment: 'general',
          pain_points: [],
          expertise: 'intermediate',
        },
        include_examples: formData.include_examples,
        include_stats: formData.include_stats,
        include_local_data: formData.content_type === 'local-guide',
      });

      if (brief?.id) {
        // Start generation
        await startGeneration(brief.id);
        router.push(`/studio/${brief.id}`);
      }
    } catch (error) {
      console.error('Failed to create brief:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  const canProceed = () => {
    switch (step) {
      case 1:
        return formData.topic.trim().length > 0;
      case 2:
        return formData.content_type !== '';
      case 3:
        return true;
      default:
        return false;
    }
  };

  return (
    <div className="animate-fade-in max-w-4xl mx-auto">
      <PageHeader
        title="Content Studio"
        description="Create research-backed content with AI"
      />

      {/* Progress Steps */}
      <div className="glass rounded-xl p-6 mb-8">
        <div className="flex items-center justify-between">
          {['Topic', 'Type', 'Options'].map((label, i) => (
            <div key={label} className="flex items-center">
              <div className={cn(
                'flex h-10 w-10 items-center justify-center rounded-full text-sm font-medium transition-colors',
                step > i + 1 ? 'bg-success text-success-foreground' :
                step === i + 1 ? 'bg-primary text-primary-foreground glow-sm' :
                'bg-secondary text-muted-foreground'
              )}>
                {step > i + 1 ? '✓' : i + 1}
              </div>
              <span className={cn(
                'ml-3 text-sm font-medium',
                step >= i + 1 ? 'text-foreground' : 'text-muted-foreground'
              )}>
                {label}
              </span>
              {i < 2 && (
                <ChevronRight className="mx-4 h-5 w-5 text-muted-foreground" />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Step Content */}
      <div className="glass rounded-xl p-8">
        {/* Step 1: Topic */}
        {step === 1 && (
          <div className="space-y-6 animate-fade-in">
            <div>
              <h2 className="text-2xl font-bold text-foreground mb-2">What do you want to write about?</h2>
              <p className="text-muted-foreground">Enter your research topic and we'll generate comprehensive content using STORM analysis.</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-foreground mb-2">
                Research Topic
              </label>
              <input
                type="text"
                value={formData.topic}
                onChange={(e) => updateForm('topic', e.target.value)}
                placeholder="e.g., Best Coffee Shops in Austin, Texas"
                className="w-full px-4 py-3 rounded-lg bg-secondary/50 border border-border text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary"
                autoFocus
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-foreground mb-2">
                Primary Keyword (SEO)
              </label>
              <input
                type="text"
                value={formData.primary_keyword}
                onChange={(e) => updateForm('primary_keyword', e.target.value)}
                placeholder="e.g., coffee shops austin"
                className="w-full px-4 py-3 rounded-lg bg-secondary/50 border border-border text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>
          </div>
        )}

        {/* Step 2: Content Type */}
        {step === 2 && (
          <div className="space-y-6 animate-fade-in">
            <div>
              <h2 className="text-2xl font-bold text-foreground mb-2">Choose content type</h2>
              <p className="text-muted-foreground">Select the format that best fits your content needs.</p>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              {contentTypes.map((type) => (
                <button
                  key={type.value}
                  onClick={() => updateForm('content_type', type.value)}
                  className={cn(
                    'p-4 rounded-lg border-2 text-left transition-all',
                    formData.content_type === type.value
                      ? 'border-primary bg-primary/10 glow-sm'
                      : 'border-border bg-secondary/30 hover:border-primary/50'
                  )}
                >
                  <h3 className="font-semibold text-foreground">{type.label}</h3>
                  <p className="text-sm text-muted-foreground mt-1">{type.description}</p>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Step 3: Options */}
        {step === 3 && (
          <div className="space-y-6 animate-fade-in">
            <div>
              <h2 className="text-2xl font-bold text-foreground mb-2">Customize your content</h2>
              <p className="text-muted-foreground">Fine-tune the generation options.</p>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Word Count
                </label>
                <input
                  type="number"
                  min="500"
                  max="10000"
                  step="500"
                  value={formData.word_count}
                  onChange={(e) => updateForm('word_count', parseInt(e.target.value))}
                  className="w-full px-4 py-3 rounded-lg bg-secondary/50 border border-border text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Tone
                </label>
                <select
                  value={formData.tone}
                  onChange={(e) => updateForm('tone', e.target.value)}
                  className="w-full px-4 py-3 rounded-lg bg-secondary/50 border border-border text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
                >
                  {tones.map((tone) => (
                    <option key={tone.value} value={tone.value}>{tone.label}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="space-y-4">
              <label className="flex items-center gap-3 p-4 rounded-lg bg-secondary/30 cursor-pointer hover:bg-secondary/50 transition-colors">
                <input
                  type="checkbox"
                  checked={formData.include_examples}
                  onChange={(e) => updateForm('include_examples', e.target.checked)}
                  className="h-5 w-5 rounded border-border text-primary focus:ring-primary"
                />
                <div>
                  <span className="font-medium text-foreground">Include Examples</span>
                  <p className="text-sm text-muted-foreground">Add real-world examples to illustrate points</p>
                </div>
              </label>

              <label className="flex items-center gap-3 p-4 rounded-lg bg-secondary/30 cursor-pointer hover:bg-secondary/50 transition-colors">
                <input
                  type="checkbox"
                  checked={formData.include_stats}
                  onChange={(e) => updateForm('include_stats', e.target.checked)}
                  className="h-5 w-5 rounded border-border text-primary focus:ring-primary"
                />
                <div>
                  <span className="font-medium text-foreground">Include Statistics</span>
                  <p className="text-sm text-muted-foreground">Add relevant data and statistics</p>
                </div>
              </label>
            </div>
          </div>
        )}

        {/* Navigation */}
        <div className="flex items-center justify-between mt-8 pt-6 border-t border-border">
          <button
            onClick={() => setStep(step - 1)}
            disabled={step === 1}
            className={cn(
              'flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-colors',
              step === 1
                ? 'text-muted-foreground cursor-not-allowed'
                : 'text-foreground hover:bg-secondary/50'
            )}
          >
            <ChevronLeft className="h-5 w-5" />
            Back
          </button>

          {step < 3 ? (
            <button
              onClick={() => setStep(step + 1)}
              disabled={!canProceed()}
              className={cn(
                'flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all',
                canProceed()
                  ? 'bg-primary text-primary-foreground hover:bg-primary/90 glow-sm'
                  : 'bg-secondary text-muted-foreground cursor-not-allowed'
              )}
            >
              Continue
              <ChevronRight className="h-5 w-5" />
            </button>
          ) : (
            <button
              onClick={handleCreate}
              disabled={isGenerating || isBriefLoading}
              className={cn(
                'flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all',
                'bg-gradient-to-r from-primary to-accent text-white hover:opacity-90 glow'
              )}
            >
              {isGenerating ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  Creating...
                </>
              ) : (
                <>
                  <Sparkles className="h-5 w-5" />
                  Generate Content
                </>
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
