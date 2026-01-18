'use client';

import { useEffect } from 'react';
import Link from 'next/link';
import { PageHeader } from '@/components/layout';
import { useBriefStore } from '@/stores';
import {
  Trophy,
  FileText,
  Clock,
  CheckCircle2,
  BarChart3,
  ExternalLink,
  Loader2
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Brief } from '@/types/api';
import { formatDistanceToNow } from 'date-fns';

function ResultCard({ brief }: { brief: Brief }) {
  const isComplete = brief.status === 'complete';

  return (
    <Link
      href={`/results/${brief.id}`}
      className="glass rounded-xl p-6 transition-all duration-300 hover:glow-sm group"
    >
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className={cn(
            'p-3 rounded-lg',
            isComplete ? 'bg-success/20' : 'bg-secondary'
          )}>
            {isComplete ? (
              <CheckCircle2 className="h-6 w-6 text-success" />
            ) : (
              <Clock className="h-6 w-6 text-muted-foreground" />
            )}
          </div>
          <div>
            <h3 className="font-semibold text-foreground group-hover:text-primary transition-colors line-clamp-1">
              {brief.topic}
            </h3>
            <p className="text-sm text-muted-foreground capitalize">
              {brief.content_type?.replace('-', ' ')}
            </p>
          </div>
        </div>
        <ExternalLink className="h-5 w-5 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
      </div>

      <div className="grid grid-cols-3 gap-4 mt-4">
        <div className="text-center p-3 rounded-lg bg-secondary/30">
          <div className="text-xl font-bold text-foreground">
            {brief.word_count?.toLocaleString() || '—'}
          </div>
          <div className="text-xs text-muted-foreground">Words</div>
        </div>
        <div className="text-center p-3 rounded-lg bg-secondary/30">
          <div className="text-xl font-bold text-foreground">—</div>
          <div className="text-xs text-muted-foreground">Sources</div>
        </div>
        <div className="text-center p-3 rounded-lg bg-secondary/30">
          <div className="text-xl font-bold text-foreground">—</div>
          <div className="text-xs text-muted-foreground">SEO Score</div>
        </div>
      </div>

      <div className="mt-4 pt-4 border-t border-border/50 flex items-center justify-between text-sm text-muted-foreground">
        <span>
          {brief.created_at
            ? formatDistanceToNow(new Date(brief.created_at), { addSuffix: true })
            : 'Just now'
          }
        </span>
        <span className={cn(
          'px-2 py-0.5 rounded-full text-xs font-medium',
          isComplete ? 'bg-success/20 text-success' : 'bg-muted text-muted-foreground'
        )}>
          {brief.status}
        </span>
      </div>
    </Link>
  );
}

export default function ResultsPage() {
  const { briefs, fetchBriefs, isLoading } = useBriefStore();

  useEffect(() => {
    fetchBriefs();
  }, [fetchBriefs]);

  const completedBriefs = briefs.filter((b: Brief) => b.status === 'complete');
  const totalWords = completedBriefs.reduce((sum: number, b: Brief) => sum + (b.word_count || 0), 0);

  return (
    <div className="animate-fade-in">
      <PageHeader
        title="Results Gallery"
        description="View and analyze your generated content"
      />

      {/* Stats Summary */}
      <div className="grid gap-6 md:grid-cols-4 mb-8">
        {[
          { label: 'Total Content', value: completedBriefs.length, icon: FileText },
          { label: 'Total Words', value: totalWords.toLocaleString(), icon: BarChart3 },
          { label: 'Avg. Words', value: completedBriefs.length ? Math.round(totalWords / completedBriefs.length).toLocaleString() : '—', icon: Trophy },
          { label: 'Success Rate', value: briefs.length ? `${Math.round((completedBriefs.length / briefs.length) * 100)}%` : '—', icon: CheckCircle2 },
        ].map((stat) => (
          <div key={stat.label} className="glass rounded-xl p-4 flex items-center gap-4">
            <div className="p-3 rounded-lg bg-primary/10">
              <stat.icon className="h-6 w-6 text-primary" />
            </div>
            <div>
              <div className="text-2xl font-bold text-foreground">{stat.value}</div>
              <div className="text-sm text-muted-foreground">{stat.label}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Results Grid */}
      {isLoading ? (
        <div className="glass rounded-xl p-12 flex flex-col items-center justify-center text-muted-foreground">
          <Loader2 className="h-12 w-12 animate-spin mb-4" />
          <p>Loading results...</p>
        </div>
      ) : completedBriefs.length === 0 ? (
        <div className="glass rounded-xl p-12 text-center">
          <Trophy className="h-16 w-16 text-muted-foreground/50 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-foreground mb-2">No Results Yet</h3>
          <p className="text-muted-foreground mb-6">
            Generate some content to see your results here.
          </p>
          <Link
            href="/studio"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground font-medium hover:bg-primary/90 transition-colors"
          >
            Create Content
          </Link>
        </div>
      ) : (
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {completedBriefs.map((brief: Brief) => (
            <ResultCard key={brief.id} brief={brief} />
          ))}
        </div>
      )}
    </div>
  );
}
