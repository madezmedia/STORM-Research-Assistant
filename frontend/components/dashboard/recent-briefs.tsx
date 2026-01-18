'use client';

import Link from 'next/link';
import { FileText, Clock, ArrowRight, Loader2, CheckCircle2, XCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Brief } from '@/types/api';
import { formatDistanceToNow } from 'date-fns';

interface RecentBriefsProps {
  briefs: Brief[];
  isLoading?: boolean;
}

function getStatusIcon(status: Brief['status']) {
  const loadingStatuses = ['generating', 'analyzing', 'researching', 'optimizing'];

  if (status === 'complete' || status === 'completed') {
    return <CheckCircle2 className="h-4 w-4 text-success" />;
  }
  if (loadingStatuses.includes(status)) {
    return <Loader2 className="h-4 w-4 text-primary animate-spin" />;
  }
  if (status === 'failed') {
    return <XCircle className="h-4 w-4 text-destructive" />;
  }
  return <Clock className="h-4 w-4 text-muted-foreground" />;
}

function getStatusText(status: Brief['status']) {
  const statusLabels: Record<string, string> = {
    complete: 'Completed',
    completed: 'Completed',
    generating: 'Generating...',
    analyzing: 'Analyzing...',
    researching: 'Researching...',
    optimizing: 'Optimizing...',
    failed: 'Failed',
    created: 'Created',
  };
  return statusLabels[status] || 'Pending';
}

export function RecentBriefs({ briefs, isLoading }: RecentBriefsProps) {
  if (isLoading) {
    return (
      <div className="glass rounded-xl p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-foreground">Recent Briefs</h2>
        </div>
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="animate-pulse flex items-center gap-4 p-4 rounded-lg bg-secondary/30">
              <div className="h-10 w-10 rounded-lg bg-secondary" />
              <div className="flex-1 space-y-2">
                <div className="h-4 w-3/4 rounded bg-secondary" />
                <div className="h-3 w-1/2 rounded bg-secondary" />
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="glass rounded-xl p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-foreground">Recent Briefs</h2>
        <Link
          href="/briefs"
          className="flex items-center gap-1 text-sm text-primary hover:text-primary/80 transition-colors"
        >
          View all
          <ArrowRight className="h-4 w-4" />
        </Link>
      </div>

      {briefs.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
          <FileText className="h-12 w-12 mb-4 opacity-50" />
          <p>No briefs yet</p>
          <Link
            href="/studio"
            className="mt-2 text-sm text-primary hover:text-primary/80"
          >
            Create your first brief
          </Link>
        </div>
      ) : (
        <div className="space-y-3">
          {briefs.slice(0, 5).map((brief) => (
            <Link
              key={brief.id}
              href={`/studio/${brief.id}`}
              className="flex items-center gap-4 p-4 rounded-lg bg-secondary/30 hover:bg-secondary/50 transition-colors group"
            >
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 group-hover:bg-primary/20 transition-colors">
                <FileText className="h-5 w-5 text-primary" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="font-medium text-foreground truncate">{brief.topic}</p>
                <p className="text-sm text-muted-foreground">
                  {brief.created_at
                    ? formatDistanceToNow(new Date(brief.created_at), { addSuffix: true })
                    : 'Just now'
                  }
                </p>
              </div>
              <div className="flex items-center gap-2">
                {getStatusIcon(brief.status)}
                <span className={cn(
                  'text-xs font-medium',
                  (brief.status === 'completed' || brief.status === 'complete') && 'text-success',
                  ['generating', 'analyzing', 'researching', 'optimizing'].includes(brief.status) && 'text-primary',
                  brief.status === 'failed' && 'text-destructive',
                  ['created', 'pending'].includes(brief.status) && 'text-muted-foreground'
                )}>
                  {getStatusText(brief.status)}
                </span>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

export default RecentBriefs;
