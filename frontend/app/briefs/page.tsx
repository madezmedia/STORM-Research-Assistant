'use client';

import { useEffect } from 'react';
import Link from 'next/link';
import { PageHeader } from '@/components/layout';
import { useBriefStore } from '@/stores';
import { Plus, FileText, Clock, CheckCircle2, XCircle, Loader2, Trash2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Brief } from '@/types/api';
import { formatDistanceToNow } from 'date-fns';

function getStatusBadge(status: Brief['status']) {
  const config: Record<string, { icon: typeof CheckCircle2; label: string; className: string }> = {
    complete: { icon: CheckCircle2, label: 'Completed', className: 'bg-success/20 text-success' },
    completed: { icon: CheckCircle2, label: 'Completed', className: 'bg-success/20 text-success' },
    generating: { icon: Loader2, label: 'Generating', className: 'bg-primary/20 text-primary' },
    analyzing: { icon: Loader2, label: 'Analyzing', className: 'bg-primary/20 text-primary' },
    researching: { icon: Loader2, label: 'Researching', className: 'bg-primary/20 text-primary' },
    optimizing: { icon: Loader2, label: 'Optimizing', className: 'bg-primary/20 text-primary' },
    failed: { icon: XCircle, label: 'Failed', className: 'bg-destructive/20 text-destructive' },
    created: { icon: Clock, label: 'Created', className: 'bg-muted text-muted-foreground' },
    pending: { icon: Clock, label: 'Pending', className: 'bg-muted text-muted-foreground' },
  };

  const defaultConfig = { icon: Clock, label: status || 'Unknown', className: 'bg-muted text-muted-foreground' };
  const { icon: Icon, label, className } = config[status] || defaultConfig;

  const isLoading = ['generating', 'analyzing', 'researching', 'optimizing'].includes(status);

  return (
    <span className={cn('inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium', className)}>
      <Icon className={cn('h-3 w-3', isLoading && 'animate-spin')} />
      {label}
    </span>
  );
}

export default function BriefsPage() {
  const { briefs, fetchBriefs, deleteBrief, isLoading } = useBriefStore();

  useEffect(() => {
    fetchBriefs();
  }, [fetchBriefs]);

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (confirm('Are you sure you want to delete this brief?')) {
      await deleteBrief(id);
    }
  };

  return (
    <div className="animate-fade-in">
      <PageHeader
        title="Content Briefs"
        description="Manage your content generation briefs"
        actions={
          <Link
            href="/studio"
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground font-medium hover:bg-primary/90 transition-colors glow-sm"
          >
            <Plus className="h-4 w-4" />
            New Brief
          </Link>
        }
      />

      {isLoading ? (
        <div className="glass rounded-xl p-12">
          <div className="flex flex-col items-center justify-center text-muted-foreground">
            <Loader2 className="h-12 w-12 animate-spin mb-4" />
            <p>Loading briefs...</p>
          </div>
        </div>
      ) : briefs.length === 0 ? (
        <div className="glass rounded-xl p-12">
          <div className="flex flex-col items-center justify-center text-muted-foreground">
            <FileText className="h-16 w-16 mb-4 opacity-50" />
            <h3 className="text-lg font-semibold text-foreground mb-2">No briefs yet</h3>
            <p className="mb-6">Create your first content brief to get started.</p>
            <Link
              href="/studio"
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground font-medium hover:bg-primary/90 transition-colors"
            >
              <Plus className="h-4 w-4" />
              Create Brief
            </Link>
          </div>
        </div>
      ) : (
        <div className="glass rounded-xl overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left p-4 text-sm font-medium text-muted-foreground">Topic</th>
                <th className="text-left p-4 text-sm font-medium text-muted-foreground">Type</th>
                <th className="text-left p-4 text-sm font-medium text-muted-foreground">Status</th>
                <th className="text-left p-4 text-sm font-medium text-muted-foreground">Created</th>
                <th className="text-left p-4 text-sm font-medium text-muted-foreground">Words</th>
                <th className="text-right p-4 text-sm font-medium text-muted-foreground">Actions</th>
              </tr>
            </thead>
            <tbody>
              {briefs.map((brief) => (
                <tr
                  key={brief.id}
                  className="border-b border-border/50 hover:bg-secondary/30 transition-colors"
                >
                  <td className="p-4">
                    <Link
                      href={`/studio/${brief.id}`}
                      className="font-medium text-foreground hover:text-primary transition-colors"
                    >
                      {brief.topic}
                    </Link>
                  </td>
                  <td className="p-4">
                    <span className="text-sm text-muted-foreground capitalize">
                      {brief.content_type?.replace('-', ' ') || 'Blog Post'}
                    </span>
                  </td>
                  <td className="p-4">
                    {getStatusBadge(brief.status)}
                  </td>
                  <td className="p-4">
                    <span className="text-sm text-muted-foreground">
                      {brief.created_at
                        ? formatDistanceToNow(new Date(brief.created_at), { addSuffix: true })
                        : 'Just now'
                      }
                    </span>
                  </td>
                  <td className="p-4">
                    <span className="text-sm text-muted-foreground">
                      {brief.word_count?.toLocaleString() || '—'}
                    </span>
                  </td>
                  <td className="p-4 text-right">
                    <button
                      onClick={(e) => handleDelete(brief.id, e)}
                      className="p-2 rounded-lg text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors"
                      title="Delete brief"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
