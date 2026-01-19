'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useGenerationStore } from '@/stores';
import { cn } from '@/lib/utils';
import {
  X,
  Loader2,
  CheckCircle2,
  XCircle,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Clock,
  Zap
} from 'lucide-react';
import type { GenerationStatus } from '@/types/api';

interface JobCardProps {
  briefId: string;
  status: GenerationStatus;
  onClose?: () => void;
}

function JobCard({ briefId, status, onClose }: JobCardProps) {
  const router = useRouter();

  const getStatusIcon = () => {
    switch (status.status) {
      case 'complete':
      case 'completed':
        return <CheckCircle2 className="w-4 h-4 text-success" />;
      case 'failed':
        return <XCircle className="w-4 h-4 text-destructive" />;
      default:
        return <Loader2 className="w-4 h-4 text-primary animate-spin" />;
    }
  };

  const getStatusColor = () => {
    switch (status.status) {
      case 'complete':
      case 'completed':
        return 'border-success/30 bg-success/5';
      case 'failed':
        return 'border-destructive/30 bg-destructive/5';
      default:
        return 'border-primary/30 bg-primary/5';
    }
  };

  const isActive = !['complete', 'completed', 'failed'].includes(status.status);

  return (
    <div className={cn(
      'p-3 rounded-lg border transition-all',
      getStatusColor()
    )}>
      <div className="flex items-start justify-between gap-2 mb-2">
        <div className="flex items-center gap-2 min-w-0">
          {getStatusIcon()}
          <span className="text-sm font-medium text-foreground truncate">
            Brief {briefId.slice(0, 8)}...
          </span>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => router.push(`/studio/${briefId}`)}
            className="p-1 hover:bg-secondary rounded text-muted-foreground hover:text-foreground transition-colors"
            title="View details"
          >
            <ExternalLink className="w-3.5 h-3.5" />
          </button>
          {onClose && (
            <button
              onClick={onClose}
              className="p-1 hover:bg-secondary rounded text-muted-foreground hover:text-foreground transition-colors"
              title="Dismiss"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Progress bar */}
      {isActive && (
        <div className="mb-2">
          <div className="h-1.5 bg-secondary rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-primary to-accent transition-all duration-300"
              style={{ width: `${status.progress || 0}%` }}
            />
          </div>
        </div>
      )}

      {/* Status details */}
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span className="flex items-center gap-1">
          {isActive ? (
            <>
              <Zap className="w-3 h-3" />
              {status.current_phase || 'Processing'}
            </>
          ) : (
            <>
              <Clock className="w-3 h-3" />
              {status.status === 'failed' ? 'Failed' : 'Complete'}
            </>
          )}
        </span>
        <span>{status.progress || 0}%</span>
      </div>

      {/* Estimated time */}
      {isActive && status.estimated_time_remaining !== undefined && status.estimated_time_remaining > 0 && (
        <div className="mt-1 text-xs text-muted-foreground">
          ~{Math.ceil(status.estimated_time_remaining / 60)} min remaining
        </div>
      )}
    </div>
  );
}

export function GenerationStatusPanel() {
  const { activeGenerations, clearGeneration } = useGenerationStore();
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [dismissedIds, setDismissedIds] = useState<Set<string>>(new Set());

  // Filter out dismissed and convert Map to array
  const visibleGenerations = Array.from(activeGenerations.entries())
    .filter(([id]) => !dismissedIds.has(id));

  // Hide panel if no visible generations
  if (visibleGenerations.length === 0) {
    return null;
  }

  const activeCount = visibleGenerations.filter(
    ([, status]) => !['complete', 'completed', 'failed'].includes(status.status)
  ).length;

  const handleDismiss = (briefId: string) => {
    setDismissedIds(prev => new Set([...prev, briefId]));
    // Also clear from store
    clearGeneration(briefId);
  };

  return (
    <div className="fixed right-4 bottom-4 z-50 w-80 max-h-[60vh] flex flex-col">
      {/* Header */}
      <button
        onClick={() => setIsCollapsed(!isCollapsed)}
        className="flex items-center justify-between w-full p-3 glass rounded-t-xl border-b border-border/50 hover:bg-secondary/50 transition-colors"
      >
        <div className="flex items-center gap-2">
          {activeCount > 0 ? (
            <Loader2 className="w-4 h-4 text-primary animate-spin" />
          ) : (
            <CheckCircle2 className="w-4 h-4 text-success" />
          )}
          <span className="font-medium text-foreground">
            {activeCount > 0 ? `${activeCount} Active` : 'Complete'}
          </span>
          <span className="text-xs text-muted-foreground">
            ({visibleGenerations.length} total)
          </span>
        </div>
        {isCollapsed ? (
          <ChevronUp className="w-4 h-4 text-muted-foreground" />
        ) : (
          <ChevronDown className="w-4 h-4 text-muted-foreground" />
        )}
      </button>

      {/* Content */}
      {!isCollapsed && (
        <div className="glass rounded-b-xl overflow-hidden">
          <div className="p-3 space-y-2 max-h-[50vh] overflow-y-auto">
            {visibleGenerations.map(([briefId, status]) => (
              <JobCard
                key={briefId}
                briefId={briefId}
                status={status}
                onClose={() => handleDismiss(briefId)}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default GenerationStatusPanel;
