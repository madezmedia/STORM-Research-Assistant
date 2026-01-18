'use client';

import { CheckCircle2, XCircle, RefreshCw, Server } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useAppStore } from '@/stores';

export function SystemStatus() {
  const { isApiHealthy, checkApiHealth } = useAppStore();

  return (
    <div className="glass rounded-xl p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-foreground">System Status</h2>
        <button
          onClick={checkApiHealth}
          className="p-2 rounded-lg hover:bg-secondary/50 transition-colors text-muted-foreground hover:text-foreground"
          title="Refresh status"
        >
          <RefreshCw className="h-4 w-4" />
        </button>
      </div>

      <div className="space-y-4">
        <div className="flex items-center justify-between p-4 rounded-lg bg-secondary/30">
          <div className="flex items-center gap-3">
            <Server className="h-5 w-5 text-muted-foreground" />
            <div>
              <p className="font-medium text-foreground">STORM API</p>
              <p className="text-xs text-muted-foreground">Backend services</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {isApiHealthy ? (
              <>
                <CheckCircle2 className="h-5 w-5 text-success" />
                <span className="text-sm font-medium text-success">Online</span>
              </>
            ) : (
              <>
                <XCircle className="h-5 w-5 text-destructive" />
                <span className="text-sm font-medium text-destructive">Offline</span>
              </>
            )}
          </div>
        </div>

        <div className={cn(
          'p-4 rounded-lg text-sm',
          isApiHealthy ? 'bg-success/10 text-success' : 'bg-destructive/10 text-destructive'
        )}>
          {isApiHealthy ? (
            <p>All systems operational. Ready to generate content.</p>
          ) : (
            <p>API is unreachable. Please check that the backend server is running on port 8000.</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default SystemStatus;
