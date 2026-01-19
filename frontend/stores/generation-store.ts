/**
 * Generation Progress Store
 *
 * Manages content generation progress with SSE streaming (default) or polling fallback.
 */
import { create } from 'zustand';
import { briefsApi } from '@/lib/api';
import type { GenerationStatus } from '@/types/api';

interface GenerationState {
  // Active generations (can track multiple)
  activeGenerations: Map<string, GenerationStatus>;

  // Current brief being viewed (for convenience)
  currentBriefId: string | null;

  // Convenience properties for current brief
  status: string | null;
  progress: number;
  currentPhase: string | null;
  content: { content: string; title: string; word_count?: number; sources?: unknown[] } | null;

  // Connection management
  pollingIntervals: Map<string, ReturnType<typeof setInterval>>;
  sseConnections: Map<string, EventSource>;

  // SSE support flag (browser compatibility)
  useSSE: boolean;

  // Actions
  startGeneration: (briefId: string) => Promise<GenerationStatus | null>;
  pollStatus: (briefId: string) => Promise<GenerationStatus | null>;
  startPolling: (briefId: string, onComplete?: (status: GenerationStatus) => void) => void;
  startStreaming: (briefId: string, onComplete?: (status: GenerationStatus) => void) => void;
  stopTracking: (briefId?: string) => void;
  clearGeneration: (briefId: string) => void;
  getStatus: (briefId: string) => GenerationStatus | undefined;
  updateStatus: (briefId: string, status: GenerationStatus) => void;
  setUseSSE: (useSSE: boolean) => void;

  // Legacy alias
  stopPolling: (briefId?: string) => void;
}

// Check if SSE is supported
const isSSESupported = typeof window !== 'undefined' && typeof EventSource !== 'undefined';

export const useGenerationStore = create<GenerationState>((set, get) => ({
  activeGenerations: new Map(),
  pollingIntervals: new Map(),
  sseConnections: new Map(),
  currentBriefId: null,
  status: null,
  progress: 0,
  currentPhase: null,
  content: null,
  useSSE: isSSESupported,

  startGeneration: async (briefId: string) => {
    try {
      const status = await briefsApi.generate(briefId);
      set((state) => {
        const newMap = new Map(state.activeGenerations);
        newMap.set(briefId, status);
        return { activeGenerations: newMap };
      });
      return status;
    } catch (err) {
      console.error('Failed to start generation:', err);
      return null;
    }
  },

  pollStatus: async (briefId: string) => {
    try {
      const status = await briefsApi.getStatus(briefId);
      set((state) => {
        const newMap = new Map(state.activeGenerations);
        newMap.set(briefId, status);
        return { activeGenerations: newMap };
      });
      return status;
    } catch (err) {
      console.error('Failed to poll status:', err);
      return null;
    }
  },

  updateStatus: (briefId: string, status: GenerationStatus) => {
    set((state) => {
      const newMap = new Map(state.activeGenerations);
      newMap.set(briefId, status);

      // Update convenience properties if this is the current brief
      const updates: Partial<GenerationState> = { activeGenerations: newMap };
      if (state.currentBriefId === briefId) {
        updates.status = status.status;
        updates.progress = status.progress;
        updates.currentPhase = status.current_phase;
      }

      return updates;
    });
  },

  /**
   * Start tracking generation progress via SSE (Server-Sent Events).
   * This is the preferred method as it provides real-time updates.
   */
  startStreaming: (briefId: string, onComplete?: (status: GenerationStatus) => void) => {
    const { sseConnections, stopTracking, updateStatus } = get();

    // Set current brief
    set({ currentBriefId: briefId });

    // Close existing connection
    if (sseConnections.has(briefId)) {
      stopTracking(briefId);
    }

    try {
      const eventSource = new EventSource(`/api/v1/briefs/${briefId}/stream`);

      eventSource.onmessage = (event) => {
        try {
          const status = JSON.parse(event.data) as GenerationStatus;
          updateStatus(briefId, status);

          // Update convenience properties
          set({
            status: status.status,
            progress: status.progress,
            currentPhase: status.current_phase,
          });

          // Check completion
          if (status.status === 'complete' || status.status === 'completed' || status.status === 'failed') {
            stopTracking(briefId);
            onComplete?.(status);
          }
        } catch (parseError) {
          console.error('Failed to parse SSE data:', parseError);
        }
      };

      eventSource.onerror = (e) => {
        console.error('SSE connection error, falling back to polling:', e);
        stopTracking(briefId);
        // Fallback to polling on SSE error
        get().startPolling(briefId, onComplete);
      };

      set((state) => {
        const newConnections = new Map(state.sseConnections);
        newConnections.set(briefId, eventSource);
        return { sseConnections: newConnections };
      });

    } catch (err) {
      console.error('Failed to start SSE, falling back to polling:', err);
      // Fallback to polling
      get().startPolling(briefId, onComplete);
    }
  },

  /**
   * Start tracking generation progress via polling.
   * Used as a fallback when SSE is not available.
   */
  startPolling: (briefId: string, onComplete?: (status: GenerationStatus) => void) => {
    const { pollingIntervals, pollStatus, stopTracking } = get();

    // Set current brief
    set({ currentBriefId: briefId });

    // Clear existing polling
    if (pollingIntervals.has(briefId)) {
      stopTracking(briefId);
    }

    const poll = async () => {
      const status = await pollStatus(briefId);
      if (status) {
        // Update convenience properties
        set({
          status: status.status,
          progress: status.progress,
          currentPhase: status.current_phase,
        });

        if (status.status === 'complete' || status.status === 'completed' || status.status === 'failed') {
          stopTracking(briefId);
          onComplete?.(status);
        }
      }
    };

    // Initial poll
    poll();

    // Set up interval (poll every 2 seconds)
    const interval = setInterval(poll, 2000);
    set((state) => {
      const newIntervals = new Map(state.pollingIntervals);
      newIntervals.set(briefId, interval);
      return { pollingIntervals: newIntervals };
    });
  },

  /**
   * Stop tracking generation progress (both SSE and polling).
   */
  stopTracking: (briefId?: string) => {
    const { pollingIntervals, sseConnections, currentBriefId } = get();
    const id = briefId || currentBriefId;
    if (!id) return;

    // Close SSE connection
    const eventSource = sseConnections.get(id);
    if (eventSource) {
      eventSource.close();
      set((state) => {
        const newConnections = new Map(state.sseConnections);
        newConnections.delete(id);
        return { sseConnections: newConnections };
      });
    }

    // Clear polling interval
    const interval = pollingIntervals.get(id);
    if (interval) {
      clearInterval(interval);
      set((state) => {
        const newIntervals = new Map(state.pollingIntervals);
        newIntervals.delete(id);
        return { pollingIntervals: newIntervals };
      });
    }
  },

  // Legacy alias for backwards compatibility
  stopPolling: (briefId?: string) => {
    get().stopTracking(briefId);
  },

  clearGeneration: (briefId: string) => {
    const { stopTracking } = get();
    stopTracking(briefId);
    set((state) => {
      const newMap = new Map(state.activeGenerations);
      newMap.delete(briefId);
      return { activeGenerations: newMap };
    });
  },

  getStatus: (briefId: string) => {
    return get().activeGenerations.get(briefId);
  },

  setUseSSE: (useSSE: boolean) => {
    set({ useSSE: useSSE && isSSESupported });
  },
}));

export default useGenerationStore;
