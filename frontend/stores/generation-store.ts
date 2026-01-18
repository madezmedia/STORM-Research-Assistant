/**
 * Generation Progress Store
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

  // Polling intervals
  pollingIntervals: Map<string, NodeJS.Timeout>;

  // Actions
  startGeneration: (briefId: string) => Promise<GenerationStatus | null>;
  pollStatus: (briefId: string) => Promise<GenerationStatus | null>;
  startPolling: (briefId: string, onComplete?: (status: GenerationStatus) => void) => void;
  stopPolling: (briefId?: string) => void;
  clearGeneration: (briefId: string) => void;
  getStatus: (briefId: string) => GenerationStatus | undefined;
}

export const useGenerationStore = create<GenerationState>((set, get) => ({
  activeGenerations: new Map(),
  pollingIntervals: new Map(),
  currentBriefId: null,
  status: null,
  progress: 0,
  currentPhase: null,
  content: null,

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

  startPolling: (briefId: string, onComplete?: (status: GenerationStatus) => void) => {
    const { pollingIntervals, pollStatus, stopPolling } = get();

    // Set current brief
    set({ currentBriefId: briefId });

    // Clear existing polling
    if (pollingIntervals.has(briefId)) {
      stopPolling(briefId);
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

        if (status.status === 'complete' || status.status === 'failed') {
          stopPolling(briefId);
          onComplete?.(status);
        }
      }
    };

    // Initial poll
    poll();

    // Set up interval
    const interval = setInterval(poll, 3000);
    set((state) => {
      const newIntervals = new Map(state.pollingIntervals);
      newIntervals.set(briefId, interval);
      return { pollingIntervals: newIntervals };
    });
  },

  stopPolling: (briefId?: string) => {
    const { pollingIntervals, currentBriefId } = get();
    const id = briefId || currentBriefId;
    if (!id) return;

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

  clearGeneration: (briefId: string) => {
    const { stopPolling } = get();
    stopPolling(briefId);
    set((state) => {
      const newMap = new Map(state.activeGenerations);
      newMap.delete(briefId);
      return { activeGenerations: newMap };
    });
  },

  getStatus: (briefId: string) => {
    return get().activeGenerations.get(briefId);
  },
}));

export default useGenerationStore;
