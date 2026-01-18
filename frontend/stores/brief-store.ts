/**
 * Brief & Content Store
 */
import { create } from 'zustand';
import { briefsApi, contentApi } from '@/lib/api';
import type { Brief, CreateBriefData, GeneratedContent } from '@/types/api';

interface BriefState {
  // Data
  briefs: Brief[];
  currentBrief: Brief | null;
  currentContent: GeneratedContent | null;

  // Loading states
  isLoadingBriefs: boolean;
  isLoadingBrief: boolean;
  isLoadingContent: boolean;
  isCreating: boolean;
  // Alias for convenience
  isLoading: boolean;

  // Error states
  error: string | null;

  // Actions
  fetchBriefs: () => Promise<void>;
  fetchBrief: (id: string) => Promise<Brief | null>;
  createBrief: (data: CreateBriefData) => Promise<Brief | null>;
  updateBrief: (id: string, data: Partial<CreateBriefData>) => Promise<Brief | null>;
  deleteBrief: (id: string) => Promise<boolean>;
  fetchContent: (briefId: string) => Promise<GeneratedContent | null>;
  setCurrentBrief: (brief: Brief | null) => void;
  setCurrentContent: (content: GeneratedContent | null) => void;
  clearError: () => void;
}

export const useBriefStore = create<BriefState>((set, get) => ({
  // Initial state
  briefs: [],
  currentBrief: null,
  currentContent: null,
  isLoadingBriefs: false,
  isLoadingBrief: false,
  isLoadingContent: false,
  isCreating: false,
  isLoading: false,
  error: null,

  // Actions
  fetchBriefs: async () => {
    set({ isLoadingBriefs: true, isLoading: true, error: null });
    try {
      const briefs = await briefsApi.list();
      set({ briefs, isLoadingBriefs: false, isLoading: false });
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : 'Failed to load briefs',
        isLoadingBriefs: false,
        isLoading: false,
      });
    }
  },

  fetchBrief: async (id: string) => {
    set({ isLoadingBrief: true, error: null });
    try {
      const brief = await briefsApi.get(id);
      set({ currentBrief: brief, isLoadingBrief: false });
      return brief;
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : 'Failed to load brief',
        isLoadingBrief: false,
      });
      return null;
    }
  },

  createBrief: async (data: CreateBriefData) => {
    set({ isCreating: true, error: null });
    try {
      const brief = await briefsApi.create(data);
      set((state) => ({
        briefs: [brief, ...state.briefs],
        currentBrief: brief,
        isCreating: false,
      }));
      return brief;
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : 'Failed to create brief',
        isCreating: false,
      });
      return null;
    }
  },

  updateBrief: async (id: string, data: Partial<CreateBriefData>) => {
    set({ error: null });
    try {
      const updated = await briefsApi.update(id, data);
      set((state) => ({
        briefs: state.briefs.map((b) => (b.id === id ? updated : b)),
        currentBrief: state.currentBrief?.id === id ? updated : state.currentBrief,
      }));
      return updated;
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : 'Failed to update brief',
      });
      return null;
    }
  },

  deleteBrief: async (id: string) => {
    set({ error: null });
    try {
      await briefsApi.delete(id);
      set((state) => ({
        briefs: state.briefs.filter((b) => b.id !== id),
        currentBrief: state.currentBrief?.id === id ? null : state.currentBrief,
      }));
      return true;
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : 'Failed to delete brief',
      });
      return false;
    }
  },

  fetchContent: async (briefId: string) => {
    set({ isLoadingContent: true, error: null });
    try {
      const content = await contentApi.get(briefId);
      set({ currentContent: content, isLoadingContent: false });
      return content;
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : 'Failed to load content',
        isLoadingContent: false,
      });
      return null;
    }
  },

  setCurrentBrief: (brief) => set({ currentBrief: brief }),
  setCurrentContent: (content) => set({ currentContent: content }),
  clearError: () => set({ error: null }),
}));

export default useBriefStore;
