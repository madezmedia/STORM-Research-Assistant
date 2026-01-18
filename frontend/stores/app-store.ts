/**
 * Global Application Store
 */
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { healthApi } from '@/lib/api';

interface AppState {
  // UI preferences
  sidebarCollapsed: boolean;
  theme: 'dark' | 'light' | 'system';
  defaultModel: string;

  // API health
  isApiHealthy: boolean;
  apiVersion: string | null;
  lastHealthCheck: Date | null;

  // Actions
  toggleSidebar: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
  setTheme: (theme: AppState['theme']) => void;
  setDefaultModel: (model: string) => void;
  checkApiHealth: () => Promise<void>;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      // Initial state
      sidebarCollapsed: false,
      theme: 'dark',
      defaultModel: 'openai/gpt-4o-mini',
      isApiHealthy: false,
      apiVersion: null,
      lastHealthCheck: null,

      // Actions
      toggleSidebar: () =>
        set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),

      setSidebarCollapsed: (collapsed) =>
        set({ sidebarCollapsed: collapsed }),

      setTheme: (theme) => set({ theme }),

      setDefaultModel: (model) => set({ defaultModel: model }),

      checkApiHealth: async () => {
        try {
          const health = await healthApi.check();
          set({
            isApiHealthy: health.status === 'running' || health.status === 'healthy',
            apiVersion: health.version || null,
            lastHealthCheck: new Date(),
          });
        } catch {
          set({
            isApiHealthy: false,
            lastHealthCheck: new Date(),
          });
        }
      },
    }),
    {
      name: 'storm-app-storage',
      partialize: (state) => ({
        sidebarCollapsed: state.sidebarCollapsed,
        theme: state.theme,
        defaultModel: state.defaultModel,
      }),
    }
  )
);

export default useAppStore;
