/**
 * UI Preferences Store
 */
import { create } from 'zustand';

export interface Toast {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  description?: string;
  duration?: number;
}

interface UIState {
  // Modal states
  isExportDialogOpen: boolean;
  exportDialogContentId: string | null;
  isDeleteDialogOpen: boolean;
  deleteDialogItemId: string | null;
  isCommandPaletteOpen: boolean;

  // Toast notifications
  toasts: Toast[];

  // Actions
  openExportDialog: (contentId: string) => void;
  closeExportDialog: () => void;
  openDeleteDialog: (itemId: string) => void;
  closeDeleteDialog: () => void;
  toggleCommandPalette: () => void;
  addToast: (toast: Omit<Toast, 'id'>) => void;
  removeToast: (id: string) => void;
  clearToasts: () => void;
}

export const useUIStore = create<UIState>((set) => ({
  // Initial state
  isExportDialogOpen: false,
  exportDialogContentId: null,
  isDeleteDialogOpen: false,
  deleteDialogItemId: null,
  isCommandPaletteOpen: false,
  toasts: [],

  // Actions
  openExportDialog: (contentId) =>
    set({ isExportDialogOpen: true, exportDialogContentId: contentId }),

  closeExportDialog: () =>
    set({ isExportDialogOpen: false, exportDialogContentId: null }),

  openDeleteDialog: (itemId) =>
    set({ isDeleteDialogOpen: true, deleteDialogItemId: itemId }),

  closeDeleteDialog: () =>
    set({ isDeleteDialogOpen: false, deleteDialogItemId: null }),

  toggleCommandPalette: () =>
    set((state) => ({ isCommandPaletteOpen: !state.isCommandPaletteOpen })),

  addToast: (toast) =>
    set((state) => ({
      toasts: [
        ...state.toasts,
        { ...toast, id: Math.random().toString(36).slice(2) },
      ],
    })),

  removeToast: (id) =>
    set((state) => ({
      toasts: state.toasts.filter((t) => t.id !== id),
    })),

  clearToasts: () => set({ toasts: [] }),
}));

export default useUIStore;
