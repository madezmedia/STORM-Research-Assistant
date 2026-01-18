/**
 * Briefs API
 */
import { api } from './client';
import type { Brief, CreateBriefData, GenerationStatus, GeneratedContent } from '@/types/api';

export const briefsApi = {
  // List all briefs
  list: () => api.get<Brief[]>('/briefs'),

  // Get single brief
  get: (id: string) => api.get<Brief>(`/briefs/${id}`),

  // Create new brief
  create: (data: CreateBriefData) => api.post<Brief>('/briefs', data),

  // Update brief
  update: (id: string, data: Partial<CreateBriefData>) =>
    api.put<Brief>(`/briefs/${id}`, data),

  // Delete brief
  delete: (id: string) => api.delete(`/briefs/${id}`),

  // Start generation
  generate: (id: string, options?: { skip_analysis?: boolean }) =>
    api.post<GenerationStatus>(`/briefs/${id}/generate`, options || {}),

  // Get generation status
  getStatus: (id: string) =>
    api.get<GenerationStatus>(`/briefs/${id}/status`),
};

export const contentApi = {
  // Get generated content
  get: (briefId: string) =>
    api.get<GeneratedContent>(`/content/${briefId}`),

  // Alias for get
  getContent: (briefId: string) =>
    api.get<GeneratedContent>(`/content/${briefId}`),

  // Update content
  update: (id: string, data: { content: string; version?: number }) =>
    api.put<GeneratedContent>(`/content/${id}`, data),

  // Export content
  export: (id: string, format: 'markdown' | 'html' | 'pdf' | 'docx') =>
    api.post<{ export_id: string; status: string; download_url?: string }>(
      `/content/${id}/export`,
      { format }
    ),

  // Get SEO analysis
  getSeo: (id: string) =>
    api.get<{ content_id: string; seo_score: Record<string, unknown>; recommendations: unknown[] }>(
      `/content/${id}/seo`
    ),

  // Alias for getSeo
  getSeoAnalysis: (id: string) =>
    api.get<{ content_id: string; seo_score: Record<string, unknown>; recommendations: unknown[] }>(
      `/content/${id}/seo`
    ),
};

export default briefsApi;
