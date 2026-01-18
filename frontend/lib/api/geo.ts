/**
 * GEO Intelligence API
 */
import { api } from './client';
import type { GeoAnalysis, GeneratedPrompt } from '@/types/api';

export const geoApi = {
  // Analyze website
  analyzeWebsite: (data: {
    url: string;
    brand_name: string;
    crawl_limit?: number;
    competitors?: string[];
  }) => api.post<GeoAnalysis>('/geo/analyze-website', data),

  // Get analysis results
  getAnalysis: (id: string) =>
    api.get<GeoAnalysis>(`/geo/analysis/${id}`),

  // Get keywords from analysis
  getKeywords: (id: string) =>
    api.get<{
      analysis_id: string;
      brand_name: string;
      keywords: [string, number][];
      topics: string[];
    }>(`/geo/keywords/${id}`),

  // Generate prompts
  generatePrompts: (data: {
    keywords: string[];
    questions?: string[];
    brand_name: string;
    competitors?: string[];
  }) =>
    api.post<{
      brand_name: string;
      prompt_count: number;
      prompts: GeneratedPrompt[];
    }>('/geo/generate-prompts', data),

  // Track response
  trackResponse: (data: {
    response_text: string;
    prompt: string;
    llm_name: string;
    keywords: string[];
    brand_name: string;
  }) =>
    api.post<{
      brand_mentioned: boolean;
      brand_mentions: Record<string, unknown>;
      keyword_mentions: Record<string, unknown>;
      total_mentions: number;
      position_score: number;
    }>('/geo/track-response', data),

  // List analyses
  list: (params?: { skip?: number; limit?: number; brand_name?: string }) => {
    const query = new URLSearchParams();
    if (params?.skip) query.set('skip', String(params.skip));
    if (params?.limit) query.set('limit', String(params.limit));
    if (params?.brand_name) query.set('brand_name', params.brand_name);
    const queryStr = query.toString();
    return api.get<{ count: number; analyses: GeoAnalysis[] }>(
      `/geo/analyses${queryStr ? `?${queryStr}` : ''}`
    );
  },

  // Export for GEGO
  exportGego: (analysisId: string) =>
    api.post<{
      brand_name: string;
      prompt_count: number;
      prompts: GeneratedPrompt[];
    }>('/geo/export-gego', { analysis_id: analysisId }),
};

export default geoApi;
