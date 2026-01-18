/**
 * API exports
 */
export { api, ApiError } from './client';
export { briefsApi, contentApi } from './briefs';
export { geoApi } from './geo';
export { slidesApi } from './slides';

// Health check
import { api } from './client';
import type { HealthStatus } from '@/types/api';

export const healthApi = {
  check: () => api.get<HealthStatus>('/'),
  ping: () => api.get<{ status: string }>('/health'),
};

// Test endpoints
export const testApi = {
  testLlm: (prompt?: string, model?: string) =>
    api.post<{
      success: boolean;
      response: string;
      model_used: string;
      provider: string;
    }>('/test-llm', { prompt, model }),

  testStorm: (topic: string, content_type?: string) =>
    api.post<{
      success: boolean;
      topic: string;
      perspectives: string[];
      outline: Record<string, unknown>;
      sample_content: string;
    }>('/test-storm', { topic, content_type }),
};
