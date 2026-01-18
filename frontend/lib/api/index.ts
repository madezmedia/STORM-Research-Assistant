/**
 * API exports
 */
export { api, ApiError } from './client';
export { briefsApi, contentApi } from './briefs';
export { geoApi } from './geo';
export { slidesApi } from './slides';

// Health check - uses root endpoints, not /api/v1
import type { HealthStatus } from '@/types/api';

export const healthApi = {
  check: async (): Promise<HealthStatus> => {
    // Health endpoint is at root, not /api/v1
    const response = await fetch('/health');
    if (!response.ok) {
      throw new Error('Health check failed');
    }
    return response.json();
  },
  root: async (): Promise<HealthStatus> => {
    // Root endpoint returns API info
    const response = await fetch('/api/v1/health');
    if (!response.ok) {
      // Fallback to /health
      const fallback = await fetch('/health');
      if (!fallback.ok) throw new Error('Health check failed');
      return fallback.json();
    }
    return response.json();
  },
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
