/**
 * API exports
 */
import { api, ApiError } from './client';
export { api, ApiError };
export { briefsApi, contentApi } from './briefs';
export { geoApi } from './geo';
export { slidesApi } from './slides';

// Health check - uses root endpoints, not /api/v1
import type { HealthStatus } from '@/types/api';

// Get backend URL for health checks
const getBackendUrl = () => process.env.NEXT_PUBLIC_STORM_API_URL || '';

export const healthApi = {
  check: async (): Promise<HealthStatus> => {
    const backendUrl = getBackendUrl();
    const url = backendUrl ? `${backendUrl}/health` : '/health';

    try {
      const response = await fetch(url, {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
      });
      if (!response.ok) {
        throw new Error('Health check failed');
      }
      return response.json();
    } catch (error) {
      // Return offline status instead of throwing
      return {
        status: 'offline',
        database: false,
        redis: false,
        storm: false,
        geo_pipeline: false,
        version: 'unknown',
      } as HealthStatus;
    }
  },
  root: async (): Promise<HealthStatus> => {
    const backendUrl = getBackendUrl();
    const url = backendUrl ? `${backendUrl}/` : '/';

    try {
      const response = await fetch(url, {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
      });
      if (!response.ok) {
        return healthApi.check(); // Fallback to /health
      }
      return response.json();
    } catch {
      return healthApi.check();
    }
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
