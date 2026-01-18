/**
 * API Client for STORM Backend
 * Provides typed, centralized API access
 */

const API_BASE = '/api/v1';

export class ApiError extends Error {
  constructor(message: string, public status: number, public data?: unknown) {
    super(message);
    this.name = 'ApiError';
  }
}

interface RequestOptions extends Omit<RequestInit, 'body'> {
  body?: unknown;
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(endpoint: string, options: RequestOptions = {}): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    const { body, ...restOptions } = options;

    const config: RequestInit = {
      ...restOptions,
      headers: {
        'Content-Type': 'application/json',
        ...restOptions.headers,
      },
    };

    if (body !== undefined) {
      config.body = JSON.stringify(body);
    }

    try {
      const response = await fetch(url, config);

      // Handle empty responses
      const text = await response.text();
      const data = text ? JSON.parse(text) : null;

      if (!response.ok) {
        throw new ApiError(
          data?.detail || data?.message || 'Request failed',
          response.status,
          data
        );
      }

      return data as T;
    } catch (error) {
      if (error instanceof ApiError) throw error;
      if (error instanceof SyntaxError) {
        throw new ApiError('Invalid response from server', 500);
      }
      throw new ApiError('Network error', 0);
    }
  }

  // Generic HTTP methods
  get<T>(endpoint: string, options?: RequestOptions) {
    return this.request<T>(endpoint, { ...options, method: 'GET' });
  }

  post<T>(endpoint: string, body?: unknown, options?: RequestOptions) {
    return this.request<T>(endpoint, { ...options, method: 'POST', body });
  }

  put<T>(endpoint: string, body?: unknown, options?: RequestOptions) {
    return this.request<T>(endpoint, { ...options, method: 'PUT', body });
  }

  delete<T>(endpoint: string, options?: RequestOptions) {
    return this.request<T>(endpoint, { ...options, method: 'DELETE' });
  }
}

// Singleton instance
export const api = new ApiClient();

// Re-export for convenience
export default api;
