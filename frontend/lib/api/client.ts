/**
 * API Client for STORM Backend
 * Provides typed, centralized API access
 *
 * Configuration:
 * Set NEXT_PUBLIC_STORM_API_URL in Vercel environment variables to point to your backend.
 * Example: https://storm-api-xxxx.vercel.app
 *
 * Set NEXT_PUBLIC_STORM_API_KEY for API authentication.
 */

// API base URL - use env var if set, otherwise relative path
const API_BASE = process.env.NEXT_PUBLIC_STORM_API_URL
  ? `${process.env.NEXT_PUBLIC_STORM_API_URL}/api/v1`
  : '/api/v1';

// API Key for authentication (optional - required when backend has STORM_API_KEY set)
const API_KEY = process.env.NEXT_PUBLIC_STORM_API_KEY;

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
        // Include API key if configured
        ...(API_KEY && { 'X-API-Key': API_KEY }),
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

      // Handle specific error codes
      if (response.status === 401) {
        throw new ApiError(
          data?.detail || 'Authentication required. Please check your API key.',
          401,
          data
        );
      }

      if (response.status === 429) {
        throw new ApiError(
          data?.detail || 'Rate limit exceeded. Please try again later.',
          429,
          data
        );
      }

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
