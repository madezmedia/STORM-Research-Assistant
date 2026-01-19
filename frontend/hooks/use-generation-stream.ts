'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { GenerationStatus } from '@/types/api';

interface UseGenerationStreamOptions {
  onComplete?: (status: GenerationStatus) => void;
  onError?: (error: Error) => void;
  enabled?: boolean;
}

interface UseGenerationStreamReturn {
  status: GenerationStatus | null;
  isStreaming: boolean;
  error: Error | null;
  startStreaming: () => void;
  stopStreaming: () => void;
}

/**
 * Hook for streaming generation status via Server-Sent Events (SSE).
 *
 * Replaces polling with real-time updates from the backend.
 * Automatically cleans up on unmount or when generation completes.
 *
 * @param briefId - The ID of the brief to stream status for
 * @param options - Configuration options
 * @returns Stream state and control functions
 *
 * @example
 * ```tsx
 * const { status, isStreaming, startStreaming } = useGenerationStream(briefId, {
 *   onComplete: (status) => console.log('Generation complete!', status),
 *   enabled: true
 * });
 * ```
 */
export function useGenerationStream(
  briefId: string | null,
  options: UseGenerationStreamOptions = {}
): UseGenerationStreamReturn {
  const { onComplete, onError, enabled = true } = options;

  const [status, setStatus] = useState<GenerationStatus | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const eventSourceRef = useRef<EventSource | null>(null);
  const onCompleteRef = useRef(onComplete);
  const onErrorRef = useRef(onError);

  // Keep callback refs up to date
  useEffect(() => {
    onCompleteRef.current = onComplete;
    onErrorRef.current = onError;
  }, [onComplete, onError]);

  const stopStreaming = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  const startStreaming = useCallback(() => {
    if (!briefId) return;

    // Close existing connection
    stopStreaming();

    try {
      const eventSource = new EventSource(`/api/v1/briefs/${briefId}/stream`);
      eventSourceRef.current = eventSource;
      setIsStreaming(true);
      setError(null);

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as GenerationStatus;
          setStatus(data);

          // Check if generation completed or failed
          if (data.status === 'complete' || data.status === 'completed' || data.status === 'failed') {
            stopStreaming();
            onCompleteRef.current?.(data);
          }
        } catch (parseError) {
          console.error('Failed to parse SSE data:', parseError);
        }
      };

      eventSource.onerror = (e) => {
        console.error('SSE error:', e);
        const err = new Error('Connection to generation stream failed');
        setError(err);
        stopStreaming();
        onErrorRef.current?.(err);
      };

      eventSource.onopen = () => {
        console.log(`SSE connection opened for brief ${briefId}`);
      };

    } catch (err) {
      const error = err instanceof Error ? err : new Error('Failed to start streaming');
      setError(error);
      onErrorRef.current?.(error);
    }
  }, [briefId, stopStreaming]);

  // Auto-start streaming when enabled and briefId changes
  useEffect(() => {
    if (enabled && briefId) {
      startStreaming();
    }

    return () => {
      stopStreaming();
    };
  }, [briefId, enabled, startStreaming, stopStreaming]);

  return {
    status,
    isStreaming,
    error,
    startStreaming,
    stopStreaming
  };
}

/**
 * Hook to manage multiple generation streams for the status panel.
 *
 * Tracks multiple briefs being generated concurrently.
 */
export function useMultipleGenerationStreams() {
  const [activeStreams, setActiveStreams] = useState<Map<string, GenerationStatus>>(new Map());
  const eventSourcesRef = useRef<Map<string, EventSource>>(new Map());

  const addStream = useCallback((briefId: string) => {
    if (eventSourcesRef.current.has(briefId)) return;

    const eventSource = new EventSource(`/api/v1/briefs/${briefId}/stream`);
    eventSourcesRef.current.set(briefId, eventSource);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as GenerationStatus;
        setActiveStreams(prev => {
          const updated = new Map(prev);
          updated.set(briefId, data);
          return updated;
        });

        if (data.status === 'complete' || data.status === 'completed' || data.status === 'failed') {
          removeStream(briefId);
        }
      } catch (err) {
        console.error('Failed to parse SSE data:', err);
      }
    };

    eventSource.onerror = () => {
      removeStream(briefId);
    };
  }, []);

  const removeStream = useCallback((briefId: string) => {
    const eventSource = eventSourcesRef.current.get(briefId);
    if (eventSource) {
      eventSource.close();
      eventSourcesRef.current.delete(briefId);
    }
    setActiveStreams(prev => {
      const updated = new Map(prev);
      updated.delete(briefId);
      return updated;
    });
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      eventSourcesRef.current.forEach(es => es.close());
      eventSourcesRef.current.clear();
    };
  }, []);

  return {
    activeStreams,
    addStream,
    removeStream,
    streamCount: activeStreams.size
  };
}

export default useGenerationStream;
