/**
 * Slideshow & Video API
 */
import { api } from './client';
import type {
  SlidesCapabilities,
  SlideGenerationRequest,
  SlideshowRequest,
  SlideshowResponse,
  VideoRequest,
  VideoResponse,
  Slide,
} from '@/types/api';

export interface BriefSlidesRequest {
  max_slides?: number;
  style?: 'professional' | 'minimalist' | 'vibrant';
  generate_images?: boolean;
  autoplay?: boolean;
  duration?: number;
  transition?: 'fade' | 'slide' | 'zoom';
  theme?: 'dark' | 'light';
}

export interface BriefSlidesResponse {
  job_id: string;
  brief_id: string;
  status: string;
  slide_count: number;
  title: string;
  html: string;
  embed_code: string;
  slides: Slide[];
}

export const slidesApi = {
  // Check capabilities
  getStatus: () =>
    api.get<SlidesCapabilities>('/slides/status'),

  // Alias for getStatus
  getCapabilities: () =>
    api.get<SlidesCapabilities>('/slides/status'),

  // Generate slides
  generate: (data: SlideGenerationRequest) =>
    api.post<{
      success: boolean;
      job_id: string;
      slide_count: number;
      style: string;
      slides: Slide[];
    }>('/slides/generate', data),

  // Create slideshow
  createSlideshow: (data: SlideshowRequest) =>
    api.post<SlideshowResponse>('/slides/slideshow', data),

  // Alias for createSlideshow
  generateSlideshow: (data: SlideshowRequest) =>
    api.post<SlideshowResponse>('/slides/slideshow', data),

  // Create video
  createVideo: (data: VideoRequest) =>
    api.post<VideoResponse>('/slides/video', data),

  // Get job result
  getJob: (jobId: string) =>
    api.get<{
      id: string;
      type: 'slides' | 'slideshow' | 'video';
      status: string;
      slides?: Slide[];
      html?: string;
      video_base64?: string;
      slide_count: number;
      created_at: string;
    }>(`/slides/${jobId}`),

  // Get embed HTML (returns raw HTML)
  getEmbed: (jobId: string) =>
    fetch(`/api/v1/slides/${jobId}/embed`).then(r => r.text()),

  /**
   * Generate slideshow directly from a brief's generated content.
   * The brief must have completed content generation first.
   */
  generateFromBrief: (briefId: string, options: BriefSlidesRequest = {}) =>
    api.post<BriefSlidesResponse>(`/briefs/${briefId}/slides`, {
      max_slides: options.max_slides ?? 10,
      style: options.style ?? 'professional',
      generate_images: options.generate_images ?? true,
      autoplay: options.autoplay ?? true,
      duration: options.duration ?? 5,
      transition: options.transition ?? 'fade',
      theme: options.theme ?? 'dark',
    }),
};

export default slidesApi;
