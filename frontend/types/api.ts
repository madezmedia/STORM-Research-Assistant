/**
 * Core API Types
 */

// Brief types
export type BriefStatus =
  | 'created'
  | 'pending'
  | 'analyzing'
  | 'researching'
  | 'generating'
  | 'optimizing'
  | 'complete'
  | 'completed'
  | 'failed';

export type ContentType =
  | 'blog-post'
  | 'guide'
  | 'comparison'
  | 'case-study'
  | 'local-guide'
  | 'landing-page'
  | 'how-to';

export interface SeoConfig {
  primary_keyword?: string;
  secondary_keywords?: string[];
  target_volume?: number;
  difficulty?: 'easy' | 'medium' | 'hard';
  intent?: 'informational' | 'commercial' | 'transactional';
}

export interface GeoConfig {
  enabled?: boolean;
  location?: {
    country?: string;
    state?: string;
    city?: string;
    zip?: string;
  };
  local_keywords?: string[];
  geo_intent?: 'local-service' | 'regional-guide' | 'national';
}

export interface TargetAudience {
  segment?: string;
  pain_points?: string[];
  expertise?: 'beginner' | 'intermediate' | 'advanced';
}

export interface Brief {
  id: string;
  topic: string;
  content_type: ContentType;
  status: BriefStatus;
  progress?: number;
  current_phase?: string;
  word_count: number;
  tone: string;
  seo?: SeoConfig;
  geo?: GeoConfig;
  target_audience?: TargetAudience;
  brand_direction?: string;
  include_examples?: boolean;
  include_stats?: boolean;
  include_local_data?: boolean;
  created_at: string;
  updated_at?: string;
}

export interface CreateBriefData {
  topic: string;
  content_type?: ContentType;
  word_count?: number;
  tone?: string;
  seo?: SeoConfig;
  geo?: GeoConfig;
  target_audience?: TargetAudience;
  brand_direction?: string;
  include_examples?: boolean;
  include_stats?: boolean;
  include_local_data?: boolean;
}

// Generation types
export interface GenerationStatus {
  brief_id: string;
  status: BriefStatus;
  progress: number;
  current_phase: string;
  estimated_time_remaining?: number;
}

// Content types
export interface ContentSection {
  heading: string;
  word_count?: number;
  content?: string;
}

export interface SeoScore {
  overall: number;
  keyword_density?: number;
  readability?: number;
  sources_cited?: number;
}

export interface ContentSource {
  url: string;
  title: string;
}

export interface GeneratedContent {
  id: string;
  brief_id: string;
  title: string;
  meta_description?: string;
  content: string;
  word_count: number;
  sections: ContentSection[];
  seo_score?: SeoScore;
  sources?: ContentSource[];
  created_at: string;
}

// GEO types
export interface GeoAnalysis {
  id: string;
  brand_name: string;
  domain: string;
  keywords: [string, number][];
  topics: string[];
  questions: string[];
  related_searches: string[];
  autocomplete: string[];
  pages_analyzed: number;
  content_summary?: string;
  status: 'pending' | 'analyzing' | 'complete' | 'failed';
  created_at: string;
}

export interface GeneratedPrompt {
  prompt_text: string;
  category: string;
  target_keywords: string[];
  expected_mention?: string;
}

// Slideshow types
export interface SlidesCapabilities {
  modules_available: boolean;
  ffmpeg: { available: boolean; version?: string };
  zai_configured: boolean;
  gateway_configured: boolean;
  supported_formats: string[];
  // Computed convenience properties
  image_generation?: boolean;
  video_generation?: boolean;
}

export interface Slide {
  slide_number: number;
  title: string;
  content: string;
  visual_prompt?: string;
  image_data?: string;
  image_url?: string;
}

export interface SlideGenerationRequest {
  content: string;
  max_slides?: number;
  style?: 'professional' | 'minimalist' | 'vibrant';
  brand_colors?: { primary?: string; secondary?: string };
  include_title_slide?: boolean;
  include_conclusion_slide?: boolean;
  generate_images?: boolean;
}

export interface SlideshowRequest {
  content: string;
  max_slides?: number;
  style?: 'professional' | 'minimalist' | 'vibrant' | 'dark' | string;
  autoplay?: boolean;
  duration?: number;
  slide_duration?: number;
  transition?: 'fade' | 'slide' | 'zoom';
  show_controls?: boolean;
  loop?: boolean;
  theme?: 'dark' | 'light' | 'business' | 'creative' | 'tech' | 'education' | string;
  title?: string;
  generate_images?: boolean;
}

export interface SlideshowResponse {
  success: boolean;
  job_id: string;
  slide_count: number;
  format: string;
  html?: string;
  embed_code?: string;
  slides?: Slide[];
}

export interface VideoRequest {
  content: string;
  max_slides?: number;
  style?: 'professional' | 'minimalist' | 'vibrant';
  duration?: number;
  resolution?: '720p' | '1080p' | '4k';
  transition?: 'fade' | 'slide';
  audio_path?: string;
}

export interface VideoResponse {
  success: boolean;
  job_id: string;
  slide_count: number;
  format: string;
  duration_seconds?: number;
  resolution?: string;
  file_size_bytes?: number;
  video_base64?: string;
  error?: string;
}

// Health check
export interface HealthStatus {
  status: string;
  api_key_configured?: boolean;
  name?: string;
  version?: string;
}
