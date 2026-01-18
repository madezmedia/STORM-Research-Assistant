"""
Image Generation Module for Slideshow Creation

Uses Vercel AI Gateway for text processing and DALL-E 3 for image generation.
Extracts key slides from content and generates visual presentations.
"""

import os
import base64
import httpx
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import json


# =============================================================================
# Pydantic Models
# =============================================================================

class SlideContent(BaseModel):
    """Content extracted for a single slide"""
    title: str
    content: str
    visual_prompt: str
    style: str = "professional"
    key_points: List[str] = []


class GeneratedSlide(BaseModel):
    """A fully generated slide with image"""
    slide_number: int
    title: str
    content: str
    visual_prompt: str
    image_data: Optional[str] = None  # base64 encoded image
    image_url: Optional[str] = None


class SlideGenerationOptions(BaseModel):
    """Options for slide generation"""
    max_slides: int = Field(default=10, ge=1, le=20)
    style: str = Field(default="professional")  # professional, minimalist, vibrant
    brand_colors: Optional[Dict[str, str]] = None  # {primary: "#hex", secondary: "#hex"}
    include_title_slide: bool = True
    include_conclusion_slide: bool = True


# =============================================================================
# Image Generation Client
# =============================================================================

class ImageGenerationClient:
    """
    Client for generating images via Z.AI GLM-Image or Vercel AI Gateway.

    Supports:
    - Z.AI GLM-Image (api.z.ai) - for high-quality text-in-image generation
    - Vercel AI Gateway (fallback) - for DALL-E 3
    """

    def __init__(
        self,
        api_key: str,
        zai_api_key: Optional[str] = None,
        gateway_url: str = "https://ai-gateway.vercel.sh/v1",
        zai_url: str = "https://api.z.ai/api/paas/v4",
        text_model: str = "openai/gpt-4o-mini",
        image_model: str = "glm-image"  # Z.AI GLM-Image model
    ):
        self.api_key = api_key  # Vercel AI Gateway key
        self.zai_api_key = zai_api_key or api_key  # Z.AI key (can be same)
        self.gateway_url = gateway_url
        self.zai_url = zai_url
        self.text_model = text_model
        self.image_model = image_model

    async def call_text_model(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> Optional[str]:
        """Call text model for content extraction via Vercel AI Gateway"""
        if not self.api_key:
            return None

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.gateway_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.text_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature
                    },
                    timeout=60.0
                )

                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Text model call failed: {e}")

        return None

    async def generate_image(
        self,
        prompt: str,
        size: str = "1728x960",  # 16:9 aspect ratio for slides
        quality: str = "standard"
    ) -> Optional[Dict[str, Any]]:
        """
        Generate image using Z.AI GLM-Image.

        GLM-Image is excellent for:
        - PPT/Slide backgrounds
        - Text-embedded images
        - Commercial posters
        - Professional graphics

        Supported sizes: 1280x1280, 1568x1056, 1056x1568, 1472x1088,
                        1088x1472, 1728x960, 960x1728
        """
        if not self.zai_api_key:
            return await self._generate_image_fallback(prompt, size, quality)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.zai_url}/images/generations",
                    headers={
                        "Authorization": f"Bearer {self.zai_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.image_model,
                        "prompt": prompt,
                        "size": size
                    },
                    timeout=120.0
                )

                if response.status_code == 200:
                    data = response.json()
                    image_url = data["data"][0].get("url")

                    # Download image and convert to base64
                    if image_url:
                        image_data = await self._download_image(image_url)
                        return {
                            "b64_json": image_data,
                            "url": image_url,
                            "model": self.image_model
                        }
                else:
                    print(f"Z.AI image generation failed: {response.status_code} - {response.text[:200]}")
                    # Try fallback
                    return await self._generate_image_fallback(prompt, size, quality)

        except Exception as e:
            print(f"Z.AI image generation error: {e}")
            return await self._generate_image_fallback(prompt, size, quality)

        return None

    async def _download_image(self, url: str) -> Optional[str]:
        """Download image from URL and return as base64"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=60.0)
                if response.status_code == 200:
                    return base64.b64encode(response.content).decode("utf-8")
        except Exception as e:
            print(f"Image download error: {e}")
        return None

    async def _generate_image_fallback(
        self,
        prompt: str,
        size: str = "1792x1024",
        quality: str = "standard"
    ) -> Optional[Dict[str, Any]]:
        """Fallback to DALL-E 3 via Vercel AI Gateway"""
        if not self.api_key:
            return None

        # Map Z.AI sizes to DALL-E sizes
        dalle_size_map = {
            "1728x960": "1792x1024",
            "960x1728": "1024x1792",
            "1280x1280": "1024x1024",
            "1568x1056": "1792x1024",
            "1056x1568": "1024x1792"
        }
        dalle_size = dalle_size_map.get(size, "1792x1024")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.gateway_url}/images/generations",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "dall-e-3",
                        "prompt": prompt,
                        "n": 1,
                        "size": dalle_size,
                        "quality": quality,
                        "response_format": "b64_json"
                    },
                    timeout=120.0
                )

                if response.status_code == 200:
                    data = response.json()
                    return {
                        "b64_json": data["data"][0].get("b64_json"),
                        "url": data["data"][0].get("url"),
                        "model": "dall-e-3"
                    }
                else:
                    print(f"DALL-E fallback failed: {response.status_code} - {response.text[:200]}")
        except Exception as e:
            print(f"DALL-E fallback error: {e}")

        return None


# =============================================================================
# Slide Extraction
# =============================================================================

async def extract_slides_from_content(
    client: ImageGenerationClient,
    content: str,
    options: SlideGenerationOptions
) -> List[SlideContent]:
    """Extract key slides from blog/article content"""

    style_descriptions = {
        "professional": "Clean corporate aesthetic, blue/white colors, sans-serif fonts, subtle gradients",
        "minimalist": "Ultra-clean whitespace, monochromatic, single accent color, geometric shapes",
        "vibrant": "Bold energetic colors, high contrast, dynamic compositions, modern gradients"
    }

    style_desc = style_descriptions.get(options.style, style_descriptions["professional"])

    extraction_prompt = f"""You are a presentation designer. Extract {options.max_slides} key slides from this content.

CONTENT:
{content[:8000]}  # Limit content length

REQUIREMENTS:
1. {"Create a title slide as slide 1" if options.include_title_slide else "Start with the first content slide"}
2. Each slide should have ONE main idea
3. Visual prompts should describe a professional presentation background/illustration
4. {"Include a conclusion/summary slide at the end" if options.include_conclusion_slide else "End with the last content point"}
5. Style: {style_desc}

Return a JSON array with this exact structure:
[
  {{
    "title": "Slide title (max 8 words)",
    "content": "2-3 bullet points or 1 short paragraph (max 50 words)",
    "visual_prompt": "Detailed image description for slide background (be specific about colors, composition, style)",
    "style": "{options.style}",
    "key_points": ["point 1", "point 2"]
  }}
]

Generate exactly {options.max_slides} slides. Return ONLY the JSON array, no other text."""

    response = await client.call_text_model(extraction_prompt, max_tokens=3000)

    if response:
        try:
            # Handle potential markdown code blocks
            json_str = response.strip()
            if json_str.startswith("```"):
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]

            slides_data = json.loads(json_str.strip())
            return [SlideContent(**slide) for slide in slides_data]
        except Exception as e:
            print(f"Failed to parse slides JSON: {e}")

    # Fallback: generate basic slides
    return _generate_fallback_slides(content, options)


def _generate_fallback_slides(content: str, options: SlideGenerationOptions) -> List[SlideContent]:
    """Generate basic fallback slides if LLM extraction fails"""
    # Split content into paragraphs
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

    slides = []

    # Title slide
    if options.include_title_slide:
        title = paragraphs[0][:50] if paragraphs else "Presentation"
        slides.append(SlideContent(
            title=title,
            content="",
            visual_prompt=f"Professional presentation title slide with abstract geometric background, {options.style} style",
            style=options.style
        ))

    # Content slides
    for i, para in enumerate(paragraphs[1:options.max_slides-1]):
        slides.append(SlideContent(
            title=f"Key Point {i+1}",
            content=para[:200],
            visual_prompt=f"Professional presentation slide background, {options.style} style, subtle abstract patterns",
            style=options.style
        ))

    # Conclusion slide
    if options.include_conclusion_slide and len(slides) < options.max_slides:
        slides.append(SlideContent(
            title="Conclusion",
            content="Thank you for your attention",
            visual_prompt=f"Professional conclusion slide with elegant gradient background, {options.style} style",
            style=options.style
        ))

    return slides[:options.max_slides]


# =============================================================================
# Slide Image Generation
# =============================================================================

async def generate_slide_image(
    client: ImageGenerationClient,
    slide: SlideContent,
    brand_colors: Optional[Dict[str, str]] = None
) -> Optional[str]:
    """Generate image for a single slide, returns base64 data"""

    color_instructions = ""
    if brand_colors:
        color_instructions = f"\nColor scheme: Primary {brand_colors.get('primary', '#6366f1')}, Secondary {brand_colors.get('secondary', '#8b5cf6')}"

    # Create detailed image prompt for presentation slide
    image_prompt = f"""Create a professional presentation slide background image.

SLIDE TITLE: {slide.title}
VISUAL CONCEPT: {slide.visual_prompt}

STYLE: {slide.style}
{color_instructions}

REQUIREMENTS:
- 16:9 aspect ratio suitable for presentations
- Clean, professional appearance
- Leave space for text overlay (don't include text in image)
- Subtle, non-distracting background that enhances readability
- High quality, modern design aesthetic

Create an abstract or illustrative background that complements the topic without including any text."""

    result = await client.generate_image(image_prompt, size="1792x1024", quality="standard")

    if result:
        return result.get("b64_json") or result.get("url")

    return None


async def generate_all_slide_images(
    client: ImageGenerationClient,
    slides: List[SlideContent],
    brand_colors: Optional[Dict[str, str]] = None,
    progress_callback: Optional[callable] = None
) -> List[GeneratedSlide]:
    """Generate images for all slides"""

    generated_slides = []

    for i, slide in enumerate(slides):
        print(f"Generating slide {i+1}/{len(slides)}: {slide.title}")

        if progress_callback:
            progress_callback(i + 1, len(slides), f"Generating: {slide.title}")

        image_data = await generate_slide_image(client, slide, brand_colors)

        generated_slides.append(GeneratedSlide(
            slide_number=i + 1,
            title=slide.title,
            content=slide.content,
            visual_prompt=slide.visual_prompt,
            image_data=image_data
        ))

        # Rate limiting - don't hammer the API
        import asyncio
        await asyncio.sleep(2)

    return generated_slides


# =============================================================================
# Main Generation Function
# =============================================================================

async def generate_slideshow_content(
    api_key: str,
    content: str,
    options: Optional[SlideGenerationOptions] = None,
    gateway_url: str = "https://ai-gateway.vercel.sh/v1",
    generate_images: bool = True,
    zai_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main function to generate slideshow content from text.

    Args:
        api_key: Vercel AI Gateway API key
        content: Blog post or article content
        options: Slide generation options
        gateway_url: AI Gateway URL
        generate_images: Whether to generate actual images (set False for text-only)
        zai_api_key: Z.AI API key for GLM-Image generation

    Returns:
        Dictionary with slides data and metadata
    """

    if options is None:
        options = SlideGenerationOptions()

    client = ImageGenerationClient(
        api_key=api_key,
        zai_api_key=zai_api_key,
        gateway_url=gateway_url
    )

    # Step 1: Extract slides from content
    print("📝 Extracting slides from content...")
    slides = await extract_slides_from_content(client, content, options)
    print(f"   Extracted {len(slides)} slides")

    # Step 2: Generate images for each slide
    if generate_images:
        print("🎨 Generating slide images...")
        generated_slides = await generate_all_slide_images(
            client,
            slides,
            options.brand_colors
        )
    else:
        # Text-only slides (no images)
        generated_slides = [
            GeneratedSlide(
                slide_number=i + 1,
                title=slide.title,
                content=slide.content,
                visual_prompt=slide.visual_prompt
            )
            for i, slide in enumerate(slides)
        ]

    return {
        "success": True,
        "slide_count": len(generated_slides),
        "style": options.style,
        "slides": [slide.model_dump() for slide in generated_slides]
    }
