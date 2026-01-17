"""
Slideshow Generator Module

Generates self-contained HTML slideshows from generated slides.
Includes autoplay, transitions, keyboard controls, and responsive design.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import html


# =============================================================================
# Models
# =============================================================================

class SlideshowOptions(BaseModel):
    """Options for HTML slideshow generation"""
    autoplay: bool = True
    duration: int = Field(default=5, ge=1, le=30)  # seconds per slide
    transition: str = Field(default="fade")  # fade, slide, zoom
    show_controls: bool = True
    show_progress: bool = True
    loop: bool = True
    theme: str = Field(default="dark")  # dark, light


class GeneratedSlide(BaseModel):
    """A slide with content and optional image"""
    slide_number: int
    title: str
    content: str
    visual_prompt: str = ""
    image_data: Optional[str] = None  # base64 encoded image
    image_url: Optional[str] = None


# =============================================================================
# HTML Slideshow Generator
# =============================================================================

def generate_html_slideshow(
    slides: List[Dict[str, Any]],
    options: Optional[SlideshowOptions] = None,
    title: str = "Presentation"
) -> str:
    """
    Generate a self-contained HTML slideshow.

    Args:
        slides: List of slide dictionaries with title, content, image_data
        options: Slideshow display options
        title: Presentation title for HTML head

    Returns:
        Complete HTML string
    """

    if options is None:
        options = SlideshowOptions()

    # Theme colors
    if options.theme == "dark":
        bg_color = "#0f172a"
        text_color = "#f8fafc"
        accent_color = "#6366f1"
        overlay_bg = "rgba(15, 23, 42, 0.85)"
    else:
        bg_color = "#ffffff"
        text_color = "#1e293b"
        accent_color = "#6366f1"
        overlay_bg = "rgba(255, 255, 255, 0.9)"

    # Generate slide HTML
    slides_html = ""
    for i, slide in enumerate(slides):
        slide_title = html.escape(slide.get("title", f"Slide {i+1}"))
        slide_content = html.escape(slide.get("content", ""))

        # Background image or gradient
        if slide.get("image_data"):
            bg_style = f"background-image: url('data:image/png;base64,{slide['image_data']}'); background-size: cover; background-position: center;"
        elif slide.get("image_url"):
            bg_style = f"background-image: url('{slide['image_url']}'); background-size: cover; background-position: center;"
        else:
            # Gradient fallback
            bg_style = f"background: linear-gradient(135deg, {accent_color}22 0%, {accent_color}44 100%);"

        active_class = "active" if i == 0 else ""

        slides_html += f'''
    <div class="slide {active_class}" data-slide="{i}" style="{bg_style}">
      <div class="slide-overlay">
        <div class="slide-content">
          <h2 class="slide-title">{slide_title}</h2>
          <p class="slide-text">{slide_content}</p>
        </div>
      </div>
    </div>
'''

    # Controls HTML
    controls_html = ""
    if options.show_controls:
        controls_html = '''
    <div class="controls">
      <button onclick="prevSlide()" aria-label="Previous slide">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="15 18 9 12 15 6"></polyline>
        </svg>
      </button>
      <button onclick="togglePlay()" id="playBtn" aria-label="Play/Pause">
        <svg id="playIcon" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="6" y="4" width="4" height="16"></rect>
          <rect x="14" y="4" width="4" height="16"></rect>
        </svg>
        <svg id="pauseIcon" style="display:none" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="5 3 19 12 5 21 5 3"></polygon>
        </svg>
      </button>
      <button onclick="nextSlide()" aria-label="Next slide">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="9 18 15 12 9 6"></polyline>
        </svg>
      </button>
    </div>
'''

    # Progress bar HTML
    progress_html = ""
    if options.show_progress:
        progress_html = '<div class="progress-bar" id="progress"></div>'

    # Generate complete HTML
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(title)}</title>
  <style>
    * {{
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
      background: {bg_color};
      color: {text_color};
      overflow: hidden;
      -webkit-font-smoothing: antialiased;
    }}

    .slideshow-container {{
      width: 100vw;
      height: 100vh;
      position: relative;
    }}

    .slide {{
      width: 100%;
      height: 100%;
      position: absolute;
      top: 0;
      left: 0;
      opacity: 0;
      visibility: hidden;
      transition: opacity 0.8s ease-in-out, visibility 0.8s ease-in-out;
    }}

    .slide.active {{
      opacity: 1;
      visibility: visible;
      z-index: 1;
    }}

    .slide-overlay {{
      position: absolute;
      inset: 0;
      background: {overlay_bg};
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 5%;
    }}

    .slide-content {{
      max-width: 900px;
      text-align: center;
      animation: fadeInUp 0.6s ease-out;
    }}

    @keyframes fadeInUp {{
      from {{
        opacity: 0;
        transform: translateY(30px);
      }}
      to {{
        opacity: 1;
        transform: translateY(0);
      }}
    }}

    .slide-title {{
      font-size: clamp(2rem, 5vw, 4rem);
      font-weight: 700;
      margin-bottom: 1.5rem;
      line-height: 1.2;
      background: linear-gradient(135deg, {text_color} 0%, {accent_color} 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }}

    .slide-text {{
      font-size: clamp(1rem, 2.5vw, 1.5rem);
      line-height: 1.8;
      opacity: 0.9;
      max-width: 700px;
      margin: 0 auto;
    }}

    .controls {{
      position: fixed;
      bottom: 40px;
      left: 50%;
      transform: translateX(-50%);
      z-index: 100;
      display: flex;
      gap: 16px;
      background: {overlay_bg};
      padding: 12px 24px;
      border-radius: 50px;
      backdrop-filter: blur(10px);
      border: 1px solid {text_color}22;
    }}

    .controls button {{
      background: transparent;
      border: none;
      color: {text_color};
      padding: 12px;
      border-radius: 50%;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      justify-content: center;
    }}

    .controls button:hover {{
      background: {accent_color}33;
      transform: scale(1.1);
    }}

    .controls button:active {{
      transform: scale(0.95);
    }}

    .progress-bar {{
      position: fixed;
      top: 0;
      left: 0;
      height: 4px;
      background: linear-gradient(90deg, {accent_color} 0%, #8b5cf6 100%);
      transition: width 0.3s ease;
      z-index: 100;
    }}

    .slide-counter {{
      position: fixed;
      bottom: 40px;
      right: 40px;
      z-index: 100;
      font-size: 14px;
      opacity: 0.7;
      font-variant-numeric: tabular-nums;
    }}

    /* Transition effects */
    .transition-slide .slide {{
      transition: transform 0.6s ease-in-out, opacity 0.6s ease-in-out;
      transform: translateX(100%);
    }}
    .transition-slide .slide.active {{
      transform: translateX(0);
    }}
    .transition-slide .slide.prev {{
      transform: translateX(-100%);
    }}

    .transition-zoom .slide {{
      transition: transform 0.6s ease-in-out, opacity 0.6s ease-in-out;
      transform: scale(0.8);
    }}
    .transition-zoom .slide.active {{
      transform: scale(1);
    }}

    /* Responsive */
    @media (max-width: 768px) {{
      .slide-overlay {{
        padding: 10%;
      }}
      .controls {{
        bottom: 20px;
        padding: 8px 16px;
        gap: 8px;
      }}
      .controls button {{
        padding: 8px;
      }}
    }}
  </style>
</head>
<body>
  <div class="slideshow-container transition-{options.transition}">
{slides_html}
  </div>

  {progress_html}

  <div class="slide-counter">
    <span id="currentSlide">1</span> / <span id="totalSlides">{len(slides)}</span>
  </div>

  {controls_html}

  <script>
    const slides = document.querySelectorAll('.slide');
    const totalSlides = slides.length;
    let currentSlide = 0;
    let isPlaying = {'true' if options.autoplay else 'false'};
    let interval;
    const duration = {options.duration * 1000};
    const shouldLoop = {'true' if options.loop else 'false'};

    function showSlide(n) {{
      slides.forEach((slide, i) => {{
        slide.classList.remove('active', 'prev');
        if (i === currentSlide) slide.classList.add('prev');
      }});

      if (shouldLoop) {{
        currentSlide = (n + totalSlides) % totalSlides;
      }} else {{
        currentSlide = Math.max(0, Math.min(n, totalSlides - 1));
      }}

      slides[currentSlide].classList.add('active');
      updateUI();
    }}

    function nextSlide() {{
      showSlide(currentSlide + 1);
    }}

    function prevSlide() {{
      showSlide(currentSlide - 1);
    }}

    function togglePlay() {{
      isPlaying = !isPlaying;
      updatePlayButton();

      if (isPlaying) {{
        startAutoplay();
      }} else {{
        stopAutoplay();
      }}
    }}

    function startAutoplay() {{
      stopAutoplay();
      interval = setInterval(nextSlide, duration);
    }}

    function stopAutoplay() {{
      if (interval) {{
        clearInterval(interval);
        interval = null;
      }}
    }}

    function updateUI() {{
      // Update progress bar
      const progress = document.getElementById('progress');
      if (progress) {{
        const percent = ((currentSlide + 1) / totalSlides) * 100;
        progress.style.width = percent + '%';
      }}

      // Update counter
      const counter = document.getElementById('currentSlide');
      if (counter) {{
        counter.textContent = currentSlide + 1;
      }}
    }}

    function updatePlayButton() {{
      const playIcon = document.getElementById('playIcon');
      const pauseIcon = document.getElementById('pauseIcon');
      if (playIcon && pauseIcon) {{
        playIcon.style.display = isPlaying ? 'block' : 'none';
        pauseIcon.style.display = isPlaying ? 'none' : 'block';
      }}
    }}

    // Keyboard controls
    document.addEventListener('keydown', (e) => {{
      switch(e.key) {{
        case 'ArrowRight':
        case ' ':
          e.preventDefault();
          nextSlide();
          break;
        case 'ArrowLeft':
          e.preventDefault();
          prevSlide();
          break;
        case 'p':
        case 'P':
          togglePlay();
          break;
        case 'Escape':
          if (document.fullscreenElement) {{
            document.exitFullscreen();
          }}
          break;
        case 'f':
        case 'F':
          if (!document.fullscreenElement) {{
            document.documentElement.requestFullscreen();
          }} else {{
            document.exitFullscreen();
          }}
          break;
      }}
    }});

    // Touch support
    let touchStartX = 0;
    let touchEndX = 0;

    document.addEventListener('touchstart', (e) => {{
      touchStartX = e.changedTouches[0].screenX;
    }});

    document.addEventListener('touchend', (e) => {{
      touchEndX = e.changedTouches[0].screenX;
      handleSwipe();
    }});

    function handleSwipe() {{
      const threshold = 50;
      const diff = touchStartX - touchEndX;
      if (Math.abs(diff) > threshold) {{
        if (diff > 0) {{
          nextSlide();
        }} else {{
          prevSlide();
        }}
      }}
    }}

    // Initialize
    updateUI();
    updatePlayButton();
    if (isPlaying) {{
      startAutoplay();
    }}
  </script>
</body>
</html>'''

    return html_content


def generate_embed_code(slideshow_url: str, width: int = 800, height: int = 450) -> str:
    """Generate embed code for the slideshow"""
    return f'<iframe src="{slideshow_url}" width="{width}" height="{height}" frameborder="0" allowfullscreen></iframe>'
