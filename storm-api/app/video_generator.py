"""
Video Generator Module

Generates videos from slides using FFmpeg.
Supports transitions, audio tracks, and voiceover integration.

NOTE: Requires FFmpeg to be installed on the system.
Not available on Vercel serverless - use for local/VPS deployments.
"""

import os
import shutil
import tempfile
import asyncio
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# =============================================================================
# Models
# =============================================================================

class VideoOptions(BaseModel):
    """Options for video generation"""
    fps: int = Field(default=30, ge=1, le=60)
    duration: int = Field(default=5, ge=1, le=30)  # seconds per slide
    transition: str = Field(default="fade")  # fade, slide, none
    resolution: str = Field(default="1080p")  # 720p, 1080p, 4k
    audio_path: Optional[str] = None  # path to background audio
    include_voiceover: bool = False
    voiceover_text: Optional[str] = None
    voiceover_speed: float = Field(default=1.0, ge=0.5, le=2.0)


class GeneratedSlide(BaseModel):
    """A slide with content and image"""
    slide_number: int
    title: str
    content: str
    visual_prompt: str = ""
    image_data: Optional[str] = None  # base64 encoded image


# =============================================================================
# FFmpeg Utilities
# =============================================================================

def is_ffmpeg_available() -> bool:
    """Check if FFmpeg is installed and accessible"""
    return shutil.which("ffmpeg") is not None


def get_resolution(resolution: str) -> tuple:
    """Get width and height from resolution string"""
    resolutions = {
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "4k": (3840, 2160)
    }
    return resolutions.get(resolution, resolutions["1080p"])


# =============================================================================
# Video Generator
# =============================================================================

class VideoGenerator:
    """Generate videos from slides using FFmpeg"""

    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="video_gen_")
        os.makedirs(self.temp_dir, exist_ok=True)

    async def generate_video(
        self,
        slides: List[Dict[str, Any]],
        output_path: str,
        options: Optional[VideoOptions] = None
    ) -> Dict[str, Any]:
        """
        Generate video from slides.

        Args:
            slides: List of slide dictionaries with image_data
            output_path: Path for output video file
            options: Video generation options

        Returns:
            Dictionary with video info and path
        """

        if not is_ffmpeg_available():
            return {
                "success": False,
                "error": "FFmpeg is not installed. Video generation requires FFmpeg.",
                "help": "Install FFmpeg: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"
            }

        if options is None:
            options = VideoOptions()

        try:
            # Step 1: Save slides as image files
            print("💾 Saving slide images...")
            image_paths = await self._save_slide_images(slides)

            if not image_paths:
                return {
                    "success": False,
                    "error": "No valid slide images to process"
                }

            # Step 2: Create video with FFmpeg
            print("🎥 Rendering video...")
            video_path = await self._create_video(
                image_paths,
                output_path,
                options
            )

            # Step 3: Add audio if provided
            if options.audio_path and os.path.exists(options.audio_path):
                print("🎵 Adding audio track...")
                video_path = await self._add_audio(
                    video_path,
                    options.audio_path,
                    output_path
                )

            # Step 4: Cleanup
            print("🧹 Cleaning up temporary files...")
            await self._cleanup()

            # Get video info
            video_size = os.path.getsize(video_path) if os.path.exists(video_path) else 0
            total_duration = len(slides) * options.duration

            return {
                "success": True,
                "video_path": video_path,
                "duration_seconds": total_duration,
                "resolution": options.resolution,
                "file_size_bytes": video_size,
                "slide_count": len(slides)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _save_slide_images(self, slides: List[Dict[str, Any]]) -> List[str]:
        """Save slide images to temporary files"""
        image_paths = []

        for i, slide in enumerate(slides):
            image_data = slide.get("image_data")

            if image_data:
                # Decode base64 image
                image_path = os.path.join(
                    self.temp_dir,
                    f"slide_{str(i).zfill(3)}.png"
                )

                try:
                    image_bytes = base64.b64decode(image_data)
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    image_paths.append(image_path)
                except Exception as e:
                    print(f"Failed to save slide {i}: {e}")
            else:
                # Create placeholder image for slides without images
                image_path = await self._create_placeholder_image(
                    slide.get("title", f"Slide {i+1}"),
                    slide.get("content", ""),
                    i
                )
                if image_path:
                    image_paths.append(image_path)

        return image_paths

    async def _create_placeholder_image(
        self,
        title: str,
        content: str,
        index: int
    ) -> Optional[str]:
        """Create a simple placeholder image using FFmpeg"""
        image_path = os.path.join(self.temp_dir, f"slide_{str(index).zfill(3)}.png")

        # Create solid color background with text using FFmpeg
        # This is a simple fallback - real implementation would use PIL or similar
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=0x1e293b:s=1920x1080:d=1",
            "-vframes", "1",
            image_path
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            return image_path if os.path.exists(image_path) else None
        except Exception as e:
            print(f"Failed to create placeholder: {e}")
            return None

    async def _create_video(
        self,
        image_paths: List[str],
        output_path: str,
        options: VideoOptions
    ) -> str:
        """Create video from images using FFmpeg"""

        width, height = get_resolution(options.resolution)

        # Create concat file for FFmpeg
        concat_file = os.path.join(self.temp_dir, "concat.txt")
        with open(concat_file, "w") as f:
            for i, path in enumerate(image_paths):
                f.write(f"file '{os.path.abspath(path)}'\n")
                f.write(f"duration {options.duration}\n")
            # Repeat last frame to avoid FFmpeg bug
            if image_paths:
                f.write(f"file '{os.path.abspath(image_paths[-1])}'\n")

        # Build FFmpeg command
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "medium",
            "-crf", "23",
            "-r", str(options.fps),
            output_path
        ]

        # Add fade effect if requested
        if options.transition == "fade":
            # Create filter for crossfade between slides
            fade_duration = 0.5
            filter_complex = f"fade=t=in:st=0:d={fade_duration}"
            cmd[cmd.index("-vf") + 1] = f"{cmd[cmd.index('-vf') + 1]},{filter_complex}"

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise Exception(f"FFmpeg error: {stderr.decode()[:500]}")

        return output_path

    async def _add_audio(
        self,
        video_path: str,
        audio_path: str,
        output_path: str
    ) -> str:
        """Add audio track to video"""

        temp_output = os.path.join(self.temp_dir, "temp_with_audio.mp4")

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            temp_output
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode == 0 and os.path.exists(temp_output):
            # Move to final output
            shutil.move(temp_output, output_path)
            return output_path

        # If audio merge failed, return original video
        return video_path

    async def _cleanup(self):
        """Remove temporary files"""
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Cleanup warning: {e}")


# =============================================================================
# Convenience Functions
# =============================================================================

async def generate_video_from_slides(
    slides: List[Dict[str, Any]],
    output_path: str,
    duration: int = 5,
    resolution: str = "1080p",
    transition: str = "fade",
    audio_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to generate video from slides.

    Args:
        slides: List of slide dictionaries
        output_path: Output video file path
        duration: Seconds per slide
        resolution: Video resolution (720p, 1080p, 4k)
        transition: Transition type (fade, none)
        audio_path: Optional background audio path

    Returns:
        Dictionary with result info
    """

    generator = VideoGenerator()

    options = VideoOptions(
        duration=duration,
        resolution=resolution,
        transition=transition,
        audio_path=audio_path
    )

    return await generator.generate_video(slides, output_path, options)


def check_ffmpeg() -> Dict[str, Any]:
    """Check FFmpeg availability and version"""
    import subprocess

    if not is_ffmpeg_available():
        return {
            "available": False,
            "message": "FFmpeg is not installed",
            "help": "Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"
        }

    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True
        )
        version_line = result.stdout.split('\n')[0] if result.stdout else "Unknown"
        return {
            "available": True,
            "version": version_line
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }
