"""
LLM Client for Vercel AI Gateway

Provides unified access to multiple LLM providers through Vercel AI Gateway's
OpenAI-compatible API endpoint.

Supported models (via Vercel AI Gateway):
- openai/gpt-5-nano, openai/gpt-4o, openai/gpt-4o-mini
- anthropic/claude-sonnet-4.5, anthropic/claude-4-opus
- xai/grok-4.1-fast-reasoning
- zai/glm-4.7
- And many more...
"""

import json
import re
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

# Load .env file from the parent directory (storm-api root)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Configuration from environment
VERCEL_AI_GATEWAY_URL = os.getenv("VERCEL_AI_GATEWAY_URL", "https://ai-gateway.vercel.sh/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openai/gpt-5-nano")


def get_llm_client() -> AsyncOpenAI:
    """Get OpenAI-compatible client configured for Vercel AI Gateway"""
    return AsyncOpenAI(
        base_url=VERCEL_AI_GATEWAY_URL,
        api_key=OPENAI_API_KEY,
    )


async def generate_text(
    prompt: str,
    model: str = None,
    system_prompt: str = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    """
    Generate text using Vercel AI Gateway.

    Args:
        prompt: The user prompt to send to the LLM
        model: Model to use (e.g., 'openai/gpt-5-nano', 'anthropic/claude-sonnet-4.5')
        system_prompt: Optional system prompt for context
        temperature: Creativity setting (0.0-1.0)
        max_tokens: Maximum tokens in response

    Returns:
        Generated text content

    Raises:
        Exception: If LLM generation fails
    """
    client = get_llm_client()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = await client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


async def generate_json(
    prompt: str,
    model: str = None,
    system_prompt: str = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> dict:
    """
    Generate JSON response using Vercel AI Gateway.

    Args:
        prompt: The user prompt (should request JSON output)
        model: Model to use
        system_prompt: Optional system prompt
        temperature: Lower for more consistent JSON
        max_tokens: Maximum tokens

    Returns:
        Parsed JSON as dict

    Raises:
        ValueError: If JSON parsing fails
    """
    response = await generate_text(
        prompt=prompt,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Try to extract JSON from response
    import logging
    logger = logging.getLogger(__name__)

    try:
        # First try direct parse
        return json.loads(response)
    except json.JSONDecodeError as e:
        logger.warning(f"Direct JSON parse failed: {e}")
        logger.debug(f"Response was: {response[:1000]}")

        # Try to find JSON in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError as e2:
                logger.warning(f"Markdown block JSON parse failed: {e2}")

        # Try to find JSON object/array pattern
        json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError as e3:
                logger.warning(f"Pattern match JSON parse failed: {e3}")

        # Log the full response for debugging
        logger.error(f"All JSON parsing attempts failed. Full response:\n{response}")
        raise ValueError(f"Failed to parse JSON from LLM response: {response[:500]}...")


async def generate_with_messages(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    """
    Generate text with custom message history.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model to use
        temperature: Creativity setting
        max_tokens: Maximum tokens

    Returns:
        Generated text content
    """
    client = get_llm_client()

    response = await client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


# Model recommendations for different tasks
RECOMMENDED_MODELS = {
    "content_generation": "openai/gpt-5-nano",  # Good for long-form content
    "analysis": "openai/gpt-5-nano",  # Good for structured analysis
    "reasoning": "xai/grok-4.1-fast-reasoning",  # Good for complex reasoning
    "fast": "openai/gpt-5-nano",  # Fast responses
    "quality": "anthropic/claude-sonnet-4.5",  # Highest quality
}


def get_recommended_model(task: str = "content_generation") -> str:
    """Get recommended model for a specific task"""
    return RECOMMENDED_MODELS.get(task, DEFAULT_MODEL)
