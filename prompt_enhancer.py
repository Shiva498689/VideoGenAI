"""
Groq-powered prompt enhancer.

enhance_image_prompt(raw)  → one detailed txt2img prompt for the source frame
enhance_prompt(raw)        → list of 5 sequential img2vid prompts for one scene

Both use llama-3.3-70b-versatile via Groq's OpenAI-compatible endpoint.
"""

import json
import httpx
from config import settings

# ── System prompts ────────────────────────────────────────────────────────────

_IMAGE_SYSTEM = """\
You are an expert image-generation prompt engineer specialising in cinematic \
keyframe stills. The image you help create will be used as the FIRST FRAME \
of an AI-generated video clip, so it must be a strong, well-composed still.

Given the user's rough scene description, produce ONE detailed txt2img prompt.
Include: subject & pose, environment, lighting mood, colour palette, camera \
angle, lens feel, art direction, and quality boosters (e.g. "sharp focus, \
8k, photorealistic, cinematic").

Rules:
- Respond with ONLY the prompt string — no explanation, no quotes, no markdown.
- Keep it under 200 words.
- Do NOT include negative prompts.
"""

_VIDEO_SYSTEM = """\
You are a cinematic AI video director and prompt engineer.
Your job is to take a user's rough scene description and expand it into \
exactly 5 detailed, SEQUENTIAL video prompts for an img2vid model \
(LTX-Video 2.3). Each prompt drives a 4-second clip; together they form a \
smooth 20-second scene.

Rules:
- Clip 1 starts from the generated keyframe image — describe its continuation.
- Each subsequent clip continues naturally from the end of the previous one.
- Include per clip: subject action, camera movement, lighting, atmosphere.
- Keep character/subject description IDENTICAL across all 5 prompts for consistency.
- Be specific and vivid; avoid vague words like "beautiful" or "amazing".
- Respond ONLY with a valid JSON array of exactly 5 strings.
  No preamble, no markdown fences, no trailing text.

Example:
["prompt1...", "prompt2...", "prompt3...", "prompt4...", "prompt5..."]
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _groq_headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }


async def _groq_chat(system: str, user: str, max_tokens: int = 1024) -> str:
    payload = {
        "model": settings.GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": 0.8,
        "max_tokens":  max_tokens,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=_groq_headers(),
            json=payload,
        )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers if Groq adds them."""
    if text.startswith("```"):
        lines = text.split("\n")
        # drop first line (```json or ```) and last line (```)
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return text.strip()


# ── Public API ────────────────────────────────────────────────────────────────

async def enhance_image_prompt(raw_prompt: str) -> str:
    """
    Returns one detailed txt2img prompt for the scene's source keyframe image.
    """
    result = await _groq_chat(
        system=_IMAGE_SYSTEM,
        user=raw_prompt,
        max_tokens=300,
    )
    return result.strip('"').strip("'")


async def enhance_prompt(raw_prompt: str) -> list[str]:
    """
    Returns a list of exactly 5 sequential img2vid prompts for one 20s scene.
    Raises ValueError if Groq doesn't return a valid 5-element JSON array.
    """
    raw = await _groq_chat(
        system=_VIDEO_SYSTEM,
        user=f"Scene description: {raw_prompt}",
        max_tokens=1200,
    )

    cleaned = _strip_fences(raw)

    try:
        prompts = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Groq returned invalid JSON: {e}\nRaw output:\n{raw}")

    if not isinstance(prompts, list) or len(prompts) != 5:
        raise ValueError(
            f"Expected JSON array of 5 prompts, got {type(prompts).__name__} "
            f"with {len(prompts) if isinstance(prompts, list) else '?'} items.\n"
            f"Raw output:\n{raw}"
        )

    return [str(p).strip() for p in prompts]
