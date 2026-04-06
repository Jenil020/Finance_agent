"""
Base utilities shared by all agents.

Provides:
  - GeminiClient singleton (google.genai SDK — not deprecated)
  - llm_call()        — single async LLM call with Tenacity retry
  - llm_call_stream() — async generator for token streaming
  - Each agent gets its own system prompt injected via GenerateContentConfig
"""
import google.genai as genai
from google.genai import types as genai_types
from typing import AsyncIterator
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from app.core.config import settings
from app.core.logging import logger

# ── Retryable exceptions from google.genai ────────────────────────────────────
# 429 = quota exceeded, 503 = service unavailable  — both are transient
_RETRYABLE = (Exception,)   # broad catch; filter below in retry condition


def _is_retryable(exc: Exception) -> bool:
    """Return True for transient API errors worth retrying."""
    msg = str(exc).lower()
    return any(k in msg for k in ["429", "503", "quota", "resource exhausted", "timeout"])


# ── Singleton client ──────────────────────────────────────────────────────────

_client: genai.Client | None = None


def get_client() -> genai.Client:
    """Return the singleton google.genai client (lazy-initialised)."""
    global _client
    if _client is None:
        if not settings.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. "
                "Get a free key at https://aistudio.google.com/app/apikey"
            )
        _client = genai.Client(api_key=settings.google_api_key)
        logger.info(f"[LLM] Gemini client initialised (model={settings.gemini_model})")
    return _client


# ── Core LLM call with retry ─────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def llm_call(
    prompt: str,
    system_instruction: str | None = None,
    temperature: float = 0.2,
    max_output_tokens: int = 2048,
    json_mode: bool = False,
) -> str:
    """
    Single async LLM call to Gemini with Tenacity retry.

    Args:
        prompt:             User/task content sent to the model.
        system_instruction: Optional system-level persona/instructions.
        temperature:        0.0 = deterministic, 1.0 = creative.
                            Agents default low (0.2) for factual consistency.
        max_output_tokens:  Hard cap on response length.
        json_mode:          If True, sets response_mime_type="application/json"
                            so the model is constrained to return valid JSON.

    Returns:
        Response text string. Raises on hard API errors after retries.
    """
    client = get_client()

    config = genai_types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json" if json_mode else "text/plain",
    )

    try:
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=config,
        )
    except Exception as e:
        if _is_retryable(e):
            logger.warning(f"[LLM] Transient error, will retry: {e}")
            raise   # tenacity catches and retries
        logger.error(f"[LLM] Non-retryable error: {e}")
        raise

    text = response.text
    if text is None:
        # Finish reason other than STOP (e.g. SAFETY) — treat as empty
        finish = (
            response.candidates[0].finish_reason
            if response.candidates else "unknown"
        )
        logger.warning(f"[LLM] Empty response text (finish_reason={finish})")
        return ""

    return text


# ── Streaming LLM call ───────────────────────────────────────────────────────

async def llm_call_stream(
    prompt: str,
    system_instruction: str | None = None,
    temperature: float = 0.2,
    max_output_tokens: int = 2048,
) -> AsyncIterator[str]:
    """
    Async generator that yields text tokens as they stream from Gemini.
    Used by the Report Agent to stream the final answer to the SSE endpoint.

    Usage:
        async for token in llm_call_stream(prompt):
            yield f"data: {token}\\n\\n"
    """
    client = get_client()

    config = genai_types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    async for chunk in await client.aio.models.generate_content_stream(
        model=settings.gemini_model,
        contents=prompt,
        config=config,
    ):
        if chunk.text:
            yield chunk.text