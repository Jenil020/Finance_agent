"""
Custom Gemini embedding model using the new google-genai SDK.

We implement LlamaIndex's BaseEmbedding directly instead of using
llama-index-embeddings-gemini, which still depends on the deprecated
google-generativeai package.

Free model: gemini-embedding-001 → 768-dimensional vectors.

Rate limiting strategy (fixes 'Server disconnected' error):
─────────────────────────────────────────────────────────────
Gemini free tier: 1500 RPM (25 RPS) for embeddings.
Root cause of the crash: LlamaIndex fires 1066 sync HTTP calls with
NO delay → overwhelms httpx connection pool → server drops connection.

Fix: SimpleRateLimiter enforces a minimum gap between API calls.
We also override _get_text_embeddings with tenacity retry on connection
errors (httpx.RemoteProtocolError), which handles transient drops.

Settings (conservative, well under free tier):
  EMBED_REQUESTS_PER_SECOND = 8   →  8 RPS  (limit is 25 RPS)
  embed_batch_size           = 5  →  5 texts per API call
"""
import time
import asyncio
from typing import List

import httpx
import google.genai as genai
from google.genai import types as genai_types
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import Field

from app.core.config import settings
from app.core.logging import logger


# ── Rate limiter ──────────────────────────────────────────────────────────────

class SimpleRateLimiter:
    """
    Token-bucket style rate limiter with a minimum inter-call interval.

    LlamaIndex calls self.rate_limiter.acquire() before each batch flush
    (see BaseEmbedding.get_text_embedding_batch). It may also call
    async_acquire() when using async embedding flows.
    """

    def __init__(self, requests_per_second: float = 8.0):
        self._min_interval = 1.0 / requests_per_second
        self._last_call: float = 0.0

    def acquire(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_call
        wait = self._min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    async def async_acquire(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_call
        wait = self._min_interval - elapsed
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_call = time.monotonic()


# ── Retry decorator for connection errors ─────────────────────────────────────

_EMBED_RETRY = retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((
        httpx.RemoteProtocolError,
        httpx.ConnectError,
        httpx.TimeoutException,
        ConnectionError,
    )),
    reraise=True,
)


# ── Embedding model ───────────────────────────────────────────────────────────

class GeminiEmbedding(BaseEmbedding):
    """
    LlamaIndex-compatible Gemini embedding model.

    Fixes vs original:
      1. SimpleRateLimiter wired into LlamaIndex's rate_limiter hook
         → enforces gap between batch calls at the LlamaIndex level.
      2. _get_text_embeddings overridden with per-chunk delay + retry
         → handles transient 'Server disconnected' errors gracefully.
      3. Async batch uses asyncio.Semaphore to cap concurrency
         → prevents overwhelming the connection pool.
    """

    model_name: str = Field(default="gemini-embedding-001")
    api_key: str = Field(default="")
    output_dimensionality: int = Field(default=3072)

    # Non-pydantic private attribute for the client
    _client: genai.Client = None  # type: ignore[assignment]

    def _get_client(self) -> genai.Client:
        if self._client is None:
            object.__setattr__(
                self, "_client",
                genai.Client(api_key=self.api_key or settings.google_api_key),
            )
        return self._client

    # ── Sync single-text embedding (with retry) ───────────────────────────────

    @_EMBED_RETRY
    def _embed_one(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
        """Single embed call with retry on connection errors."""
        client = self._get_client()
        response = client.models.embed_content(
            model=self.model_name,
            contents=text,
            config=genai_types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.output_dimensionality,
            ),
        )
        return list(response.embeddings[0].values)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed_one(text, "RETRIEVAL_DOCUMENT")

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed_one(query, "RETRIEVAL_QUERY")

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Override the default list-comprehension with per-chunk throttling.

        LlamaIndex calls this for each batch (of size embed_batch_size).
        We add a small sleep between individual calls inside the batch
        for extra safety on top of the batch-level rate limiter.
        """
        results = []
        for i, text in enumerate(texts):
            if i > 0:
                time.sleep(0.12)   # 120ms = ~8 RPS within a batch
            results.append(self._embed_one(text, "RETRIEVAL_DOCUMENT"))
        return results

    # ── Async embedding ───────────────────────────────────────────────────────

    async def _aget_query_embedding(self, query: str) -> List[float]:
        client = self._get_client()
        response = await client.aio.models.embed_content(
            model=self.model_name,
            contents=query,
            config=genai_types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self.output_dimensionality,
            ),
        )
        return list(response.embeddings[0].values)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        client = self._get_client()
        response = await client.aio.models.embed_content(
            model=self.model_name,
            contents=text,
            config=genai_types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self.output_dimensionality,
            ),
        )
        return list(response.embeddings[0].values)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Async batch embedding with a semaphore to cap concurrency.
        Max 5 concurrent requests → stays well under connection pool limits.
        """
        sem = asyncio.Semaphore(5)

        async def _embed_with_sem(text: str) -> List[float]:
            async with sem:
                return await self._aget_text_embedding(text)

        results = await asyncio.gather(*[_embed_with_sem(t) for t in texts])
        logger.debug(f"[Embeddings] Async batch: {len(texts)} chunks embedded")
        return list(results)


# ── Module-level singleton ────────────────────────────────────────────────────

_embed_model: GeminiEmbedding | None = None


def get_embed_model() -> GeminiEmbedding:
    """Return the singleton embedding model instance."""
    global _embed_model
    if _embed_model is None:
        _embed_model = GeminiEmbedding(
            model_name=settings.gemini_embed_model,
            api_key=settings.google_api_key,
            output_dimensionality=settings.gemini_embed_output_dim,
            embed_batch_size=5,            # 5 texts per LlamaIndex batch call
            rate_limiter=SimpleRateLimiter(
                requests_per_second=8.0    # 8 RPS << free tier limit of 25 RPS
            ),
        )
        logger.info(
            f"[Embeddings] Model ready: {settings.gemini_embed_model} "
            f"| dim={settings.gemini_embed_output_dim} | batch_size=5 | rate=8 RPS"
        )
    return _embed_model
