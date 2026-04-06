"""
Custom Gemini embedding model using the new google-genai SDK.

We implement LlamaIndex's BaseEmbedding directly instead of using
llama-index-embeddings-gemini, which still depends on the deprecated
google-generativeai package. This way we use the current google.genai
SDK cleanly with no deprecation warnings.

Free model: text-embedding-004 → 768-dimensional vectors.
"""
from typing import List
import asyncio
import google.genai as genai
from google.genai import types as genai_types
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import Field
from app.core.config import settings
from app.core.logging import logger


class GeminiEmbedding(BaseEmbedding):
    """
    LlamaIndex-compatible embedding model backed by google.genai.

    Implements the 3 required abstract methods:
      - _get_query_embedding   (sync, single)
      - _get_text_embedding    (sync, single)
      - _aget_query_embedding  (async, single)

    The base class handles batching and caching on top of these.
    """

    model_name: str = Field(default="text-embedding-004")
    api_key: str = Field(default="")

    # We keep the genai client out of pydantic fields (not serialisable)
    _client: genai.Client = None   # type: ignore[assignment]

    def _get_client(self) -> genai.Client:
        if self._client is None:
            object.__setattr__(
                self, "_client",
                genai.Client(api_key=self.api_key or settings.google_api_key),
            )
        return self._client

    # ── Core sync embedding ──────────────────────────────────────────────────

    def _embed(self, text: str) -> List[float]:
        """Call Gemini embed_content synchronously."""
        client = self._get_client()
        response = client.models.embed_content(
            model=self.model_name,
            contents=text,
            config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        return list(response.embeddings[0].values)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        client = self._get_client()
        response = client.models.embed_content(
            model=self.model_name,
            contents=query,
            config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return list(response.embeddings[0].values)

    # ── Async embedding ──────────────────────────────────────────────────────

    async def _aget_query_embedding(self, query: str) -> List[float]:
        client = self._get_client()
        response = await client.aio.models.embed_content(
            model=self.model_name,
            contents=query,
            config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return list(response.embeddings[0].values)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        client = self._get_client()
        response = await client.aio.models.embed_content(
            model=self.model_name,
            contents=text,
            config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        return list(response.embeddings[0].values)

    # ── Batch async (overrides base for efficiency) ──────────────────────────

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts concurrently."""
        tasks = [self._aget_text_embedding(t) for t in texts]
        results = await asyncio.gather(*tasks)
        logger.debug(f"[Embeddings] Batch embedded {len(texts)} chunks")
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
            embed_batch_size=10,   # stay under free-tier rate limits
        )
        logger.info(f"[Embeddings] Loaded model: {settings.gemini_embed_model}")
    return _embed_model