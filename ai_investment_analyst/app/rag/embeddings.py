"""Google Gemini embedding model (free API)."""
from llama_index.embeddings.gemini import GeminiEmbedding
from app.core.config import settings

_embed_model = None


def get_embed_model() -> GeminiEmbedding:
    global _embed_model
    if _embed_model is None:
        _embed_model = GeminiEmbedding(
            model_name=settings.embed_model,
            api_key=settings.google_api_key,
        )
    return _embed_model
