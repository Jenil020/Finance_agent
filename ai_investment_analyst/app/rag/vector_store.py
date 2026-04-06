"""Qdrant vector store — local disk mode (free, no cloud needed)."""
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from app.core.config import settings

_client = None
_vector_store = None


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        # Local disk persistence — free, no cloud account needed
        _client = QdrantClient(path=settings.qdrant_path)
    return _client


def get_vector_store() -> QdrantVectorStore:
    global _vector_store
    if _vector_store is None:
        client = get_qdrant_client()
        _vector_store = QdrantVectorStore(
            client=client,
            collection_name=settings.qdrant_collection,
        )
    return _vector_store
