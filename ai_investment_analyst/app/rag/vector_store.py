"""
Qdrant vector store — local disk mode (completely free, no cloud account).

Data persists at ./data/qdrant between restarts.
Collection is auto-created on first use with correct vector dimensions.
"""
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from llama_index.vector_stores.qdrant import QdrantVectorStore
from app.core.config import settings
from app.core.logging import logger

_client: QdrantClient | None = None
_vector_store: QdrantVectorStore | None = None


def get_qdrant_client() -> QdrantClient:
    """Return singleton local Qdrant client."""
    global _client
    if _client is None:
        _client = QdrantClient(path=settings.qdrant_path)
        logger.info(f"[VectorStore] Qdrant client opened at {settings.qdrant_path}")
        _ensure_collection(_client)
    return _client


def _ensure_collection(client: QdrantClient) -> None:
    """Create the collection if it doesn't exist yet."""
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=qdrant_models.VectorParams(
                size=settings.qdrant_vector_size,   # 768 for text-embedding-004
                distance=qdrant_models.Distance.COSINE,
            ),
        )
        logger.info(
            f"[VectorStore] Created collection '{settings.qdrant_collection}' "
            f"(dim={settings.qdrant_vector_size}, distance=COSINE)"
        )
    else:
        logger.debug(f"[VectorStore] Collection '{settings.qdrant_collection}' already exists")


def get_vector_store() -> QdrantVectorStore:
    """Return singleton LlamaIndex QdrantVectorStore wrapper."""
    global _vector_store
    if _vector_store is None:
        client = get_qdrant_client()
        _vector_store = QdrantVectorStore(
            collection_name=settings.qdrant_collection,
            client=client,
        )
        logger.info("[VectorStore] QdrantVectorStore ready")
    return _vector_store


def get_collection_stats() -> dict:
    """Return collection size — useful for health checks and the /ingest endpoint."""
    client = get_qdrant_client()
    info = client.get_collection(settings.qdrant_collection)
    return {
        "collection": settings.qdrant_collection,
        "vectors_count": info.vectors_count,
        "points_count": info.points_count,
        "status": str(info.status),
    }


def delete_collection() -> None:
    """Drop and recreate the collection (dev/testing only)."""
    client = get_qdrant_client()
    client.delete_collection(settings.qdrant_collection)
    global _vector_store
    _vector_store = None
    _ensure_collection(client)
    logger.warning(f"[VectorStore] Collection '{settings.qdrant_collection}' wiped and recreated")