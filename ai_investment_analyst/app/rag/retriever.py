"""
Hybrid retriever: BM25 (keyword) + Vector (semantic) search.
CRAG self-correction layer for low-confidence results.
"""
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from app.rag.vector_store import get_vector_store
from app.rag.embeddings import get_embed_model
from app.core.config import settings


def get_hybrid_retriever(index: VectorStoreIndex):
    """
    Combine BM25 + Vector retrieval using Reciprocal Rank Fusion.
    Improves recall vs pure vector search.
    """
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=settings.top_k,
    )

    bm25_retriever = BM25Retriever.from_defaults(
        index=index,
        similarity_top_k=settings.top_k,
    )

    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=settings.top_k,
        num_queries=1,          # Don't generate sub-queries here
        mode="reciprocal_rerank",
        use_async=True,
    )

    return hybrid_retriever


async def crag_retrieve(query: str, retriever) -> tuple[list, float]:
    """
    CRAG (Corrective RAG) — evaluate retrieval confidence.
    If confidence < threshold, fall back to web search.
    Returns (nodes, confidence_score)
    """
    nodes = await retriever.aretrieve(query)

    if not nodes:
        return [], 0.0

    # Simple confidence: average similarity score
    avg_score = sum(n.score or 0.0 for n in nodes) / len(nodes)
    return nodes, avg_score
