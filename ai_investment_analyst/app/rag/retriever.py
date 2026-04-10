"""
Hybrid retriever: BM25 (keyword) + Vector (semantic) fused via Reciprocal Rank Fusion.
CRAG (Corrective RAG) self-correction layer on top.

Why hybrid?
  - BM25 catches exact ticker/term matches ("NVDA", "P/E ratio")
  - Vector catches semantic similarity ("AI chip leader", "semiconductor")
  - RRF fusion combines both without needing score normalisation

Why CRAG?
  - If retrieved docs score low → they're probably not relevant
  - We flag this so the Research Agent knows to fall back to web search
  - Prevents hallucinated answers from irrelevant context
"""
import asyncio
from typing import List, Tuple
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever
from app.core.config import settings
from app.core.logging import logger


def build_index(vector_store, embed_model) -> VectorStoreIndex:
    """
    Build a VectorStoreIndex from an existing Qdrant collection.
    No re-embedding happens here — we just attach the store.
    """
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
        storage_context=storage_context,
    )
    return index


def get_hybrid_retriever(index: VectorStoreIndex) -> VectorIndexRetriever:
    """
    Build a hybrid retriever combining BM25 + Vector search.

    Uses Reciprocal Rank Fusion (RRF) to merge ranked lists — no score
    calibration needed, robust to different score scales.
    
    Falls back to vector-only if BM25 cannot be initialized (e.g., loading
    from existing Qdrant without node docstore).
    """
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=settings.top_k,
    )

    # BM25 works on nodes already stored in the index's docstore
    # If nodes are not available (e.g., when loading from existing Qdrant),
    # we fall back to vector-only retrieval
    try:
        bm25_retriever = BM25Retriever.from_defaults(
            index=index,
            similarity_top_k=settings.top_k,
            skip_stemming=False,    # stemming improves recall for financial terms
            language="en",
        )

        # QueryFusionRetriever merges and re-ranks results
        # Note: num_queries=1 means it won't generate sub-queries (no LLM call)
        hybrid_retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            retriever_weights=[0.6, 0.4],   # slightly favour semantic over keyword
            similarity_top_k=settings.top_k,
            num_queries=1,          # no LLM-generated sub-queries (saves API calls)
            mode="reciprocal_rerank",
            use_async=True,
        )
        return hybrid_retriever
    except (ValueError, Exception) as e:
        logger.warning(
            f"[Retriever] BM25 initialization failed: {e}. "
            "Using vector-only retrieval. "
            "(This is expected when loading from existing Qdrant. "
            "BM25 will work after re-ingestion.)"
        )
        return vector_retriever


async def crag_retrieve(
    query: str,
    retriever: QueryFusionRetriever,
) -> Tuple[List[NodeWithScore], float]:
    """
    CRAG: Corrective Retrieval Augmented Generation.

    Steps:
      1. Retrieve top-k nodes
      2. Compute average relevance confidence
      3. Return nodes + confidence score

    Callers should check confidence against settings.crag_confidence_threshold.
    If confidence < threshold → supplement with web search (Research Agent handles this).

    Returns:
        (nodes, confidence)  where confidence ∈ [0.0, 1.0]
    """
    # Local-disk Qdrant cannot be opened twice for sync + async access.
    # Run sync retrieval in a worker thread to keep the async app responsive.
    nodes: List[NodeWithScore] = await asyncio.to_thread(retriever.retrieve, query)

    if not nodes:
        logger.warning(f"[CRAG] Zero nodes retrieved for: '{query[:60]}'")
        return [], 0.0

    # Average similarity score across top-k nodes as confidence proxy
    scores = [n.score for n in nodes if n.score is not None]
    confidence = sum(scores) / len(scores) if scores else 0.0

    logger.info(
        f"[CRAG] Retrieved {len(nodes)} nodes | "
        f"avg_score={confidence:.3f} | "
        f"threshold={settings.crag_confidence_threshold}"
    )

    if confidence < settings.crag_confidence_threshold:
        logger.warning(
            f"[CRAG] Low confidence ({confidence:.3f}) — "
            "Research Agent will supplement with web search"
        )

    return nodes, confidence
