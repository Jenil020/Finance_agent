"""
RAG query tool — used by the Research Agent.

Wraps the full hybrid retrieval pipeline:
  1. Load index from existing Qdrant collection
  2. Run hybrid BM25 + Vector retrieval (QueryFusionRetriever)
  3. Apply CRAG confidence check
  4. Return formatted context OR a low-confidence signal

The low-confidence signal tells the Research Agent to also run
web search — it doesn't abort, it supplements.
"""
from typing import TypedDict
from llama_index.core.schema import NodeWithScore
from app.rag.vector_store import get_vector_store, get_collection_stats
from app.rag.embeddings import get_embed_model
from app.rag.retriever import build_index, get_hybrid_retriever, crag_retrieve
from app.core.config import settings
from app.core.logging import logger


class RAGResult(TypedDict):
    context: str          # formatted text chunks for the LLM
    sources: list[str]    # source file names for citation
    confidence: float     # CRAG score (0.0 – 1.0)
    needs_web_search: bool  # True when confidence < threshold


def _format_nodes(nodes: list[NodeWithScore]) -> tuple[str, list[str]]:
    """
    Format retrieved nodes into (context_text, source_list).
    Includes metadata like source filename and page number when available.
    """
    parts = []
    sources = []

    for i, node in enumerate(nodes, 1):
        meta = node.metadata or {}
        source = meta.get("file_name", meta.get("source", "unknown"))
        page = meta.get("page_label", meta.get("page_number", ""))
        page_str = f" (p.{page})" if page else ""

        parts.append(
            f"[Source {i}: {source}{page_str}]\n"
            f"{node.get_content()}"
        )

        if source and source not in sources:
            sources.append(source)

    return "\n\n---\n\n".join(parts), sources


async def rag_query_tool(query: str) -> RAGResult:
    """
    Query the knowledge base with hybrid search + CRAG self-correction.

    Returns a RAGResult dict containing context text, sources,
    confidence score, and a flag indicating whether web search
    should supplement this answer.
    """
    logger.info(f"[RAGQuery] '{query[:80]}'")

    # Guard: if collection is empty, skip retrieval entirely
    try:
        stats = get_collection_stats()
        if not stats.get("vectors_count"):
            logger.warning("[RAGQuery] Collection is empty — skipping retrieval")
            return RAGResult(
                context="[No documents in knowledge base yet]",
                sources=[],
                confidence=0.0,
                needs_web_search=True,
            )
    except Exception as e:
        logger.error(f"[RAGQuery] Could not check collection stats: {e}")

    try:
        vector_store = get_vector_store()
        embed_model = get_embed_model()

        # Build index view over existing Qdrant data (no re-embedding)
        index = build_index(vector_store, embed_model)
        retriever = get_hybrid_retriever(index)

        # CRAG retrieve
        nodes, confidence = await crag_retrieve(query, retriever)

        needs_web = not nodes or confidence < settings.crag_confidence_threshold

        if not nodes:
            return RAGResult(
                context="[No relevant documents found in knowledge base]",
                sources=[],
                confidence=0.0,
                needs_web_search=True,
            )

        context, sources = _format_nodes(nodes)

        logger.info(
            f"[RAGQuery] Returning {len(nodes)} chunks | "
            f"confidence={confidence:.3f} | needs_web={needs_web}"
        )

        return RAGResult(
            context=context,
            sources=sources,
            confidence=round(confidence, 4),
            needs_web_search=needs_web,
        )

    except Exception as e:
        logger.error(f"[RAGQuery] Retrieval failed: {e}", exc_info=True)
        return RAGResult(
            context=f"[RAG retrieval error: {e}]",
            sources=[],
            confidence=0.0,
            needs_web_search=True,
        )