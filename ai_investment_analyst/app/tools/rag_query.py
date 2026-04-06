"""RAG query tool — used by Research Agent."""
from app.rag.vector_store import get_vector_store
from app.rag.embeddings import get_embed_model
from app.rag.retriever import get_hybrid_retriever, crag_retrieve
from llama_index.core import VectorStoreIndex
from app.core.config import settings
from app.core.logging import logger

WEB_SEARCH_THRESHOLD = 0.4  # Below this, fall back to web search


async def rag_query_tool(query: str) -> str:
    """
    Query the RAG knowledge base with CRAG self-correction.
    Falls back to web-search note if confidence is low.
    """
    logger.info(f"[RAG Query] {query[:80]}")
    try:
        vector_store = get_vector_store()
        embed_model = get_embed_model()
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )
        retriever = get_hybrid_retriever(index)
        nodes, confidence = await crag_retrieve(query, retriever)

        if not nodes or confidence < WEB_SEARCH_THRESHOLD:
            logger.warning(f"[CRAG] Low confidence ({confidence:.2f}), flagging for web search")
            return f"[Low RAG confidence: {confidence:.2f}] Recommend supplementing with web search."

        texts = [n.get_content() for n in nodes[:settings.top_k]]
        return "\n\n".join(texts)

    except Exception as e:
        logger.error(f"[RAG Query] Error: {e}")
        return f"RAG unavailable: {e}"
