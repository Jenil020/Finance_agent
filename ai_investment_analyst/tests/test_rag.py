"""Tests for RAG pipeline."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_rag_query_returns_string():
    from app.tools.rag_query import rag_query_tool
    with patch("app.tools.rag_query.get_vector_store"), \
         patch("app.tools.rag_query.get_embed_model"), \
         patch("app.tools.rag_query.VectorStoreIndex") as mock_index, \
         patch("app.tools.rag_query.get_hybrid_retriever") as mock_ret, \
         patch("app.tools.rag_query.crag_retrieve", new_callable=AsyncMock) as mock_crag:
        mock_node = MagicMock()
        mock_node.get_content.return_value = "AAPL has strong cash flow."
        mock_crag.return_value = ([mock_node], 0.85)
        result = await rag_query_tool("What is Apple's financial health?")
    assert isinstance(result, str)
    assert len(result) > 0
