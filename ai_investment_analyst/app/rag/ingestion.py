"""
Document ingestion pipeline.
Loads PDF/DOCX/CSV → chunks → embeds → stores in Qdrant.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from app.rag.vector_store import get_vector_store
from app.rag.embeddings import get_embed_model
from app.core.config import settings
from app.core.logging import logger


async def ingest_documents(
    file_paths: List[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Ingest documents into the RAG knowledge base.
    Supports: PDF, DOCX, CSV, TXT
    """
    logger.info(f"[Ingestion] Starting ingestion for {len(file_paths)} files")

    # Load documents
    documents = SimpleDirectoryReader(input_files=file_paths).load_data()

    # Attach metadata (tenant, category, etc.)
    if metadata:
        for doc in documents:
            doc.metadata.update(metadata)

    # Semantic chunking
    splitter = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(documents)

    # Index into Qdrant
    vector_store = get_vector_store()
    embed_model = get_embed_model()
    index = VectorStoreIndex(
        nodes,
        vector_store=vector_store,
        embed_model=embed_model,
    )

    logger.info(f"[Ingestion] Indexed {len(nodes)} chunks")
    return {"count": len(nodes), "files": file_paths}
