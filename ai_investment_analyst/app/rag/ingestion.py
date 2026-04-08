"""
Document ingestion pipeline.

Flow:
  file paths → SimpleDirectoryReader (PDF/DOCX/CSV/TXT)
             → SentenceSplitter (semantic chunking)
             → GeminiEmbedding (gemini-embedding-001)
             → QdrantVectorStore (local disk)

No Celery — pure async Python (sufficient for a resume project;
production would use a task queue for large batches).

Supported file types: .pdf  .docx  .txt  .csv  .md
"""
from pathlib import Path
from typing import List, Dict, Any, Optional

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document

from app.rag.vector_store import get_vector_store, get_collection_stats
from app.rag.embeddings import get_embed_model
from app.core.config import settings
from app.core.logging import logger

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv", ".md"}


def _validate_files(file_paths: List[str]) -> List[str]:
    """Return only existing, supported files. Log anything skipped."""
    valid = []
    for fp in file_paths:
        p = Path(fp)
        if not p.exists():
            logger.warning(f"[Ingestion] File not found, skipping: {fp}")
        elif p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.warning(f"[Ingestion] Unsupported type '{p.suffix}', skipping: {fp}")
        else:
            valid.append(fp)
    return valid


def _load_documents(file_paths: List[str]) -> List[Document]:
    """Load documents using LlamaIndex SimpleDirectoryReader."""
    reader = SimpleDirectoryReader(
        input_files=file_paths,
        filename_as_id=True,   # use filename as doc_id for deduplication
    )
    docs = reader.load_data()
    logger.info(f"[Ingestion] Loaded {len(docs)} document(s) from {len(file_paths)} file(s)")
    return docs


def _attach_metadata(docs: List[Document], metadata: Dict[str, Any]) -> List[Document]:
    """Attach user-supplied metadata to every document (e.g. tenant, category)."""
    for doc in docs:
        doc.metadata.update(metadata)
    return docs


def _chunk_documents(docs: List[Document]) -> list:
    """
    Split documents into chunks using SentenceSplitter.

    SentenceSplitter respects sentence boundaries (unlike naive character
    splitting) — important for financial text where a number mid-sentence
    loses context if split.
    """
    splitter = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(docs, show_progress=False)
    logger.info(
        f"[Ingestion] Chunked into {len(nodes)} nodes "
        f"(chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap})"
    )
    return nodes


async def ingest_documents(
    file_paths: List[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Full ingestion pipeline: validate → load → chunk → embed → store.

    Args:
        file_paths: List of absolute or relative paths to documents.
        metadata:   Optional dict attached to every chunk (e.g. {"source": "10-K", "ticker": "AAPL"})

    Returns:
        {"count": int, "files": list, "collection_stats": dict}
    """
    logger.info(f"[Ingestion] Starting pipeline for {len(file_paths)} file(s)")

    # 1. Validate
    valid_paths = _validate_files(file_paths)
    if not valid_paths:
        logger.error("[Ingestion] No valid files to ingest")
        return {"count": 0, "files": [], "collection_stats": {}}

    # 2. Load
    docs = _load_documents(valid_paths)
    if not docs:
        return {"count": 0, "files": valid_paths, "collection_stats": {}}

    # 3. Attach metadata
    if metadata:
        docs = _attach_metadata(docs, metadata)

    # 4. Chunk
    nodes = _chunk_documents(docs)

    # 5. Embed + store via LlamaIndex VectorStoreIndex
    vector_store = get_vector_store()
    embed_model = get_embed_model()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    logger.info(f"[Ingestion] Embedding {len(nodes)} nodes with {settings.gemini_embed_model}...")
    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    # 6. Report
    stats = get_collection_stats()
    logger.info(
        f"[Ingestion] ✅ Done — {len(nodes)} chunks indexed | "
        f"Total vectors in collection: {stats.get('vectors_count', '?')}"
    )

    return {
        "count": len(nodes),
        "files": valid_paths,
        "collection_stats": stats,
    }


async def ingest_directory(
    directory: str,
    metadata: Optional[Dict[str, Any]] = None,
    recursive: bool = False,
) -> Dict[str, Any]:
    """
    Convenience wrapper: ingest all supported files from a directory.
    Used by scripts/ingest_sample.py.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    pattern = "**/*" if recursive else "*"
    file_paths = [
        str(p) for p in dir_path.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    logger.info(f"[Ingestion] Found {len(file_paths)} files in {directory}")
    return await ingest_documents(file_paths, metadata)