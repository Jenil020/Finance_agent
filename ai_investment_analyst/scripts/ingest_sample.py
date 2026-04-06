"""
Quick script to ingest sample documents into Qdrant.
Usage: python scripts/ingest_sample.py
"""
import asyncio
from app.rag.ingestion import ingest_documents


async def main():
    sample_files = [
        "./data/sample_docs/annual_report.pdf",
        # Add more files here
    ]
    result = await ingest_documents(sample_files, metadata={"source": "manual"})
    print(f"Ingested {result['count']} chunks from {result['files']}")


if __name__ == "__main__":
    asyncio.run(main())
