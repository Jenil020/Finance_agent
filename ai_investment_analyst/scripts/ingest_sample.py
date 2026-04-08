"""
Quick script to ingest sample documents into Qdrant.
Usage: python scripts/ingest_sample.py
"""
import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from app.rag.ingestion import ingest_documents
except ModuleNotFoundError as exc:
    missing = exc.name
    print("\nERROR: missing module while importing the project:\n")
    print(f"  {missing}\n")
    if missing == "app":
        print("  The project root is not on Python path.\n"
              f"  This script already adds {PROJECT_ROOT} to sys.path.\n"
              "  Run it from the ai_investment_analyst directory or install the package.\n")
    else:
        print("  It looks like a required dependency is not installed.\n"
              "  Install dependencies from the project root with:\n"
              "      python setup.py\n"
              "  or\n"
              "      pip install -e .\n")
    raise


async def main():
    sample_files = [
        "ai_investment_analyst/data/sample_docs/annual_report.pdf",
        # Add more files here
    ]
    result = await ingest_documents(sample_files, metadata={"source": "manual"})
    print(f"Ingested {result['count']} chunks from {result['files']}")


if __name__ == "__main__":
    asyncio.run(main())
