"""
Quick smoke test for the full agent pipeline.
Usage: python scripts/test_chat.py
"""
import asyncio
from app.orchestration.graph import run_agent_stream


async def main():
    print("Running test query through agent pipeline...\n")
    async for token in run_agent_stream(
        query="Should I invest in NVIDIA given current AI trends?",
        session_id="smoke-test-001",
    ):
        print(token, end="", flush=True)
    print("\n\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
