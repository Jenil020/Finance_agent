import asyncio

from app.core.logging import setup_logging
from app.orchestration.checkpointer import init_checkpointer, close_checkpointer
from app.orchestration.graph import run_agent_stream

setup_logging()

async def main():
    await init_checkpointer()
    try:
        async for token in run_agent_stream(
            query="Should i invest in NVIDIA ?",
            session_id="test-session-01",
            portfolio=[],
        ):
            print(token, end="", flush=True)
    finally:
        await close_checkpointer()

if __name__ == "__main__":
    asyncio.run(main())
