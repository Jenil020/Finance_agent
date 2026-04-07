"""
Async SQLite checkpointer for LangGraph state persistence.

Why async?
  SqliteSaver (sync) raises NotImplementedError when used with async graphs.
  AsyncSqliteSaver + aiosqlite is the correct pairing for FastAPI.

Why persistent connection (not context manager)?
  AsyncSqliteSaver.from_conn_string() is an asynccontextmanager — it closes
  the DB when the context exits. For a long-running FastAPI app we need the
  connection to stay open for the app's lifetime. We open it in the FastAPI
  lifespan startup hook (app/main.py) and close it on shutdown.

Usage:
    # In main.py lifespan:
    await init_checkpointer()   # startup
    await close_checkpointer()  # shutdown

    # In graph.py:
    from app.orchestration.checkpointer import get_checkpointer
    graph = workflow.compile(checkpointer=get_checkpointer())
"""
import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from app.core.config import settings
from app.core.logging import logger

_conn: aiosqlite.Connection | None = None
_checkpointer: AsyncSqliteSaver | None = None


async def init_checkpointer() -> None:
    """
    Open the aiosqlite connection and create the AsyncSqliteSaver.
    Must be called once during FastAPI app startup (lifespan).
    """
    global _conn, _checkpointer
    if _checkpointer is not None:
        return  # already initialised

    logger.info(f"[Checkpointer] Opening SQLite at {settings.sqlite_db_path}")
    _conn = await aiosqlite.connect(settings.sqlite_db_path)
    _checkpointer = AsyncSqliteSaver(_conn)

    # Create checkpoint tables if they don't exist
    await _checkpointer.setup()
    logger.info("[Checkpointer] AsyncSqliteSaver ready")


async def close_checkpointer() -> None:
    """
    Close the aiosqlite connection.
    Must be called during FastAPI app shutdown (lifespan).
    """
    global _conn, _checkpointer
    if _conn is not None:
        await _conn.close()
        _conn = None
        _checkpointer = None
        logger.info("[Checkpointer] SQLite connection closed")


def get_checkpointer() -> AsyncSqliteSaver:
    """
    Return the live checkpointer instance.
    Raises RuntimeError if init_checkpointer() has not been called.
    """
    if _checkpointer is None:
        raise RuntimeError(
            "Checkpointer not initialised. "
            "Ensure init_checkpointer() is called in the FastAPI lifespan."
        )
    return _checkpointer