"""
SQLite checkpointer for LangGraph state persistence.
Enables human-in-the-loop and resume-from-checkpoint.
"""
from langgraph.checkpoint.sqlite import SqliteSaver

_checkpointer = None

def get_checkpointer() -> SqliteSaver:
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = SqliteSaver.from_conn_string("./data/checkpoints.db")
    return _checkpointer
