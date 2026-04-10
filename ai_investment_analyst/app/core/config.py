from functools import lru_cache
import os
from pathlib import Path

from pydantic_settings import BaseSettings

"""App-wide settings loaded from .env via pydantic-settings."""


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # ── LLM (Google Gemini - free tier) ──────────────────────────────────────
    google_api_key: str = "AIzaSyC4zwwOJVo-_9uPJWsN2qeJPURIEcPdGr8"
    gemini_model: str = "gemini-2.5-flash"
    gemini_model_fallbacks: tuple[str, ...] = (
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-flash-latest",
    )
    gemini_embed_model: str = "gemini-embedding-001"  # Free tier embedding
    gemini_embed_output_dim: int = 3072

    # ── Redis (Upstash free or local) ────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"
    redis_cache_ttl: int = 3600      # 1 hour semantic cache TTL
    redis_session_ttl: int = 86400   # 24 hour conversation TTL

    # ── LangSmith (optional - free 5k traces/month) ──────────────────────────
    langchain_api_key: str = os.getenv("LANGSMITH_API_KEY")
    langchain_tracing_v2: bool = True
    langchain_project: str = "ai-investment-analyst"

    # ── App ───────────────────────────────────────────────────────────────────
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    log_file_path: str = str(BASE_DIR / "logs" / "app.log")

    # ── Vector DB (Qdrant local disk - completely free) ───────────────────────
    qdrant_path: str = str(BASE_DIR / "data" / "qdrant")
    qdrant_collection: str = "investments"
    qdrant_vector_size: int = 3072

    # ── RAG Pipeline ─────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    crag_confidence_threshold: float = 0.4   # Below → fallback to web search

    # ── SQLite checkpointer (LangGraph state persistence) ─────────────────────
    sqlite_db_path: str = str(BASE_DIR / "data" / "checkpoints.db")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
