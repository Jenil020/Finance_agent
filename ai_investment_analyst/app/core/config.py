"""App-wide settings loaded from .env via pydantic-settings."""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── LLM (Google Gemini - free tier) ──────────────────────────────────────
    google_api_key: str = "AIzaSyCP2gL1nBa3cGDJBIc3dcqguzdIXFpBi0E"
    gemini_model: str = "gemini-1.5-flash"          # Free tier, fast
    gemini_embed_model: str = "gemini-embedding-001"  # Free tier embedding

    # ── Redis (Upstash free or local) ────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"
    redis_cache_ttl: int = 3600      # 1 hour semantic cache TTL
    redis_session_ttl: int = 86400   # 24 hour conversation TTL

    # ── LangSmith (optional - free 5k traces/month) ──────────────────────────
    langchain_api_key: str = ""
    langchain_tracing_v2: bool = False
    langchain_project: str = "ai-investment-analyst"

    # ── App ───────────────────────────────────────────────────────────────────
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # ── Vector DB (Qdrant local disk - completely free) ───────────────────────
    qdrant_path: str = "./data/qdrant"
    qdrant_collection: str = "investments"
    qdrant_vector_size: int = 768    # models/text_embedding_004 output dim

    # ── RAG Pipeline ─────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    crag_confidence_threshold: float = 0.4   # Below → fallback to web search

    # ── SQLite checkpointer (LangGraph state persistence) ─────────────────────
    sqlite_db_path: str = "./data/checkpoints.db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()