"""FastAPI application entry point with SSE streaming."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.core.logging import setup_logging
from app.orchestration.checkpointer import init_checkpointer, close_checkpointer
from app.core.config import settings

setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_checkpointer()
    yield
    await close_checkpointer()

app = FastAPI(
    title="AI Investment Analyst",
    description="Multi-agent investment research system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "ai-investment-analyst"}
