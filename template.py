"""
template.py — AI Investment Analyst Project Structure Generator
Run: python template.py
Creates the full folder/file skeleton for the project.
"""

import os
from pathlib import Path

# ── Project root name ──────────────────────────────────────────────────────────
PROJECT_NAME = "ai_investment_analyst"

# ── File tree definition ───────────────────────────────────────────────────────
# Each entry: ("relative/path/to/file", "optional starter content")
STRUCTURE = [
    # ── Root config files ──────────────────────────────────────────────────────
    (".env.example",                """# ── LLM ───────────────────────────────────
GOOGLE_API_KEY=your_gemini_api_key_here

# ── Redis (Upstash free tier or local) ───
REDIS_URL=redis://localhost:6379

# ── LangSmith (optional, free tier) ──────
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=ai-investment-analyst

# ── App ───────────────────────────────────
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO
"""),

    ("README.md",                   """# AI Investment Analyst 🤖📈

A multi-agent AI system for investment research and analysis.

## Architecture
- **LangGraph** — Stateful multi-agent orchestration
- **LlamaIndex** — RAG knowledge base with hybrid search
- **Google Gemini** — Free LLM API (gemini-1.5-flash)
- **Qdrant** — Local vector database (HNSW indexing)
- **Redis** — Conversation memory & semantic cache
- **FastAPI** — Streaming SSE API

## Agents
1. **Research Agent** — ReAct pattern, web search tools
2. **Analysis Agent** — Tool calling, portfolio calculations
3. **Report Agent** — Structured JSON output via Pydantic

## Setup
```bash
python setup.py        # Install dependencies & verify env
cp .env.example .env   # Fill in your API keys
python template.py     # (already done) Generate structure
uvicorn app.main:app --reload
```

## Features (Production Practices)
- Hybrid Search (BM25 + Vector)
- Semantic caching (Redis)
- Streaming API (FastAPI SSE)
- Structured output (Pydantic)
- CRAG self-correction
- Conversation summary memory
- Tool retry (Tenacity)
- LangSmith traces
"""),

    ("pyproject.toml",              """[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
line-length = 100
"""),

    # ── Application package ────────────────────────────────────────────────────
    ("app/__init__.py",             ""),
    ("app/main.py",                 '''"""FastAPI application entry point with SSE streaming."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.core.config import settings

app = FastAPI(
    title="AI Investment Analyst",
    description="Multi-agent investment research system",
    version="1.0.0",
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
'''),

    # ── Core config ────────────────────────────────────────────────────────────
    ("app/core/__init__.py",        ""),
    ("app/core/config.py",          '''"""App-wide settings loaded from .env via pydantic-settings."""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    google_api_key: str = ""

    # Redis
    redis_url: str = "redis://localhost:6379"

    # LangSmith
    langchain_api_key: str = ""
    langchain_tracing_v2: bool = False
    langchain_project: str = "ai-investment-analyst"

    # App
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # Vector DB
    qdrant_path: str = "./data/qdrant"   # local disk storage (free)
    qdrant_collection: str = "investments"

    # RAG
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    embed_model: str = "models/embedding-001"   # Gemini embedding (free)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
'''),

    ("app/core/logging.py",         '''"""Structured logging setup."""
import logging
import sys
from app.core.config import settings


def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


logger = logging.getLogger("investment_analyst")
'''),

    # ── API layer ──────────────────────────────────────────────────────────────
    ("app/api/__init__.py",         ""),
    ("app/api/routes.py",           '''"""API routes — chat, document ingestion, health."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.api.schemas import ChatRequest, ChatResponse, IngestRequest
from app.orchestration.graph import run_agent_stream
from app.rag.ingestion import ingest_documents

router = APIRouter()


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream SSE tokens from the multi-agent pipeline."""
    return StreamingResponse(
        run_agent_stream(
            query=request.query,
            session_id=request.session_id,
            portfolio=request.portfolio,
        ),
        media_type="text/event-stream",
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint (collects full response)."""
    # TODO: Implement non-streaming wrapper
    raise HTTPException(status_code=501, detail="Use /chat/stream instead")


@router.post("/ingest")
async def ingest(request: IngestRequest):
    """Ingest documents into the RAG knowledge base."""
    result = await ingest_documents(request.file_paths, request.metadata)
    return {"status": "ok", "indexed": result["count"]}


@router.delete("/ingest/{collection}")
async def clear_collection(collection: str):
    """Clear a Qdrant collection (for dev/testing)."""
    # TODO: Implement collection cleanup
    return {"status": "cleared", "collection": collection}
'''),

    ("app/api/schemas.py",          '''"""Pydantic request/response schemas."""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class PortfolioItem(BaseModel):
    ticker: str
    quantity: float
    avg_cost: float


class ChatRequest(BaseModel):
    query: str = Field(..., description="User investment query")
    session_id: str = Field(..., description="Unique session identifier")
    portfolio: Optional[List[PortfolioItem]] = Field(
        default=None, description="User portfolio for analysis"
    )


class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []
    agent_trace: List[str] = []
    session_id: str


class IngestRequest(BaseModel):
    file_paths: List[str]
    metadata: Optional[Dict[str, Any]] = None


class InvestmentReport(BaseModel):
    """Structured output from the Report Agent."""
    ticker: str
    recommendation: str = Field(..., description="BUY / SELL / HOLD")
    confidence: float = Field(..., ge=0.0, le=1.0)
    summary: str
    key_risks: List[str]
    key_catalysts: List[str]
    target_price: Optional[float] = None
    sources: List[str] = []
'''),

    # ── Orchestration (LangGraph) ─────────────────────────────────────────────
    ("app/orchestration/__init__.py",""),
    ("app/orchestration/graph.py",  '''"""
LangGraph stateful orchestration graph.
Defines nodes (agents) and edges (routing logic).
"""
from typing import AsyncGenerator
from langgraph.graph import StateGraph, END
from app.orchestration.state import AgentState
from app.agents.research import research_agent_node
from app.agents.analysis import analysis_agent_node
from app.agents.report import report_agent_node
from app.orchestration.router import route_after_research


def build_graph() -> StateGraph:
    """Build and compile the LangGraph agent graph."""
    workflow = StateGraph(AgentState)

    # Add agent nodes
    workflow.add_node("research", research_agent_node)
    workflow.add_node("analysis", analysis_agent_node)
    workflow.add_node("report", report_agent_node)

    # Entry point
    workflow.set_entry_point("research")

    # Conditional routing after research
    workflow.add_conditional_edges(
        "research",
        route_after_research,
        {
            "analysis": "analysis",
            "report": "report",      # Skip analysis if not needed
            "end": END,
        },
    )

    workflow.add_edge("analysis", "report")
    workflow.add_edge("report", END)

    return workflow.compile()


# Singleton compiled graph
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


async def run_agent_stream(
    query: str,
    session_id: str,
    portfolio=None,
) -> AsyncGenerator[str, None]:
    """Run the graph and stream SSE events."""
    graph = get_graph()
    initial_state = AgentState(
        query=query,
        session_id=session_id,
        portfolio=portfolio or [],
        messages=[],
        research_results=[],
        analysis_results={},
        final_report=None,
        error=None,
    )
    async for event in graph.astream(initial_state):
        # Stream each node output as SSE data
        for node_name, output in event.items():
            if isinstance(output, dict) and "stream_token" in output:
                yield f"data: {output['stream_token']}\\n\\n"
        yield "data: [DONE]\\n\\n"
'''),

    ("app/orchestration/state.py",  '''"""LangGraph shared state definition."""
from typing import TypedDict, List, Optional, Dict, Any
from app.api.schemas import PortfolioItem, InvestmentReport


class AgentState(TypedDict):
    query: str
    session_id: str
    portfolio: List[PortfolioItem]
    messages: List[Dict[str, Any]]       # Conversation history
    research_results: List[str]           # Web search / RAG results
    analysis_results: Dict[str, Any]      # Portfolio metrics
    final_report: Optional[InvestmentReport]
    error: Optional[str]
    stream_token: Optional[str]           # For SSE streaming
'''),

    ("app/orchestration/router.py", '''"""Routing logic between LangGraph nodes."""
from app.orchestration.state import AgentState


def route_after_research(state: AgentState) -> str:
    """Decide next node after research agent completes."""
    if state.get("error"):
        return "end"
    # If portfolio provided, run full analysis; else go straight to report
    if state.get("portfolio"):
        return "analysis"
    return "report"
'''),

    ("app/orchestration/checkpointer.py", '''"""
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
'''),

    # ── Agents ────────────────────────────────────────────────────────────────
    ("app/agents/__init__.py",      ""),
    ("app/agents/base.py",          '''"""Base class / shared utilities for all agents."""
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
from app.core.config import settings
from app.core.logging import logger

genai.configure(api_key=settings.google_api_key)


def get_gemini_model(model_name: str = "gemini-1.5-flash"):
    """Return a configured Gemini model instance."""
    return genai.GenerativeModel(model_name)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
async def llm_call_with_retry(model, prompt: str) -> str:
    """LLM call with exponential backoff retry (Tenacity)."""
    response = await model.generate_content_async(prompt)
    return response.text
'''),

    ("app/agents/research.py",      '''"""
Research Agent — ReAct pattern with web search tool.
Responsibilities: fetch news, financial data, RAG retrieval.
"""
from app.orchestration.state import AgentState
from app.agents.base import get_gemini_model, llm_call_with_retry
from app.tools.search import web_search_tool
from app.tools.rag_query import rag_query_tool
from app.core.logging import logger

RESEARCH_PROMPT = """You are a financial research analyst. 
Given the query: {query}

Use the available tools to gather:
1. Recent news and market data
2. Relevant documents from the knowledge base
3. Key financial metrics

Be thorough but concise. Focus on data relevant to investment decisions."""


async def research_agent_node(state: AgentState) -> AgentState:
    """LangGraph node: Research Agent."""
    logger.info(f"[Research Agent] Processing query: {state['query'][:80]}")

    model = get_gemini_model()

    # Step 1: RAG retrieval from knowledge base
    rag_results = await rag_query_tool(state["query"])

    # Step 2: Web search for latest data
    search_results = await web_search_tool(state["query"])

    # Step 3: Synthesize with LLM
    prompt = RESEARCH_PROMPT.format(query=state["query"])
    prompt += f"\\n\\nRAG Results:\\n{rag_results}"
    prompt += f"\\n\\nWeb Search Results:\\n{search_results}"

    synthesis = await llm_call_with_retry(model, prompt)

    return {
        **state,
        "research_results": [synthesis],
        "messages": state["messages"] + [
            {"role": "research_agent", "content": synthesis}
        ],
    }
'''),

    ("app/agents/analysis.py",      '''"""
Analysis Agent — Portfolio calculations and financial analysis.
Uses tool calling for quantitative computations.
"""
from app.orchestration.state import AgentState
from app.agents.base import get_gemini_model, llm_call_with_retry
from app.tools.portfolio import calculate_portfolio_metrics
from app.core.logging import logger

ANALYSIS_PROMPT = """You are a quantitative investment analyst.

Query: {query}
Research findings: {research}
Portfolio: {portfolio}
Portfolio Metrics: {metrics}

Provide a detailed analysis including:
- Risk assessment (volatility, beta, correlation)
- Return analysis (current vs benchmark)
- Position sizing recommendations
- Entry/exit considerations

Be specific with numbers."""


async def analysis_agent_node(state: AgentState) -> AgentState:
    """LangGraph node: Analysis Agent."""
    logger.info("[Analysis Agent] Running portfolio analysis")

    model = get_gemini_model()

    # Calculate portfolio metrics using tool
    metrics = {}
    if state.get("portfolio"):
        metrics = await calculate_portfolio_metrics(state["portfolio"])

    prompt = ANALYSIS_PROMPT.format(
        query=state["query"],
        research="\\n".join(state.get("research_results", [])),
        portfolio=state.get("portfolio", []),
        metrics=metrics,
    )

    analysis = await llm_call_with_retry(model, prompt)

    return {
        **state,
        "analysis_results": {"summary": analysis, "metrics": metrics},
        "messages": state["messages"] + [
            {"role": "analysis_agent", "content": analysis}
        ],
    }
'''),

    ("app/agents/report.py",        '''"""
Report Agent — Structured output generation using Pydantic.
Produces final investment report as validated JSON.
"""
import json
from app.orchestration.state import AgentState
from app.agents.base import get_gemini_model, llm_call_with_retry
from app.api.schemas import InvestmentReport
from app.core.logging import logger

REPORT_PROMPT = """You are a senior investment analyst writing a formal report.

Query: {query}
Research: {research}
Analysis: {analysis}

Produce a JSON investment report with EXACTLY this structure:
{{
  "ticker": "<stock ticker or 'PORTFOLIO'>",
  "recommendation": "<BUY|SELL|HOLD>",
  "confidence": <0.0 to 1.0>,
  "summary": "<2-3 sentence executive summary>",
  "key_risks": ["<risk1>", "<risk2>", "<risk3>"],
  "key_catalysts": ["<catalyst1>", "<catalyst2>"],
  "target_price": <float or null>,
  "sources": ["<source1>", "<source2>"]
}}

Return ONLY valid JSON, no markdown, no explanation."""


async def report_agent_node(state: AgentState) -> AgentState:
    """LangGraph node: Report Agent — structured JSON output."""
    logger.info("[Report Agent] Generating structured report")

    model = get_gemini_model()

    prompt = REPORT_PROMPT.format(
        query=state["query"],
        research="\\n".join(state.get("research_results", [])),
        analysis=state.get("analysis_results", {}).get("summary", "N/A"),
    )

    raw_output = await llm_call_with_retry(model, prompt)

    # Parse and validate with Pydantic
    try:
        cleaned = raw_output.strip().strip("```json").strip("```").strip()
        report_data = json.loads(cleaned)
        report = InvestmentReport(**report_data)
    except Exception as e:
        logger.error(f"[Report Agent] Failed to parse report: {e}")
        report = InvestmentReport(
            ticker="UNKNOWN",
            recommendation="HOLD",
            confidence=0.0,
            summary=raw_output[:500],
            key_risks=["Parse error"],
            key_catalysts=[],
        )

    return {
        **state,
        "final_report": report,
        "messages": state["messages"] + [
            {"role": "report_agent", "content": report.model_dump_json()}
        ],
    }
'''),

    # ── RAG pipeline (LlamaIndex) ─────────────────────────────────────────────
    ("app/rag/__init__.py",         ""),
    ("app/rag/ingestion.py",        '''"""
Document ingestion pipeline.
Loads PDF/DOCX/CSV → chunks → embeds → stores in Qdrant.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from app.rag.vector_store import get_vector_store
from app.rag.embeddings import get_embed_model
from app.core.config import settings
from app.core.logging import logger


async def ingest_documents(
    file_paths: List[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Ingest documents into the RAG knowledge base.
    Supports: PDF, DOCX, CSV, TXT
    """
    logger.info(f"[Ingestion] Starting ingestion for {len(file_paths)} files")

    # Load documents
    documents = SimpleDirectoryReader(input_files=file_paths).load_data()

    # Attach metadata (tenant, category, etc.)
    if metadata:
        for doc in documents:
            doc.metadata.update(metadata)

    # Semantic chunking
    splitter = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(documents)

    # Index into Qdrant
    vector_store = get_vector_store()
    embed_model = get_embed_model()
    index = VectorStoreIndex(
        nodes,
        vector_store=vector_store,
        embed_model=embed_model,
    )

    logger.info(f"[Ingestion] Indexed {len(nodes)} chunks")
    return {"count": len(nodes), "files": file_paths}
'''),

    ("app/rag/retriever.py",        '''"""
Hybrid retriever: BM25 (keyword) + Vector (semantic) search.
CRAG self-correction layer for low-confidence results.
"""
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from app.rag.vector_store import get_vector_store
from app.rag.embeddings import get_embed_model
from app.core.config import settings


def get_hybrid_retriever(index: VectorStoreIndex):
    """
    Combine BM25 + Vector retrieval using Reciprocal Rank Fusion.
    Improves recall vs pure vector search.
    """
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=settings.top_k,
    )

    bm25_retriever = BM25Retriever.from_defaults(
        index=index,
        similarity_top_k=settings.top_k,
    )

    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=settings.top_k,
        num_queries=1,          # Don't generate sub-queries here
        mode="reciprocal_rerank",
        use_async=True,
    )

    return hybrid_retriever


async def crag_retrieve(query: str, retriever) -> tuple[list, float]:
    """
    CRAG (Corrective RAG) — evaluate retrieval confidence.
    If confidence < threshold, fall back to web search.
    Returns (nodes, confidence_score)
    """
    nodes = await retriever.aretrieve(query)

    if not nodes:
        return [], 0.0

    # Simple confidence: average similarity score
    avg_score = sum(n.score or 0.0 for n in nodes) / len(nodes)
    return nodes, avg_score
'''),

    ("app/rag/vector_store.py",     '''"""Qdrant vector store — local disk mode (free, no cloud needed)."""
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from app.core.config import settings

_client = None
_vector_store = None


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        # Local disk persistence — free, no cloud account needed
        _client = QdrantClient(path=settings.qdrant_path)
    return _client


def get_vector_store() -> QdrantVectorStore:
    global _vector_store
    if _vector_store is None:
        client = get_qdrant_client()
        _vector_store = QdrantVectorStore(
            client=client,
            collection_name=settings.qdrant_collection,
        )
    return _vector_store
'''),

    ("app/rag/embeddings.py",       '''"""Google Gemini embedding model (free API)."""
from llama_index.embeddings.gemini import GeminiEmbedding
from app.core.config import settings

_embed_model = None


def get_embed_model() -> GeminiEmbedding:
    global _embed_model
    if _embed_model is None:
        _embed_model = GeminiEmbedding(
            model_name=settings.embed_model,
            api_key=settings.google_api_key,
        )
    return _embed_model
'''),

    # ── Tools ─────────────────────────────────────────────────────────────────
    ("app/tools/__init__.py",       ""),
    ("app/tools/search.py",         '''"""
Web search tool for the Research Agent.
Uses DuckDuckGo (free, no API key needed) via duckduckgo-search.
"""
from duckduckgo_search import DDGS
from app.core.logging import logger


async def web_search_tool(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo.
    Returns formatted string of results.
    """
    logger.info(f"[Web Search] Query: {query[:80]}")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                query + " investment finance stock",
                max_results=max_results,
            ))
        formatted = []
        for r in results:
            formatted.append(f"Title: {r.get('title', '')}\\n"
                             f"URL: {r.get('href', '')}\\n"
                             f"Snippet: {r.get('body', '')}\\n")
        return "\\n---\\n".join(formatted)
    except Exception as e:
        logger.error(f"[Web Search] Failed: {e}")
        return f"Web search unavailable: {e}"
'''),

    ("app/tools/rag_query.py",      '''"""RAG query tool — used by Research Agent."""
from app.rag.vector_store import get_vector_store
from app.rag.embeddings import get_embed_model
from app.rag.retriever import get_hybrid_retriever, crag_retrieve
from llama_index.core import VectorStoreIndex
from app.core.config import settings
from app.core.logging import logger

WEB_SEARCH_THRESHOLD = 0.4  # Below this, fall back to web search


async def rag_query_tool(query: str) -> str:
    """
    Query the RAG knowledge base with CRAG self-correction.
    Falls back to web-search note if confidence is low.
    """
    logger.info(f"[RAG Query] {query[:80]}")
    try:
        vector_store = get_vector_store()
        embed_model = get_embed_model()
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )
        retriever = get_hybrid_retriever(index)
        nodes, confidence = await crag_retrieve(query, retriever)

        if not nodes or confidence < WEB_SEARCH_THRESHOLD:
            logger.warning(f"[CRAG] Low confidence ({confidence:.2f}), flagging for web search")
            return f"[Low RAG confidence: {confidence:.2f}] Recommend supplementing with web search."

        texts = [n.get_content() for n in nodes[:settings.top_k]]
        return "\\n\\n".join(texts)

    except Exception as e:
        logger.error(f"[RAG Query] Error: {e}")
        return f"RAG unavailable: {e}"
'''),

    ("app/tools/portfolio.py",      '''"""
Portfolio calculation tools for the Analysis Agent.
Pure Python — no external API needed.
"""
from typing import List, Dict, Any
from app.api.schemas import PortfolioItem


async def calculate_portfolio_metrics(
    portfolio: List[PortfolioItem],
) -> Dict[str, Any]:
    """
    Calculate basic portfolio metrics.
    In production: integrate with yfinance for live prices.
    """
    if not portfolio:
        return {}

    total_cost = sum(item.quantity * item.avg_cost for item in portfolio)
    tickers = [item.ticker for item in portfolio]
    weights = {
        item.ticker: (item.quantity * item.avg_cost) / total_cost
        for item in portfolio
    }

    return {
        "total_cost_basis": round(total_cost, 2),
        "tickers": tickers,
        "weights": weights,
        "num_positions": len(portfolio),
        # TODO: Fetch live prices via yfinance and compute P&L, beta, etc.
    }
'''),

    # ── Memory (Redis) ────────────────────────────────────────────────────────
    ("app/memory/__init__.py",      ""),
    ("app/memory/redis_memory.py",  '''"""
Redis-backed conversation memory & semantic cache.
Uses Upstash free tier or local Redis.
"""
import json
import redis.asyncio as aioredis
from app.core.config import settings
from app.core.logging import logger

_redis_client = None


async def get_redis() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = await aioredis.from_url(
            settings.redis_url, decode_responses=True
        )
    return _redis_client


async def save_conversation(session_id: str, messages: list, ttl: int = 86400):
    """Persist conversation history to Redis (TTL: 24h default)."""
    client = await get_redis()
    key = f"session:{session_id}:messages"
    await client.set(key, json.dumps(messages), ex=ttl)
    logger.debug(f"[Redis] Saved {len(messages)} messages for session {session_id}")


async def load_conversation(session_id: str) -> list:
    """Load conversation history from Redis."""
    client = await get_redis()
    key = f"session:{session_id}:messages"
    data = await client.get(key)
    return json.loads(data) if data else []


async def semantic_cache_get(query_hash: str) -> str | None:
    """Check semantic cache for a previously answered query."""
    client = await get_redis()
    return await client.get(f"cache:{query_hash}")


async def semantic_cache_set(query_hash: str, response: str, ttl: int = 3600):
    """Store response in semantic cache (TTL: 1h)."""
    client = await get_redis()
    await client.set(f"cache:{query_hash}", response, ex=ttl)
'''),

    # ── Data directory ────────────────────────────────────────────────────────
    ("data/.gitkeep",               "# Local data — Qdrant DB, SQLite checkpoints, sample docs"),
    ("data/sample_docs/.gitkeep",   "# Place PDF/DOCX/CSV files here for ingestion"),

    # ── Tests ─────────────────────────────────────────────────────────────────
    ("tests/__init__.py",           ""),
    ("tests/test_agents.py",        '''"""Unit tests for agent nodes."""
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_research_agent_returns_results():
    """Research agent should populate research_results."""
    from app.agents.research import research_agent_node
    state = {
        "query": "Should I buy Apple stock?",
        "session_id": "test-123",
        "portfolio": [],
        "messages": [],
        "research_results": [],
        "analysis_results": {},
        "final_report": None,
        "error": None,
        "stream_token": None,
    }
    with patch("app.agents.research.llm_call_with_retry", new_callable=AsyncMock) as mock_llm, \\
         patch("app.agents.research.rag_query_tool", new_callable=AsyncMock) as mock_rag, \\
         patch("app.agents.research.web_search_tool", new_callable=AsyncMock) as mock_web:
        mock_llm.return_value = "Apple looks strong with solid fundamentals."
        mock_rag.return_value = "RAG result: AAPL P/E ratio is 28."
        mock_web.return_value = "Apple Q4 earnings beat estimates."
        result = await research_agent_node(state)
    assert len(result["research_results"]) > 0


@pytest.mark.asyncio
async def test_report_agent_validates_schema():
    """Report agent must produce Pydantic-valid InvestmentReport."""
    import json
    from app.agents.report import report_agent_node
    valid_json = json.dumps({
        "ticker": "AAPL",
        "recommendation": "BUY",
        "confidence": 0.78,
        "summary": "Apple is a strong buy.",
        "key_risks": ["macro headwinds"],
        "key_catalysts": ["AI integration"],
        "target_price": 220.0,
        "sources": ["Reuters"],
    })
    state = {
        "query": "Analyse AAPL",
        "session_id": "test-456",
        "portfolio": [],
        "messages": [],
        "research_results": ["Apple is innovating rapidly."],
        "analysis_results": {"summary": "Metrics look good."},
        "final_report": None,
        "error": None,
        "stream_token": None,
    }
    with patch("app.agents.report.llm_call_with_retry", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = valid_json
        result = await report_agent_node(state)
    assert result["final_report"] is not None
    assert result["final_report"].recommendation in ("BUY", "SELL", "HOLD")
'''),

    ("tests/test_rag.py",           '''"""Tests for RAG pipeline."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_rag_query_returns_string():
    from app.tools.rag_query import rag_query_tool
    with patch("app.tools.rag_query.get_vector_store"), \\
         patch("app.tools.rag_query.get_embed_model"), \\
         patch("app.tools.rag_query.VectorStoreIndex") as mock_index, \\
         patch("app.tools.rag_query.get_hybrid_retriever") as mock_ret, \\
         patch("app.tools.rag_query.crag_retrieve", new_callable=AsyncMock) as mock_crag:
        mock_node = MagicMock()
        mock_node.get_content.return_value = "AAPL has strong cash flow."
        mock_crag.return_value = ([mock_node], 0.85)
        result = await rag_query_tool("What is Apple's financial health?")
    assert isinstance(result, str)
    assert len(result) > 0
'''),

    ("tests/test_portfolio.py",     '''"""Tests for portfolio calculation tools."""
import pytest
from app.tools.portfolio import calculate_portfolio_metrics
from app.api.schemas import PortfolioItem


@pytest.mark.asyncio
async def test_portfolio_metrics_basic():
    portfolio = [
        PortfolioItem(ticker="AAPL", quantity=10, avg_cost=150.0),
        PortfolioItem(ticker="GOOGL", quantity=5, avg_cost=140.0),
    ]
    metrics = await calculate_portfolio_metrics(portfolio)
    assert metrics["total_cost_basis"] == 2200.0
    assert metrics["num_positions"] == 2
    assert abs(metrics["weights"]["AAPL"] - 0.6818) < 0.01


@pytest.mark.asyncio
async def test_empty_portfolio():
    metrics = await calculate_portfolio_metrics([])
    assert metrics == {}
'''),

    # ── Notebooks ─────────────────────────────────────────────────────────────
    ("notebooks/.gitkeep",          "# Jupyter notebooks for exploration and demos"),

    # ── Scripts ───────────────────────────────────────────────────────────────
    ("scripts/ingest_sample.py",    '''"""
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
    print(f"Ingested {result[\'count\']} chunks from {result[\'files\']}")


if __name__ == "__main__":
    asyncio.run(main())
'''),

    ("scripts/test_chat.py",        '''"""
Quick smoke test for the full agent pipeline.
Usage: python scripts/test_chat.py
"""
import asyncio
from app.orchestration.graph import run_agent_stream


async def main():
    print("Running test query through agent pipeline...\\n")
    async for token in run_agent_stream(
        query="Should I invest in NVIDIA given current AI trends?",
        session_id="smoke-test-001",
    ):
        print(token, end="", flush=True)
    print("\\n\\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
'''),
]


def create_structure(base_path: Path, structure: list) -> None:
    print(f"\n🚀 Creating project: {base_path.name}\n{'─'*50}")
    created_files = 0
    created_dirs = set()

    for rel_path, content in structure:
        full_path = base_path / rel_path
        parent = full_path.parent

        # Create parent dirs
        if parent not in created_dirs:
            parent.mkdir(parents=True, exist_ok=True)
            if parent != base_path:
                created_dirs.add(parent)
                print(f"  📁 {parent.relative_to(base_path)}/")

        # Write file (skip if already exists)
        if not full_path.exists():
            full_path.write_text(content, encoding="utf-8")
            print(f"     ✅ {rel_path}")
            created_files += 1
        else:
            print(f"     ⏭  {rel_path}  (already exists, skipped)")

    print(f"\n{'─'*50}")
    print(f"✨ Done! Created {created_files} files in ./{base_path.name}/")
    print(f"\nNext steps:")
    print(f"  1. cd {base_path.name}")
    print(f"  2. python setup.py          # install deps & verify env")
    print(f"  3. cp .env.example .env     # add your API keys")
    print(f"  4. uvicorn app.main:app --reload")


if __name__ == "__main__":
    project_root = Path(PROJECT_NAME)
    create_structure(project_root, STRUCTURE)