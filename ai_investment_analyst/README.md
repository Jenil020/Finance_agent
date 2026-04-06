# AI Investment Analyst 🤖📈

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
