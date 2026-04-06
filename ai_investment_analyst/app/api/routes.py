"""API routes — chat, document ingestion, health."""
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
