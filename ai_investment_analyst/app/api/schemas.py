"""Pydantic request/response schemas."""
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
