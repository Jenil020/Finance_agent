"""
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
        research="\n".join(state.get("research_results", [])),
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
