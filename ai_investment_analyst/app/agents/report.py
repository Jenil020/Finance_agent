"""
Report Agent — Final structured output generator.

Responsibilities:
  1. Synthesise research + analysis into an investment recommendation
  2. Output a Pydantic-validated InvestmentReport (JSON)
  3. Stream the narrative to the SSE endpoint token-by-token

JSON mode (response_mime_type="application/json") is set in llm_call so
Gemini is hard-constrained to return valid JSON — no markdown fences,
no preamble. A secondary plain-text parse fallback handles edge cases.
"""
import json
import re
from app.orchestration.state import AgentState
from app.agents.base import llm_call, llm_call_stream
from app.api.schemas import InvestmentReport
from app.core.logging import logger

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a managing director of equity research at a bulge-bracket bank.
You write the final investment recommendation based on evidence from your research
and quantitative analysis teams. Your recommendations are measured, evidence-based,
and clearly communicate both upside potential and downside risks.
Never recommend based on speculation — only verifiable data."""

# ── JSON report prompt ────────────────────────────────────────────────────────
REPORT_PROMPT = """INVESTMENT REPORT GENERATION
=============================
Query: {query}

RESEARCH BRIEF:
{research}

QUANTITATIVE ANALYSIS:
{analysis}

TASK
----
Produce a JSON investment report. Return ONLY a JSON object matching this schema exactly:
{{
  "ticker":          "<primary stock ticker, or 'PORTFOLIO' for multi-stock, or 'MARKET'>",
  "recommendation":  "<BUY | SELL | HOLD>",
  "confidence":      <float 0.0–1.0, e.g. 0.72>,
  "summary":         "<2-3 sentence executive summary of the recommendation>",
  "key_risks": [
    "<specific risk 1 with data>",
    "<specific risk 2 with data>",
    "<specific risk 3 with data>"
  ],
  "key_catalysts": [
    "<specific catalyst 1 with data>",
    "<specific catalyst 2 with data>"
  ],
  "target_price":    <float or null — only if single ticker and price data available>,
  "sources":         ["<source1>", "<source2>"]
}}

Rules:
- confidence 0.0–0.4 = LOW (significant uncertainty)
- confidence 0.4–0.7 = MEDIUM (reasonable basis)
- confidence 0.7–1.0 = HIGH (strong evidence)
- key_risks and key_catalysts must contain specific numbers, not vague statements
- target_price must be null if you cannot justify it with data"""

# ── Narrative streaming prompt ────────────────────────────────────────────────
NARRATIVE_PROMPT = """INVESTMENT NARRATIVE
====================
Query: {query}

Based on this recommendation:
{report_json}

Write a 3-4 paragraph narrative investment memo (plain English, no JSON).
Structure:
  Para 1: Executive summary and recommendation rationale
  Para 2: Key supporting evidence from research
  Para 3: Risk factors and what could invalidate the thesis
  Para 4: Conclusion — what an investor should watch for

Tone: professional, concise, institutional-grade."""


def _build_fallback_report(state: AgentState, query: str, raw_error: Exception) -> InvestmentReport:
    """Return a minimal but valid report when report generation is unavailable."""
    sources = _sources_from_state(state)
    analysis_summary = state.get("analysis_results", {}).get("summary", "")
    research_summary = "\n".join(state.get("research_results", []))
    summary = (
        "Report generation used a fallback because the LLM was unavailable. "
        "Review the research and analysis sections for the raw evidence."
    )
    if analysis_summary:
        summary += f" Analysis snapshot: {analysis_summary[:220]}"
    elif research_summary:
        summary += f" Research snapshot: {research_summary[:220]}"

    return InvestmentReport(
        ticker=_extract_ticker_heuristic(query),
        recommendation="HOLD",
        confidence=0.0,
        summary=summary,
        key_risks=[
            "Final report generation was unavailable due to LLM/API limits",
            "Recommendation confidence is low because structured synthesis did not run",
        ],
        key_catalysts=[
            "Retry once model quota resets",
            "Use the research and portfolio metrics already gathered in this run",
        ],
        target_price=None,
        sources=sources,
    )


def _build_fallback_narrative(report: InvestmentReport, raw_error: Exception) -> str:
    """Return a simple narrative when LLM narrative generation is unavailable."""
    return (
        f"Recommendation: {report.recommendation} with confidence {report.confidence:.2f}.\n\n"
        "This narrative was generated from a fallback path because the language model "
        f"was unavailable: {raw_error}\n\n"
        f"Summary: {report.summary}"
    )


def _extract_json_from_text(text: str) -> dict | None:
    """
    Multi-strategy JSON extraction for cases where the model adds extra text.
    Strategy 1: Direct parse
    Strategy 2: Strip markdown fences (```json ... ```)
    Strategy 3: Regex extract first { ... } block
    """
    # Strategy 1 — direct
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2 — strip markdown fences
    cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3 — find outermost { }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def _sources_from_state(state: AgentState) -> list[str]:
    """Extract source tags accumulated across agent messages."""
    sources = set()
    for msg in state.get("messages", []):
        for s in msg.get("sources", []):
            sources.add(s)
    return sorted(sources)


# ── Agent node ────────────────────────────────────────────────────────────────

async def report_agent_node(state: AgentState) -> dict:
    """
    LangGraph node: Report Agent.

    Returns partial state — updates final_report, messages.
    """
    query = state["query"]
    research = "\n".join(state.get("research_results", ["No research available."]))
    analysis = state.get("analysis_results", {}).get("summary", "No analysis available.")

    logger.info("[Report] Generating structured investment report")

    # ── Step 1: Generate JSON report ─────────────────────────────────────────
    json_prompt = REPORT_PROMPT.format(
        query=query,
        research=research,
        analysis=analysis,
    )

    try:
        raw_json = await llm_call(
            prompt=json_prompt,
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1,         # very low — structured output must be precise
            max_output_tokens=1024,
            json_mode=True,          # constrains Gemini to valid JSON output
        )
    except Exception as e:
        logger.warning(f"[Report] Structured report generation unavailable, using fallback: {e}")
        raw_json = ""
        report = _build_fallback_report(state, query, e)

    # ── Step 2: Parse + validate with Pydantic ────────────────────────────────
    report: InvestmentReport | None = locals().get("report")
    sources = _sources_from_state(state)

    parsed = _extract_json_from_text(raw_json) if raw_json else None
    if report is None and parsed is not None:
        try:
            # Inject sources from state if model left them empty
            if not parsed.get("sources") and sources:
                parsed["sources"] = sources
            report = InvestmentReport(**parsed)
            logger.info(
                f"[Report] Parsed OK | "
                f"ticker={report.ticker} | "
                f"rec={report.recommendation} | "
                f"confidence={report.confidence:.2f}"
            )
        except Exception as e:
            logger.error(f"[Report] Pydantic validation failed: {e} | data={parsed}")

    # ── Step 3: Fallback if parsing fails ────────────────────────────────────
    if report is None:
        logger.warning("[Report] Using fallback report (JSON parse failed)")
        report = InvestmentReport(
            ticker=_extract_ticker_heuristic(query),
            recommendation="HOLD",
            confidence=0.0,
            summary=(
                "Report generation encountered a parsing error. "
                f"Raw output: {raw_json[:300]}"
            ),
            key_risks=["Unable to parse structured report"],
            key_catalysts=["Retry with a more specific query"],
            target_price=None,
            sources=sources,
        )

    # ── Step 4: Generate narrative (streamed in graph.py via astream_events) ──
    narrative_prompt = NARRATIVE_PROMPT.format(
        query=query,
        report_json=report.model_dump_json(indent=2),
    )
    try:
        narrative = await llm_call(
            prompt=narrative_prompt,
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,        # slightly higher — narrative can be more fluent
            max_output_tokens=800,
        )
    except Exception as e:
        logger.warning(f"[Report] Narrative generation unavailable, using fallback: {e}")
        narrative = _build_fallback_narrative(report, e)

    logger.info("[Report] Complete")

    return {
        "final_report": report,
        "messages": state["messages"] + [
            {
                "role": "report_agent",
                "content": report.model_dump_json(),
                "narrative": narrative,
            }
        ],
    }


def _extract_ticker_heuristic(query: str) -> str:
    """Best-effort ticker extraction for the fallback report."""
    import re
    matches = re.findall(r'\b([A-Z]{1,5})\b', query)
    skip = {"I", "A", "THE", "FOR", "IS", "AI", "US", "BUY", "SELL", "HOLD"}
    for m in matches:
        if m not in skip and len(m) >= 2:
            return m
    return "UNKNOWN"
