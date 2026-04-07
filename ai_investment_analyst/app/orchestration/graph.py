"""
LangGraph stateful orchestration graph.

Graph topology:
  [research] → route_after_research() → [analysis] → [report] → END
                                      ↘ [report] → END          (no portfolio)
                                      ↘ END                     (error)

Key implementation details:
  - AsyncSqliteSaver checkpointer gives every session its own state thread.
    Passing thread_id = session_id means the graph can resume mid-run
    if interrupted (human-in-the-loop ready).

  - astream_events(version='v2') is used instead of astream() because it
    emits fine-grained events (node start/end) that we convert to SSE frames.
    Each SSE frame is a JSON line the frontend can parse incrementally.

  - run_agent_stream() is an async generator consumed by FastAPI's
    StreamingResponse. It yields SSE-formatted strings.

SSE event format (each line sent to client):
  data: {"type": "progress", "node": "research", "status": "started"}\n\n
  data: {"type": "progress", "node": "research", "status": "done"}\n\n
  data: {"type": "result",   "report": {...}}\n\n
  data: [DONE]\n\n
"""
import json
from typing import AsyncGenerator
from langgraph.graph import StateGraph, END

from app.orchestration.state import AgentState
from app.orchestration.checkpointer import get_checkpointer
from app.orchestration.router import route_after_research
from app.agents.research import research_agent_node
from app.agents.analysis import analysis_agent_node
from app.agents.report import report_agent_node
from app.core.logging import logger

# ── Node names (single source of truth) ──────────────────────────────────────
RESEARCH = "research"
ANALYSIS = "analysis"
REPORT   = "report"

# ── Agent nodes we emit SSE progress events for ───────────────────────────────
_TRACKED_NODES = {RESEARCH, ANALYSIS, REPORT}


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Build and compile the investment analyst LangGraph.
    Called once at startup; the compiled graph is reused for all requests.
    """
    workflow = StateGraph(AgentState)

    # Register agent nodes
    workflow.add_node(RESEARCH, research_agent_node)
    workflow.add_node(ANALYSIS, analysis_agent_node)
    workflow.add_node(REPORT,   report_agent_node)

    # Entry point is always Research
    workflow.set_entry_point(RESEARCH)

    # After Research: conditional route (analysis | report | end)
    workflow.add_conditional_edges(
        RESEARCH,
        route_after_research,
        {
            "analysis": ANALYSIS,
            "report":   REPORT,
            "end":      END,
        },
    )

    # Fixed edges
    workflow.add_edge(ANALYSIS, REPORT)
    workflow.add_edge(REPORT,   END)

    # Compile with async checkpointer (wired in at request time via get_checkpointer)
    return workflow.compile(checkpointer=get_checkpointer())


# ── Singleton compiled graph ──────────────────────────────────────────────────
# Initialised lazily on first request (checkpointer must be ready first).

_graph: StateGraph | None = None


def get_graph() -> StateGraph:
    """Return the compiled singleton graph. Builds it on first call."""
    global _graph
    if _graph is None:
        logger.info("[Graph] Compiling LangGraph workflow")
        _graph = build_graph()
        logger.info("[Graph] Compiled OK")
    return _graph


# ── SSE helpers ───────────────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    """Format a dict as a Server-Sent Events data frame."""
    return f"data: {json.dumps(payload)}\n\n"


def _node_display_name(node: str) -> str:
    return {
        RESEARCH: "Research Agent",
        ANALYSIS: "Analysis Agent",
        REPORT:   "Report Agent",
    }.get(node, node)


# ── Public streaming interface ────────────────────────────────────────────────

async def run_agent_stream(
    query: str,
    session_id: str,
    portfolio=None,
) -> AsyncGenerator[str, None]:
    """
    Run the full multi-agent graph and yield SSE-formatted strings.

    Each session_id maps to its own LangGraph checkpoint thread, so state
    is persisted across calls and the graph can resume if interrupted.

    Yields:
        SSE data frames (strings ending in \\n\\n) that FastAPI's
        StreamingResponse sends directly to the client.

    SSE event types:
        {"type": "progress", "node": "...", "status": "started"|"done"}
        {"type": "result",   "report": {...}, "narrative": "..."}
        {"type": "error",    "message": "..."}
        "data: [DONE]"
    """
    graph = get_graph()

    initial_state: AgentState = {
        "query":            query,
        "session_id":       session_id,
        "portfolio":        portfolio or [],
        "messages":         [],
        "research_results": [],
        "analysis_results": {},
        "final_report":     None,
        "error":            None,
    }

    # thread_id ties this run to a persistent checkpoint slot
    config = {"configurable": {"thread_id": session_id}}

    logger.info(f"[Graph] Run started | session={session_id} | query='{query[:60]}'")

    try:
        async for event in graph.astream_events(
            initial_state, config=config, version="v2"
        ):
            event_type = event["event"]
            node_name  = event.get("name", "")

            # ── Node started → emit progress ──────────────────────────────
            if event_type == "on_chain_start" and node_name in _TRACKED_NODES:
                logger.info(f"[Graph] Node started: {node_name}")
                yield _sse({
                    "type":   "progress",
                    "node":   node_name,
                    "label":  _node_display_name(node_name),
                    "status": "started",
                })

            # ── Node finished → emit progress + extract final report ───────
            elif event_type == "on_chain_end" and node_name in _TRACKED_NODES:
                logger.info(f"[Graph] Node done: {node_name}")
                yield _sse({
                    "type":   "progress",
                    "node":   node_name,
                    "label":  _node_display_name(node_name),
                    "status": "done",
                })

                # If Report Agent just finished, emit the structured result
                if node_name == REPORT:
                    output = event.get("data", {}).get("output", {})
                    final_report = output.get("final_report")
                    messages     = output.get("messages", [])

                    # Extract narrative from last report_agent message
                    narrative = ""
                    for msg in reversed(messages):
                        if msg.get("role") == "report_agent" and msg.get("narrative"):
                            narrative = msg["narrative"]
                            break

                    report_dict = (
                        final_report.model_dump()
                        if hasattr(final_report, "model_dump")
                        else final_report or {}
                    )

                    yield _sse({
                        "type":      "result",
                        "report":    report_dict,
                        "narrative": narrative,
                    })

        logger.info(f"[Graph] Run complete | session={session_id}")

    except Exception as e:
        logger.error(f"[Graph] Stream error | session={session_id} | {e}", exc_info=True)
        yield _sse({"type": "error", "message": str(e)})

    finally:
        yield "data: [DONE]\n\n"