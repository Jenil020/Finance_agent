"""
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
                yield f"data: {output['stream_token']}\n\n"
        yield "data: [DONE]\n\n"
