"""
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
    prompt += f"\n\nRAG Results:\n{rag_results}"
    prompt += f"\n\nWeb Search Results:\n{search_results}"

    synthesis = await llm_call_with_retry(model, prompt)

    return {
        **state,
        "research_results": [synthesis],
        "messages": state["messages"] + [
            {"role": "research_agent", "content": synthesis}
        ],
    }
