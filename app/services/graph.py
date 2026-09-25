"""Agentic retrieval workflow built with LangGraph (senior-track, Part 3).

The graph turns a raw user question into a grounded, cited answer:

    START -> analyze_query -> retrieve (vector-DB tool) -> generate -> END

* ``analyze_query`` extracts structured filters (company / year / quarter) from
  the natural-language question and merges them with any explicit API filters.
* ``retrieve`` invokes a LangChain ``search_financial_reports`` tool that queries
  the hybrid vector store.
* ``generate`` produces a schema-validated, citation-audited answer.

A ``MemorySaver`` checkpointer keyed by ``session_id`` gives the graph
conversational memory across turns.
"""
from __future__ import annotations

import re
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.config import Settings
from app.services.generation import generate_answer
from app.services.vector_store import RAGStore

COMPANY_ALIASES = {
    "apple": "AAPL", "aapl": "AAPL",
    "amazon": "AMZN", "amzn": "AMZN",
    "intel": "INTC", "intc": "INTC",
    "microsoft": "MSFT", "msft": "MSFT",
    "nvidia": "NVDA", "nvda": "NVDA",
}


class RAGState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    query: str
    filters: dict[str, Any]
    documents: list[dict[str, Any]]
    answer: dict[str, Any]
    generation_mode: str
    session_id: str


def _extract_filters(query: str, explicit: dict[str, Any]) -> dict[str, Any]:
    filters: dict[str, Any] = {k: v for k, v in explicit.items() if v is not None}
    lowered = query.lower()
    if "company" not in filters:
        for alias, ticker in COMPANY_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", lowered):
                filters["company"] = ticker
                break
    if "quarter" not in filters:
        quarter = re.search(r"\bq([1-4])\b", lowered)
        if quarter:
            filters["quarter"] = f"Q{quarter.group(1)}"
    if "year" not in filters:
        year = re.search(r"\b(20\d{2})\b", query)
        if year:
            filters["year"] = int(year.group(1))
    return filters


def build_rag_graph(store: RAGStore, settings: Settings):
    """Compile and return the LangGraph agent with conversational memory."""

    def _search_tool(query: str, company: str | None = None, year: int | None = None, quarter: str | None = None) -> list[dict[str, Any]]:
        return store.hybrid_search(query, {"company": company, "year": year, "quarter": quarter})

    search_financial_reports = StructuredTool.from_function(
        func=_search_tool,
        name="search_financial_reports",
        description="Retrieve the most relevant financial-report chunks for a query, optionally filtered by company ticker, year, and quarter.",
    )

    def analyze_query(state: RAGState) -> RAGState:
        query = state["query"]
        filters = _extract_filters(query, state.get("filters", {}))
        return {"filters": filters, "messages": [HumanMessage(content=query)]}

    def retrieve(state: RAGState) -> RAGState:
        filters = state.get("filters", {})
        documents = search_financial_reports.invoke({
            "query": state["query"],
            "company": filters.get("company"),
            "year": filters.get("year"),
            "quarter": filters.get("quarter"),
        })
        return {"documents": documents}

    def generate(state: RAGState) -> RAGState:
        answer, mode = generate_answer(state["query"], state.get("documents", []), settings.openai_model)
        return {
            "answer": answer.model_dump(),
            "generation_mode": mode,
            "messages": [AIMessage(content=answer.answer)],
        }

    graph = StateGraph(RAGState)
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "analyze_query")
    graph.add_edge("analyze_query", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile(checkpointer=MemorySaver())
