"""Retrieval agent built with LangChain's create_agent.

The agent receives a role-based conversation (system + user/ai messages), calls
the `search_financial_reports` tool, and returns a schema-validated answer. An
in-process checkpointer gives it per-session memory.
"""
import json
import re
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from loguru import logger

from app.config import get_settings
from app.errors import AgentError, LLMNotConfiguredError
from app.prompts.prompts import build_user_message, get_prompt
from app.schemas import Answer, Source
from app.search.search import SearchService

settings = get_settings()


class RAGAgent:
    def __init__(self, search_service: SearchService) -> None:
        self.search_service = search_service
        # Built only when an LLM is configured; the tool captures the search service.
        self._agent = self._build_agent() if settings.openai_enabled else None

    @property
    def available(self) -> bool:
        return self._agent is not None

    def _build_agent(self) -> Any:
        model = ChatOpenAI(model=settings.openai_model, temperature=0)
        # MemorySaver is in-process; use a Postgres/Redis checkpointer in production.
        return create_agent(
            model,
            tools=[self._make_search_tool()],
            system_prompt=get_prompt("system"),
            response_format=Answer,
            checkpointer=InMemorySaver(),
        )

    def _make_search_tool(self):
        search_service = self.search_service

        @tool
        async def search_financial_reports(
            query: str,
            company: str | None = None,
            year: int | None = None,
            quarter: str | None = None,
        ) -> str:
            """Search the financial reports. Provide company ticker, year and quarter when known."""
            documents = await search_service.search(
                query, {"company": company, "year": year, "quarter": quarter}, settings.top_k
            )
            return json.dumps(documents)

        return search_financial_reports

    async def answer(
        self, query: str, filters: dict[str, Any], session_id: str
    ) -> tuple[Answer, list[dict[str, Any]]]:
        if self._agent is None:
            raise LLMNotConfiguredError("Set OPENAI_API_KEY to enable the retrieval agent.")

        payload = {"messages": [{"role": "user", "content": build_user_message(query, filters)}]}
        config = {"configurable": {"thread_id": session_id}}
        try:
            result = await self._agent.ainvoke(payload, config=config)
        except Exception as exc:
            logger.exception("Agent invocation failed")
            raise AgentError(str(exc)) from exc

        answer = result.get("structured_response")
        if not isinstance(answer, Answer):
            raise AgentError("The agent did not return a structured answer.")

        documents = self._last_tool_documents(result.get("messages", []))
        return self._audit_citations(answer, documents), documents

    @staticmethod
    def _last_tool_documents(messages: list[Any]) -> list[dict[str, Any]]:
        for message in reversed(messages):
            if isinstance(message, ToolMessage):
                try:
                    return json.loads(message.content)
                except (TypeError, json.JSONDecodeError):
                    return []
        return []

    @staticmethod
    def _audit_citations(answer: Answer, documents: list[dict[str, Any]]) -> Answer:
        retrieved = {document["chunk_id"]: document for document in documents}
        valid: list[Source] = []
        for source in answer.sources:
            document = retrieved.get(source.chunk_id)
            if not document:
                continue
            quote = re.sub(r"\s+", " ", source.quote).lower()
            text = re.sub(r"\s+", " ", document.get("text", "")).lower()
            if source.document_name == document.get("filename") and source.page == document.get("page") and quote in text:
                valid.append(source)

        if len(valid) != len(answer.sources):
            answer.limitations.append("One or more invalid citations were removed.")
            answer.confidence = min(answer.confidence, 0.5)
            answer.sources = valid
        return answer
