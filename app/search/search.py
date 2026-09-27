"""Search service: semantic retrieval followed by cross-encoder reranking."""
from typing import Any

from app.config import get_settings
from app.search.reranker import Reranker
from app.search.vector_store import VectorStore

settings = get_settings()


class SearchService:
    def __init__(self, vector_store: VectorStore, reranker: Reranker) -> None:
        self.vector_store = vector_store
        self.reranker = reranker

    async def search(
        self, query: str, filters: dict[str, Any] | None = None, top_k: int = settings.top_k
    ) -> list[dict[str, Any]]:
        candidates = await self.vector_store.search(query, filters or {}, settings.retrieve_candidates)
        return await self.reranker.rerank(query, candidates, top_k)
