"""Cross-encoder reranking using a second, HuggingFace model."""
import asyncio
from typing import Any

from sentence_transformers import CrossEncoder

from app.config import get_settings

settings = get_settings()


class Reranker:
    """Reorders candidate chunks by relevance with a HuggingFace cross-encoder.

    The bi-encoder in the vector store is fast but approximate; this second model
    scores each (query, chunk) pair directly for sharper ordering.
    """

    def __init__(self) -> None:
        self.model = CrossEncoder(settings.reranker_model)

    def _score(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(score) for score in self.model.predict(pairs)]

    async def rerank(
        self, query: str, documents: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        if not documents:
            return []

        pairs = [(query, document["text"]) for document in documents]
        scores = await asyncio.to_thread(self._score, pairs)
        for document, score in zip(documents, scores):
            document["rerank_score"] = round(score, 4)

        ranked = sorted(documents, key=lambda item: item["rerank_score"], reverse=True)[:top_k]
        for rank, document in enumerate(ranked, start=1):
            document["rank"] = rank
        return ranked
