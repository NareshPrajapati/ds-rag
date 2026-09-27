"""Vector store: chunk embeddings and semantic search backed by Chroma."""
import asyncio
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from app.config import get_settings

settings = get_settings()


class VectorStore:
    """Embeds chunks and stores them in a persistent Chroma collection.

    The embedding model is local for this demo; it can be replaced by OpenAI or
    Azure OpenAI embeddings. Chroma can likewise be replaced by a managed vector
    database such as Azure AI Search.
    """

    def __init__(self) -> None:
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.collection = self.client.get_or_create_collection(
            settings.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self.embedding_model.encode(
            texts, batch_size=64, normalize_embeddings=True
        ).tolist()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed, texts)

    async def has_document(self, file_hash: str) -> bool:
        existing = await asyncio.to_thread(
            self.collection.get, where={"file_hash": {"$eq": file_hash}}, limit=1
        )
        return bool(existing["ids"])

    async def add(
        self,
        chunk_ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        embeddings = await self.embed(texts)
        await asyncio.to_thread(
            self.collection.upsert,
            ids=chunk_ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    async def count_chunks(self) -> int:
        return await asyncio.to_thread(self.collection.count)

    async def count_documents(self) -> int:
        stored = await asyncio.to_thread(self.collection.get, include=["metadatas"])
        return len({m["document_id"] for m in stored["metadatas"] if m.get("document_id")})

    @staticmethod
    def _where(filters: dict[str, Any]) -> dict[str, Any] | None:
        clauses = [{key: {"$eq": value}} for key, value in filters.items() if value not in (None, "")]
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    async def search(
        self, query: str, filters: dict[str, Any], top_k: int
    ) -> list[dict[str, Any]]:
        count = await self.count_chunks()
        if count == 0:
            return []

        embedding = await self.embed([query])
        result = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=embedding,
            n_results=min(top_k, count),
            where=self._where(filters),
        )

        documents: list[dict[str, Any]] = []
        for chunk_id, text, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            documents.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "vector_score": round(1 - distance, 4),
                    **metadata,
                }
            )
        return documents
