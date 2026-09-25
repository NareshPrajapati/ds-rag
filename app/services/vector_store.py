"""Persistent vector store + BM25 hybrid retriever (Parts 2 of the assignment).

Wraps a Chroma collection for dense retrieval and an in-memory BM25 index for
keyword retrieval, fusing them with Reciprocal Rank Fusion. Designed as a
process-wide singleton created at application startup.
"""
from __future__ import annotations

import logging
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

from app.services.embeddings import EmbeddingModel
from app.services.pdf_processing import tokenize

logger = logging.getLogger("financial-rag.store")

_METADATA_COLUMNS = [
    "document_id", "filename", "company", "year", "quarter",
    "reporting_period", "document_type", "file_hash", "page",
]


class RAGStore:
    def __init__(
        self,
        chroma_dir: Path,
        collection_name: str,
        embedder: EmbeddingModel,
        semantic_top_k: int,
        keyword_top_k: int,
        final_top_k: int,
    ) -> None:
        self._lock = threading.Lock()
        self.embedder = embedder
        self.semantic_top_k = semantic_top_k
        self.keyword_top_k = keyword_top_k
        self.final_top_k = final_top_k

        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = self.client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": embedder.model_name},
        )

        self._chunk_ids: list[str] = []
        self._chunk_texts: list[str] = []
        self._chunk_meta: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None
        self._rebuild_keyword_index()

    # ---------- indexing ----------
    def _rebuild_keyword_index(self) -> None:
        stored = self.collection.get(include=["documents", "metadatas"])
        self._chunk_ids = stored.get("ids", []) or []
        self._chunk_texts = stored.get("documents", []) or []
        self._chunk_meta = stored.get("metadatas", []) or []
        self._bm25 = BM25Okapi([tokenize(text) for text in self._chunk_texts]) if self._chunk_texts else None
        logger.info("Keyword index rebuilt over %d chunks", len(self._chunk_ids))

    def upsert_document(self, document: dict[str, Any]) -> int:
        """Embed and upsert one document's chunks. Returns chunk count."""
        chunks = document["chunks"]
        if not chunks:
            return 0
        with self._lock:
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedder.encode(texts)
            self.collection.upsert(
                ids=[chunk["chunk_id"] for chunk in chunks],
                documents=texts,
                metadatas=[{key: chunk[key] for key in _METADATA_COLUMNS} for chunk in chunks],
                embeddings=embeddings.tolist(),
            )
        return len(chunks)

    def finalize(self) -> None:
        """Rebuild the keyword index after a batch of upserts."""
        with self._lock:
            self._rebuild_keyword_index()

    def has_document(self, file_hash: str) -> bool:
        try:
            existing = self.collection.get(where={"file_hash": {"$eq": file_hash}}, limit=1)
            return bool(existing.get("ids"))
        except Exception:
            return False

    # ---------- stats ----------
    def count_chunks(self) -> int:
        return self.collection.count()

    def count_documents(self) -> int:
        return len({meta.get("document_id") for meta in self._chunk_meta}) if self._chunk_meta else 0

    def companies(self) -> list[str]:
        return sorted({str(meta.get("company")) for meta in self._chunk_meta if meta.get("company")})

    # ---------- retrieval ----------
    @staticmethod
    def _build_where(filters: dict[str, Any] | None) -> dict[str, Any] | None:
        if not filters:
            return None
        clauses = [{key: {"$eq": value}} for key, value in filters.items() if value is not None]
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def hybrid_search(
        self, query: str, filters: dict[str, Any] | None = None, final_k: int | None = None
    ) -> list[dict[str, Any]]:
        if not self._chunk_ids:
            return []
        final_k = final_k or self.final_top_k
        clean_filters = {k: v for k, v in (filters or {}).items() if v is not None}

        index_by_id = {chunk_id: i for i, chunk_id in enumerate(self._chunk_ids)}
        allowed = {
            chunk_id
            for chunk_id, meta in zip(self._chunk_ids, self._chunk_meta)
            if all(str(meta.get(key, "")).upper() == str(value).upper() for key, value in clean_filters.items())
        }
        if not allowed:
            return []

        # Dense retrieval
        where = self._build_where(clean_filters)
        semantic = self.collection.query(
            query_embeddings=[self.embedder.encode_one(query)],
            n_results=min(self.semantic_top_k, len(allowed)),
            where=where,
        )
        semantic_ids = semantic["ids"][0]
        semantic_scores = {cid: 1 - dist for cid, dist in zip(semantic_ids, semantic["distances"][0])}

        # Keyword retrieval
        keyword_ids: list[str] = []
        keyword_scores: dict[str, float] = {}
        if self._bm25 is not None:
            raw = np.asarray(self._bm25.get_scores(tokenize(query)))
            eligible = [index_by_id[cid] for cid in allowed]
            ranked = sorted(eligible, key=lambda i: raw[i], reverse=True)[: self.keyword_top_k]
            keyword_ids = [self._chunk_ids[i] for i in ranked]
            keyword_scores = {self._chunk_ids[i]: float(raw[i]) for i in ranked}

        # Reciprocal rank fusion
        fused: dict[str, float] = defaultdict(float)
        for rank, cid in enumerate(semantic_ids, 1):
            fused[cid] += 1 / (60 + rank)
        for rank, cid in enumerate(keyword_ids, 1):
            fused[cid] += 1 / (60 + rank)

        results: list[dict[str, Any]] = []
        for cid, rrf in sorted(fused.items(), key=lambda item: item[1], reverse=True):
            if cid not in index_by_id:
                continue
            i = index_by_id[cid]
            meta = self._chunk_meta[i]
            results.append({
                "chunk_id": cid,
                "text": self._chunk_texts[i],
                "rrf_score": rrf,
                "semantic_score": semantic_scores.get(cid),
                "bm25_score": keyword_scores.get(cid),
                **{key: meta.get(key) for key in _METADATA_COLUMNS},
            })
        return self._deduplicate(results, final_k)

    @staticmethod
    def _deduplicate(results: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for result in results:
            result_tokens = set(tokenize(result["text"]))
            duplicate = any(
                result["document_id"] == prior["document_id"]
                and result["page"] == prior["page"]
                and len(result_tokens & set(tokenize(prior["text"]))) / max(1, len(result_tokens)) > 0.8
                for prior in selected
            )
            if not duplicate:
                selected.append(result)
            if len(selected) == limit:
                break
        return selected
