"""Pydantic request/response schemas shared across the API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------- Ingestion ----------
class FolderIngestRequest(BaseModel):
    folder: str | None = Field(default=None, description="Folder path to ingest. Defaults to the configured data directory.")
    recursive: bool = Field(default=True, description="Recurse into sub-directories.")


class IngestedDocument(BaseModel):
    document_id: str
    filename: str
    company: str
    reporting_period: str
    pages: int
    chunks: int
    status: Literal["indexed", "skipped", "failed"]
    detail: str | None = None


class IngestResponse(BaseModel):
    ingested: int
    skipped: int
    failed: int
    total_chunks_in_store: int
    documents: list[IngestedDocument]


# ---------- Retrieval ----------
class QueryRequest(BaseModel):
    query: str = Field(min_length=3, description="Natural-language question about the financial reports.")
    session_id: str = Field(default="default", description="Conversation id; enables multi-turn memory.")
    company: str | None = Field(default=None, description="Optional ticker filter, e.g. MSFT.")
    year: int | None = Field(default=None, description="Optional reporting year filter, e.g. 2023.")
    quarter: str | None = Field(default=None, description="Optional quarter filter, e.g. Q3.")
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceCitation(BaseModel):
    document_name: str
    page: int = Field(ge=1)
    chunk_id: str
    quote: str = Field(min_length=1, max_length=500)


class RAGAnswer(BaseModel):
    answer: str
    supporting_evidence: list[str] = Field(default_factory=list)
    sources: list[SourceCitation] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    limitations: list[str] = Field(default_factory=list)


class RetrievalDiagnostic(BaseModel):
    rank: int
    chunk_id: str
    document: str
    company: str
    page: int
    rrf_score: float
    semantic_score: float | None = None
    bm25_score: float | None = None
    preview: str


class QueryResponse(BaseModel):
    query: str
    session_id: str
    detected_filters: dict[str, Any]
    generation_mode: Literal["openai", "extractive_fallback"]
    response: RAGAnswer
    retrieval_diagnostics: list[RetrievalDiagnostic]
    latency_seconds: float


# ---------- Health ----------
class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    documents_indexed: int
    chunks_indexed: int
    openai_enabled: bool
    embedding_model: str
