"""API endpoints: ingestion, query and health."""
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from app.config import get_settings
from app.errors import AgentError, DocumentParsingError, LLMNotConfiguredError
from app.schemas import IngestRequest, QueryRequest

settings = get_settings()
router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    store = request.app.state.store
    return {
        "status": "ok",
        "documents_indexed": await store.count_documents(),
        "chunks_indexed": await store.count_chunks(),
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "agent_enabled": request.app.state.agent.available,
        "openai_enabled": settings.openai_enabled,
    }


@router.post("/ingest")
async def ingest(request: Request, body: IngestRequest) -> dict[str, Any]:
    try:
        return await request.app.state.ingestion.ingest_folder(Path(body.folder))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DocumentParsingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/query")
async def query(request: Request, body: QueryRequest) -> dict[str, Any]:
    store = request.app.state.store
    agent = request.app.state.agent

    if await store.count_chunks() == 0:
        raise HTTPException(status_code=409, detail="Run POST /ingest before querying")
    if not agent.available:
        raise HTTPException(status_code=503, detail="Set OPENAI_API_KEY to enable the retrieval agent")

    filters = {
        "company": body.company.upper() if body.company else None,
        "year": body.year,
        "quarter": body.quarter.upper() if body.quarter else None,
    }
    filters = {key: value for key, value in filters.items() if value is not None}

    started = time.perf_counter()
    try:
        answer, documents = await agent.answer(body.query, filters, body.session_id)
    except LLMNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AgentError:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail="Query processing failed")

    return {
        "query": body.query,
        "session_id": body.session_id,
        "requested_filters": filters,
        "response": answer.model_dump(),
        "retrieval_diagnostics": [
            {
                "rank": document.get("rank"),
                "document": document.get("filename"),
                "page": document.get("page"),
                "chunk_id": document.get("chunk_id"),
                "chunk_type": document.get("chunk_type"),
                "vector_score": document.get("vector_score"),
                "rerank_score": document.get("rerank_score"),
                "preview": document.get("text", "")[:220].replace("\n", " "),
            }
            for document in documents
        ],
        "latency_seconds": round(time.perf_counter() - started, 4),
    }
