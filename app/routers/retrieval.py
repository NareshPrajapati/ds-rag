"""Retrieval endpoint: LangGraph agent turns a question into a cited answer."""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException

from app.config import Settings
from app.dependencies import get_graph, get_settings_dep, get_store
from app.schemas import QueryRequest, QueryResponse, RAGAnswer, RetrievalDiagnostic
from app.services.vector_store import RAGStore

router = APIRouter(tags=["retrieval"])


@router.post("/query", response_model=QueryResponse, summary="Answer a question over the indexed reports")
def query(
    request: QueryRequest,
    graph=Depends(get_graph),
    store: RAGStore = Depends(get_store),
    settings: Settings = Depends(get_settings_dep),
) -> QueryResponse:
    if store.count_chunks() == 0:
        raise HTTPException(status_code=409, detail="No documents indexed yet. Call /ingest/folder or /ingest/files first.")

    started = time.perf_counter()
    initial_state = {
        "query": request.query,
        "session_id": request.session_id,
        "filters": {"company": request.company, "year": request.year, "quarter": request.quarter},
    }
    config = {"configurable": {"thread_id": request.session_id}}
    try:
        final_state = graph.invoke(initial_state, config=config)
    except Exception as exc:  # noqa: BLE001 - surface generation/tool failures cleanly
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

    documents = final_state.get("documents", [])
    diagnostics = [
        RetrievalDiagnostic(
            rank=rank,
            chunk_id=item["chunk_id"],
            document=item["filename"],
            company=item["company"],
            page=item["page"],
            rrf_score=item["rrf_score"],
            semantic_score=item.get("semantic_score"),
            bm25_score=item.get("bm25_score"),
            preview=item["text"][:220].replace("\n", " "),
        )
        for rank, item in enumerate(documents, 1)
    ]

    return QueryResponse(
        query=request.query,
        session_id=request.session_id,
        detected_filters={k: v for k, v in final_state.get("filters", {}).items() if v is not None},
        generation_mode=final_state.get("generation_mode", "extractive_fallback"),
        response=RAGAnswer.model_validate(final_state["answer"]),
        retrieval_diagnostics=diagnostics,
        latency_seconds=round(time.perf_counter() - started, 4),
    )
