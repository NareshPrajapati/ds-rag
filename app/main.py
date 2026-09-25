"""FastAPI application entry point.

Wires configuration, the vector store, and the LangGraph agent into a small
service exposing two ingestion endpoints and one retrieval endpoint.

Run: ``uvicorn app.main:app --reload``
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from app import __version__
from app.config import get_settings
from app.dependencies import get_settings_dep, get_store
from app.schemas import HealthResponse
from app.services.embeddings import EmbeddingModel
from app.services.graph import build_rag_graph
from app.services.vector_store import RAGStore
from app.routers import ingestion, retrieval

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("financial-rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Loading embedding model: %s", settings.embedding_model)
    embedder = EmbeddingModel(settings.embedding_model)
    store = RAGStore(
        chroma_dir=settings.chroma_dir,
        collection_name=settings.collection_name,
        embedder=embedder,
        semantic_top_k=settings.semantic_top_k,
        keyword_top_k=settings.keyword_top_k,
        final_top_k=settings.final_top_k,
    )
    app.state.settings = settings
    app.state.store = store
    app.state.graph = build_rag_graph(store, settings)
    logger.info(
        "Startup complete | chunks=%d | openai=%s", store.count_chunks(), settings.openai_enabled
    )
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Financial Reports RAG API",
    description="Minimal, production-oriented RAG pipeline over quarterly financial PDFs.",
    version=__version__,
    lifespan=lifespan,
)

app.include_router(ingestion.router)
app.include_router(retrieval.router)


@app.get("/", tags=["health"], summary="Service metadata")
def root() -> dict[str, str]:
    return {"service": "Financial Reports RAG API", "version": __version__, "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["health"], summary="Liveness and index stats")
def health(store: RAGStore = Depends(get_store), settings=Depends(get_settings_dep)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        documents_indexed=store.count_documents(),
        chunks_indexed=store.count_chunks(),
        openai_enabled=settings.openai_enabled,
        embedding_model=settings.embedding_model,
    )
