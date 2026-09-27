from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from app.agent.agent import RAGAgent
from app.config import get_settings
from app.ingestion.ingestion import IngestionService
from app.ingestion.pdf_parser import PDFParser
from app.router import router
from app.search.reranker import Reranker
from app.search.search import SearchService
from app.search.vector_store import VectorStore

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models: embedding={} reranker={}", settings.embedding_model, settings.reranker_model)
    store = VectorStore()
    search_service = SearchService(store, Reranker())

    app.state.store = store
    app.state.ingestion = IngestionService(store, PDFParser())
    app.state.agent = RAGAgent(search_service)

    logger.info(
        "API ready | chunks={} | agent_enabled={}",
        await store.count_chunks(),
        app.state.agent.available,
    )
    yield


app = FastAPI(
    title="Financial Reports RAG API",
    description="RAG API for quarterly financial PDF reports.",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
