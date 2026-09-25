"""Centralised, environment-driven configuration."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # Paths
    data_dir: Path = Path("data")
    chroma_dir: Path = Path("artifacts/chroma")
    embedding_cache_dir: Path = Path("artifacts/embeddings")
    collection_name: str = "financial_reports_v1"

    # Models
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_model: str = "gpt-4.1-mini"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Retrieval
    semantic_top_k: int = 12
    keyword_top_k: int = 12
    final_top_k: int = 6

    @property
    def openai_enabled(self) -> bool:
        import os

        return bool(os.getenv("OPENAI_API_KEY"))

    def ensure_dirs(self) -> None:
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
