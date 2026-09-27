import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    data_dir: Path = Field(default=Path("data"))
    chroma_dir: Path = Field(default=Path("artifacts/chroma"))

    # Two models: one embeds chunks for the vector store, a separate HuggingFace
    # cross-encoder reranks the retrieved candidates.
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    openai_model: str = Field(default="gpt-4.1-mini")
    openai_api_key: str | None = Field(default="123")

    collection_name: str = Field(default="financial_reports")
    chunk_size: int = Field(default=1_000, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)

    # Retrieve a wider candidate set from the vector store, then rerank down to top_k.
    retrieve_candidates: int = Field(default=20, gt=0)
    top_k: int = Field(default=6, gt=0)

    @property
    def openai_enabled(self) -> bool:
        return bool(self.openai_api_key)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    # Make the key available to the OpenAI client when it is only set in .env.
    if settings.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    return settings
