from pydantic import BaseModel, Field

from app.config import get_settings

settings = get_settings()


class IngestRequest(BaseModel):
    folder: str = Field(default=str(settings.data_dir), description="Folder containing PDF files")


class QueryRequest(BaseModel):
    query: str = Field(min_length=3)
    session_id: str = Field(default="default", min_length=1)
    company: str | None = None
    year: int | None = None
    quarter: str | None = None
    top_k: int = Field(default=settings.top_k, ge=1, le=20)


class Source(BaseModel):
    document_name: str
    page: int
    chunk_id: str
    quote: str = Field(max_length=500)


class Answer(BaseModel):
    answer: str
    supporting_evidence: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    limitations: list[str] = Field(default_factory=list)
