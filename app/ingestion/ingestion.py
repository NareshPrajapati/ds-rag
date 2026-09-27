"""Ingestion service: parse PDFs and load their chunks into the vector store."""
import asyncio
from pathlib import Path
from typing import Any

from loguru import logger

from app.errors import DocumentParsingError
from app.ingestion.helpers import file_hash, metadata_from_filename
from app.ingestion.pdf_parser import PDFParser
from app.search.vector_store import VectorStore


class IngestionService:
    def __init__(self, vector_store: VectorStore, parser: PDFParser | None = None) -> None:
        self.vector_store = vector_store
        self.parser = parser or PDFParser()

    async def ingest_folder(self, folder: Path) -> dict[str, Any]:
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")

        pdf_files = sorted(folder.rglob("*.pdf"))
        ingested = skipped = failed = 0
        details: list[dict[str, Any]] = []

        for path in pdf_files:
            try:
                details.append(await self._ingest_one(path))
                status = details[-1]["status"]
                ingested += status == "indexed"
                skipped += status == "skipped"
            except Exception as exc:  # isolate failures so one bad file does not stop the batch
                failed += 1
                details.append({"filename": path.name, "status": "failed", "detail": str(exc)})
                logger.exception("Could not ingest {}", path.name)

        return {
            "pdfs_found": len(pdf_files),
            "ingested": ingested,
            "skipped": skipped,
            "failed": failed,
            "chunks_in_store": await self.vector_store.count_chunks(),
            "documents": details,
        }

    async def _ingest_one(self, path: Path) -> dict[str, Any]:
        hash_value = file_hash(path)
        if await self.vector_store.has_document(hash_value):
            return {"filename": path.name, "status": "skipped"}

        parsed = await asyncio.to_thread(self.parser.parse, path)
        if not parsed:
            raise DocumentParsingError(f"No text extracted from {path.name}; it may require OCR")

        document_id = hash_value[:16]
        file_metadata = metadata_from_filename(path)
        chunk_ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for chunk in parsed:
            chunk_ids.append(
                f"{document_id}-p{chunk['page']:03d}-{chunk['type'][0]}{chunk['index']:03d}"
            )
            texts.append(chunk["text"])
            metadatas.append(
                {
                    "document_id": document_id,
                    "filename": path.name,
                    "file_hash": hash_value,
                    "page": chunk["page"],
                    "chunk_type": chunk["type"],
                    "document_type": "quarterly financial report",
                    **file_metadata,
                }
            )

        await self.vector_store.add(chunk_ids, texts, metadatas)
        logger.info("Indexed {} with {} chunks", path.name, len(texts))
        return {"filename": path.name, "status": "indexed", "chunks": len(texts)}
