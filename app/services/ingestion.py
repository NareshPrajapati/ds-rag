"""Ingestion orchestration: folder scan and uploaded files -> vector store."""
from __future__ import annotations

import logging
from pathlib import Path

from app.config import Settings
from app.schemas import IngestedDocument, IngestResponse
from app.services import pdf_processing
from app.services.vector_store import RAGStore

logger = logging.getLogger("financial-rag.ingestion")


def ingest_paths(paths: list[Path], store: RAGStore, settings: Settings, skip_existing: bool = True) -> IngestResponse:
    documents: list[IngestedDocument] = []
    ingested = skipped = failed = 0

    for path in paths:
        try:
            file_hash = pdf_processing.sha256_file(path)
            if skip_existing and store.has_document(file_hash):
                meta = pdf_processing.parse_filename_metadata(path)
                documents.append(IngestedDocument(
                    document_id=file_hash[:16], filename=path.name, company=meta["company"],
                    reporting_period=meta["reporting_period"], pages=0, chunks=0,
                    status="skipped", detail="Already indexed (matching file hash).",
                ))
                skipped += 1
                continue

            document = pdf_processing.process_pdf(path, settings.chunk_size, settings.chunk_overlap)
            chunk_count = store.upsert_document(document)
            documents.append(IngestedDocument(
                document_id=document["document_id"], filename=document["filename"],
                company=document["company"], reporting_period=document["reporting_period"],
                pages=document["pages"], chunks=chunk_count, status="indexed",
            ))
            ingested += 1
        except Exception as exc:  # noqa: BLE001 - report per-file failures without aborting the batch
            logger.exception("Failed to ingest %s", path)
            documents.append(IngestedDocument(
                document_id="", filename=path.name, company="UNKNOWN", reporting_period="UNKNOWN",
                pages=0, chunks=0, status="failed", detail=str(exc),
            ))
            failed += 1

    store.finalize()
    return IngestResponse(
        ingested=ingested, skipped=skipped, failed=failed,
        total_chunks_in_store=store.count_chunks(), documents=documents,
    )


def ingest_folder(folder: Path, recursive: bool, store: RAGStore, settings: Settings) -> IngestResponse:
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    paths = pdf_processing.discover_pdfs(folder, recursive=recursive)
    logger.info("Discovered %d PDF(s) under %s", len(paths), folder)
    return ingest_paths(paths, store, settings)
