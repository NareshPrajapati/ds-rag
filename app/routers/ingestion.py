"""Ingestion endpoints (2): folder-scan and direct file upload."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.config import Settings
from app.dependencies import get_settings_dep, get_store
from app.schemas import FolderIngestRequest, IngestResponse
from app.services import ingestion
from app.services.vector_store import RAGStore

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/folder", response_model=IngestResponse, summary="Ingest all PDFs from a folder")
def ingest_from_folder(
    request: FolderIngestRequest,
    store: RAGStore = Depends(get_store),
    settings: Settings = Depends(get_settings_dep),
) -> IngestResponse:
    folder = Path(request.folder) if request.folder else settings.data_dir
    try:
        return ingestion.ingest_folder(folder, request.recursive, store, settings)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/files", response_model=IngestResponse, summary="Ingest one or more uploaded PDF files")
async def ingest_from_files(
    files: list[UploadFile] = File(..., description="PDF files to ingest."),
    store: RAGStore = Depends(get_store),
    settings: Settings = Depends(get_settings_dep),
) -> IngestResponse:
    saved: list[Path] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="rag_upload_"))
    try:
        for upload in files:
            if not (upload.filename or "").lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail=f"Only PDF files are supported: {upload.filename}")
            target = tmp_dir / Path(upload.filename).name
            with target.open("wb") as buffer:
                shutil.copyfileobj(upload.file, buffer)
            saved.append(target)
        return ingestion.ingest_paths(saved, store, settings)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
