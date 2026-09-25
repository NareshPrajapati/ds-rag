"""PDF discovery, extraction, cleaning, and metadata-aware chunking.

Mirrors the notebook pipeline (Parts 1) so the API and notebook stay consistent.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import fitz

FILENAME_RE = re.compile(r"(?P<year>20\d{2})\s+Q(?P<quarter>[1-4])\s+(?P<company>[A-Z]+)", re.I)
TOKEN_RE = re.compile(r"[A-Za-z0-9$%.-]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_filename_metadata(path: Path) -> dict[str, Any]:
    match = FILENAME_RE.search(path.stem)
    return {
        "company": match.group("company").upper() if match else "UNKNOWN",
        "year": int(match.group("year")) if match else None,
        "quarter": f"Q{match.group('quarter')}" if match else "UNKNOWN",
        "reporting_period": f"{match.group('year')} Q{match.group('quarter')}" if match else "UNKNOWN",
        "document_type": "quarterly financial report",
    }


def discover_pdfs(folder: Path, recursive: bool = True) -> list[Path]:
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(folder.glob(pattern))


def extract_pages(path: Path) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    with fitz.open(path) as pdf:
        for page_index, page in enumerate(pdf):
            text = page.get_text("text", sort=True)
            tables: list[str] = []
            if hasattr(page, "find_tables"):
                try:
                    for table in page.find_tables().tables:
                        rows = table.extract()
                        tables.append("\n".join(" | ".join(str(cell or "") for cell in row) for row in rows))
                except Exception:
                    pass
            pages.append({"page": page_index + 1, "raw_text": text, "table_text": "\n\n".join(tables)})
    return pages


def _repeated_margin_lines(pages: list[dict[str, Any]], min_ratio: float = 0.35) -> set[str]:
    candidates: list[str] = []
    for page in pages:
        lines = [re.sub(r"\s+", " ", line).strip() for line in page["raw_text"].splitlines() if line.strip()]
        candidates.extend(line for line in (lines[:2] + lines[-2:]) if len(line) <= 120)
    counts = Counter(candidates)
    threshold = max(3, int(len(pages) * min_ratio))
    return {line for line, count in counts.items() if count >= threshold}


def clean_text(text: str, removable_lines: set[str]) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    lines = []
    for line in text.splitlines():
        normalized = re.sub(r"[ \t]+", " ", line).strip()
        if normalized and normalized not in removable_lines and not re.fullmatch(r"(?:Page\s+)?\d+", normalized, re.I):
            lines.append(normalized)
    text = "\n".join(lines)
    text = re.sub(r"(?<=\w)-\n(?=[a-z])", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def recursive_chunks(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text] if text.strip() else []
    chunks, start = [], 0
    separators = ["\n\n", "\n", ". ", "; ", " "]
    while start < len(text):
        target_end = min(start + size, len(text))
        end = target_end
        if target_end < len(text):
            floor = start + int(size * 0.6)
            boundary = max(text.rfind(separator, floor, target_end) for separator in separators)
            if boundary > start:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def process_pdf(path: Path, chunk_size: int, chunk_overlap: int) -> dict[str, Any]:
    """Return document metadata plus citation-ready chunks for one PDF."""
    file_hash = sha256_file(path)
    document_id = file_hash[:16]
    metadata = parse_filename_metadata(path)
    pages = extract_pages(path)
    removable = _repeated_margin_lines(pages)

    base_meta = {
        "document_id": document_id,
        "filename": path.name,
        "file_hash": file_hash,
        **metadata,
    }

    chunks: list[dict[str, Any]] = []
    for page in pages:
        combined = page["raw_text"] + ("\n\nTABLES\n" + page["table_text"] if page["table_text"] else "")
        cleaned = clean_text(combined, removable)
        for chunk_index, chunk_text in enumerate(recursive_chunks(cleaned, chunk_size, chunk_overlap)):
            chunk_id = f"{document_id}-p{page['page']:03d}-c{chunk_index:03d}"
            chunks.append({"chunk_id": chunk_id, "text": chunk_text, "page": page["page"], **base_meta})

    return {**base_meta, "pages": len(pages), "chunks": chunks}


def iter_documents(paths: Iterable[Path], chunk_size: int, chunk_overlap: int) -> Iterable[dict[str, Any]]:
    for path in paths:
        yield process_pdf(path, chunk_size, chunk_overlap)
