"""Helpers shared by the ingestion pipeline."""
import hashlib
import re
from pathlib import Path
from typing import Any

FILE_PATTERN = re.compile(
    r"(?P<year>20\d{2})\s+Q(?P<quarter>[1-4])\s+(?P<company>[A-Z]+)", re.I
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata_from_filename(path: Path) -> dict[str, Any]:
    match = FILE_PATTERN.search(path.stem)
    if not match:
        return {"company": "UNKNOWN", "year": 0, "quarter": "UNKNOWN", "reporting_period": "UNKNOWN"}
    return {
        "company": match.group("company").upper(),
        "year": int(match.group("year")),
        "quarter": f"Q{match.group('quarter')}",
        "reporting_period": f"{match.group('year')} Q{match.group('quarter')}",
    }
