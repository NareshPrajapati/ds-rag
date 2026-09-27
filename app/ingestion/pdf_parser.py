"""PDF parsing with LangChain's PyMuPDFParser and RecursiveCharacterTextSplitter."""
import re
from pathlib import Path
from typing import Any

from langchain_community.document_loaders.parsers import PyMuPDFParser
from langchain_core.document_loaders import Blob
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings

settings = get_settings()


class PDFParser:
    """Parse PDFs page-by-page and split them into overlapping chunks.

    Uses LangChain's PyMuPDFParser (tables rendered as Markdown) and the standard
    RecursiveCharacterTextSplitter. Tables are kept as whole chunks so their rows
    stay intact, and each chunk is prefixed with its document and page for better
    retrieval grounding. Swap PyMuPDFParser for a service such as Azure Document
    Intelligence when scanned documents or complex layouts appear.
    """

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ) -> None:
        self.table_max = chunk_size * 2  # split only unusually large tables
        try:
            # mode="page" keeps page boundaries; extract_tables keeps tables readable.
            self.parser = PyMuPDFParser(mode="page", extract_tables="markdown")
        except TypeError:
            self.parser = PyMuPDFParser()  # older versions already yield one document per page
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def parse(self, path: Path) -> list[dict[str, Any]]:
        blob = Blob.from_path(path)
        chunks: list[dict[str, Any]] = []
        for page in self.parser.lazy_parse(blob):
            page_number = int(page.metadata.get("page", 0)) + 1  # parser pages are 0-indexed
            header = f"Document: {path.name} | Page: {page_number}\n\n"
            for block in re.split(r"\n\s*\n", page.page_content):
                block = block.strip()
                if not block:
                    continue
                if self._is_markdown_table(block):
                    pieces = [block] if len(block) <= self.table_max else self.splitter.split_text(block)
                    block_type = "table"
                else:
                    pieces = self.splitter.split_text(block)
                    block_type = "text"
                for piece in pieces:
                    piece = piece.strip()
                    if piece:
                        chunks.append(
                            {
                                "text": header + piece,
                                "page": page_number,
                                "type": block_type,
                                "index": len(chunks),
                            }
                        )
        return chunks
    
    def _is_markdown_table(self, text: str) -> bool:
        return sum(1 for line in text.splitlines() if line.count("|") >= 2) >= 2
