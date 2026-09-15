from __future__ import annotations

import hashlib
from pathlib import Path

from .types import DocumentFile, LoadedSection

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


def discover_documents(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Document directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Document path is not a directory: {root}")
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_document(path: Path, root: Path) -> DocumentFile:
    raw = path.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    suffix = path.suffix.lower()

    if suffix in {".md", ".txt"}:
        sections = [LoadedSection(path.read_text(encoding="utf-8"))]
    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "PDF support requires pypdf. Run: pip install -r requirements.txt"
            ) from exc

        reader = PdfReader(path)
        sections = [
            LoadedSection(page.extract_text() or "", locator=f"page {number}")
            for number, page in enumerate(reader.pages, start=1)
        ]
    else:
        raise ValueError(f"Unsupported document type: {path.suffix}")

    return DocumentFile(
        path=path,
        relative_path=path.relative_to(root).as_posix(),
        checksum=checksum,
        sections=sections,
    )
