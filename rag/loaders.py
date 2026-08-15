"""Document loading.

Deliberately dependency-light: plain text and Markdown are read directly, PDFs go
through `pypdf`. Each loader returns LangChain `Document`s tagged with a stable,
repo-relative `source` so citations point at something a human can open.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES


def _title_from(path: Path) -> str:
    return path.stem.replace("-", " ").replace("_", " ").title()


def load_text(path: Path) -> list[Document]:
    return [Document(page_content=path.read_text(encoding="utf-8"))]


def load_pdf(path: Path) -> list[Document]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    docs: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"page": page_number}))
    return docs


def load_file(path: Path, docs_dir: Path) -> list[Document]:
    """Load one file and stamp source metadata onto every resulting Document."""
    suffix = path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        docs = load_pdf(path)
    elif suffix in TEXT_SUFFIXES:
        docs = load_text(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    source = path.relative_to(docs_dir).as_posix()
    title = _title_from(path)
    for doc in docs:
        doc.metadata["source"] = source
        doc.metadata["title"] = title
    return docs


def discover_files(docs_dir: Path) -> tuple[list[Path], list[Path]]:
    """Return (supported, skipped) files found under `docs_dir`."""
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory does not exist: {docs_dir}")

    supported: list[Path] = []
    skipped: list[Path] = []
    for path in sorted(p for p in docs_dir.rglob("*") if p.is_file()):
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            supported.append(path)
        else:
            skipped.append(path)
    return supported, skipped
