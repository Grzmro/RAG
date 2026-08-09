"""Document ingestion: load -> chunk -> embed -> persist in Chroma.

Chunk IDs are deterministic (`<relative-path>::<index>`), so re-running
ingestion upserts instead of duplicating.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import Settings
from rag.loaders import SUPPORTED_SUFFIXES, discover_files, load_file
from rag.store import collection_size, get_vectorstore


@dataclass
class IngestReport:
    files: int
    chunks: int
    skipped: list[Path]
    collection_count: int
    failed: list[tuple[Path, str]] = field(default_factory=list)


def chunk_documents(docs: list[Document], settings: Settings) -> list[Document]:
    """Split documents and attach stable per-chunk identifiers."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)

    per_source_index: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        index = per_source_index.get(source, 0)
        per_source_index[source] = index + 1
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_id"] = f"{source}::{index}"
        chunk.metadata["content_sha1"] = hashlib.sha1(
            chunk.page_content.encode("utf-8")
        ).hexdigest()[:12]
    return chunks


def load_and_chunk(
    files: list[Path], settings: Settings
) -> tuple[list[Document], list[Path], list[tuple[Path, str]]]:
    """Load and split every file, tolerating individual failures.

    One unreadable file (corrupt PDF, bad encoding) must not abort the run —
    index what we can and report the rest. Returns (chunks, ingested, failed).
    """
    chunks: list[Document] = []
    ingested: list[Path] = []
    failed: list[tuple[Path, str]] = []

    for path in files:
        try:
            docs = load_file(path, settings.docs_dir)
        except Exception as exc:  # noqa: BLE001 - loader backends raise their own types
            failed.append((path, f"{type(exc).__name__}: {exc}"))
            continue
        chunks.extend(chunk_documents(docs, settings))
        ingested.append(path)

    return chunks, ingested, failed


def ingest(settings: Settings, reset: bool = False) -> IngestReport:
    """Run the full ingestion pipeline over `settings.docs_dir`."""
    files, skipped = discover_files(settings.docs_dir)
    if not files:
        raise FileNotFoundError(
            f"No supported documents in {settings.docs_dir} "
            f"(looking for {sorted(SUPPORTED_SUFFIXES)})"
        )

    all_chunks, ingested, failed = load_and_chunk(files, settings)
    if not ingested:
        details = "; ".join(f"{p.name}: {reason}" for p, reason in failed)
        raise RuntimeError(f"Every document failed to load. {details}")

    store = get_vectorstore(settings)
    if reset:
        store.reset_collection()

    # Batch to stay well under Chroma's per-call limits on large corpora.
    batch_size = 256
    for start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[start : start + batch_size]
        store.add_documents(batch, ids=[c.metadata["chunk_id"] for c in batch])

    return IngestReport(
        files=len(ingested),
        chunks=len(all_chunks),
        skipped=skipped,
        collection_count=collection_size(store),
        failed=failed,
    )
