"""Chroma vector store wiring."""

from __future__ import annotations

from langchain_chroma import Chroma

from rag.config import Settings
from rag.embeddings import build_embeddings


def get_vectorstore(settings: Settings) -> Chroma:
    """Open (or create) the persistent Chroma collection."""
    settings.persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=build_embeddings(settings),
        persist_directory=str(settings.persist_dir),
    )


def collection_size(store: Chroma) -> int:
    return store._collection.count()  # noqa: SLF001 - no public accessor in langchain-chroma


def all_chunks(store: Chroma) -> tuple[list[str], list[str], list[dict]]:
    """Read the whole collection back out as (chunk_ids, texts, metadatas).

    The lexical half of hybrid retrieval needs the full corpus in memory, and
    reading it from Chroma rather than re-reading `docs_dir` guarantees both
    channels range over exactly the same chunks.
    """
    got = store.get(include=["documents", "metadatas"])
    return got["ids"], got["documents"], got["metadatas"]
