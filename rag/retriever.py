"""Embedding-based semantic search over the Chroma collection."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from langchain_core.documents import Document

from rag.config import Settings
from rag.hybrid import BM25Index, fuse_rankings
from rag.rerank import build_reranker, candidate_count, renumber, rerank_chunks
from rag.store import all_chunks, collection_size, get_vectorstore


@dataclass
class RetrievedChunk:
    """One retrieved passage, numbered for citation."""

    index: int  # 1-based, matches the [n] markers shown to the model
    document: Document
    score: float
    # Every stage records its own score rather than overwriting the one before:
    # a cosine similarity, a BM25 score, an RRF score and a cross-encoder logit
    # share no scale, so collapsing them into one number destroys all four.
    # `score` is the dense 0..1 relevance, and is 0.0 for a lexical-only hit.
    lexical_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None

    @property
    def ranking_score(self) -> float:
        """The score that actually determined this chunk's final position."""
        for score in (self.rerank_score, self.fusion_score):
            if score is not None:
                return score
        return self.score

    @property
    def channel(self) -> str:
        """Which retrieval channel(s) found this chunk."""
        if self.lexical_score is None:
            return "dense"
        return "both" if self.score > 0 else "lexical"

    @property
    def source(self) -> str:
        return self.document.metadata.get("source", "unknown")

    @property
    def chunk_id(self) -> str:
        return self.document.metadata.get("chunk_id", f"unknown::{self.index}")

    @property
    def text(self) -> str:
        return self.document.page_content

    def snippet(self, limit: int = 220) -> str:
        flat = " ".join(self.text.split())
        return flat if len(flat) <= limit else flat[: limit - 1] + "…"


class Retriever:
    """Thin wrapper over the vector store that returns citation-ready chunks."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = get_vectorstore(settings)
        # Built eagerly so a bad RERANK_PROVIDER or a missing optional extra
        # surfaces where the API already turns it into a readable 503, rather
        # than from inside a request handler. The model itself still loads lazily.
        self.reranker = build_reranker(settings)

    def ensure_indexed(self) -> None:
        if collection_size(self.store) == 0:
            raise RuntimeError(
                "The vector store is empty. Run `python -m rag.cli ingest` first."
            )

    @cached_property
    def _corpus(self) -> tuple[dict[str, Document], BM25Index]:
        """The whole collection, plus a BM25 index over it.

        Lazy: a dense-only run never reads the corpus back or pays to tokenize
        it. Cached per Retriever, and the API drops the Retriever after every
        ingest, so the index cannot outlive the collection it was built from.
        """
        ids, texts, metadatas = all_chunks(self.store)
        documents = {
            chunk_id: Document(page_content=text, metadata=dict(metadata))
            for chunk_id, text, metadata in zip(ids, texts, metadatas)
        }
        index = BM25Index.build(
            ids, texts, k1=self.settings.bm25_k1, b=self.settings.bm25_b
        )
        return documents, index

    def _dense(self, query: str, fetch_k: int) -> list[RetrievedChunk]:
        hits = self.store.similarity_search_with_relevance_scores(query, k=fetch_k)
        # The threshold is deliberately applied to vector scores, before fusion
        # and reranking: a cutoff on either of those would need per-model
        # calibration. A lexical-only hit has no dense score to threshold.
        return [
            RetrievedChunk(index=0, document=document, score=float(score))
            for document, score in hits
            if not (
                self.settings.score_threshold > 0
                and score < self.settings.score_threshold
            )
        ]

    def _hybrid(self, query: str, fetch_k: int) -> list[RetrievedChunk]:
        """Fuse the dense and lexical channels by reciprocal rank."""
        dense = self._dense(query, fetch_k)
        documents, index = self._corpus
        lexical = index.search(query, fetch_k)

        by_id = {chunk.chunk_id: chunk for chunk in dense}
        lexical_scores = dict(lexical)
        for chunk_id, score in lexical:
            existing = by_id.get(chunk_id)
            if existing is not None:
                existing.lexical_score = score
            elif chunk_id in documents:
                by_id[chunk_id] = RetrievedChunk(
                    index=0,
                    document=documents[chunk_id],
                    score=0.0,  # never scored by the dense channel
                    lexical_score=score,
                )

        # Dense ranking first so it breaks ties in fuse_rankings.
        fused = fuse_rankings(
            [[c.chunk_id for c in dense], [chunk_id for chunk_id, _ in lexical]],
            k=self.settings.rrf_k,
        )
        ordered: list[RetrievedChunk] = []
        for chunk_id, rrf in fused:
            chunk = by_id.get(chunk_id)
            if chunk is None:
                continue
            chunk.fusion_score = rrf
            chunk.lexical_score = lexical_scores.get(chunk_id, chunk.lexical_score)
            ordered.append(chunk)
        return ordered

    def search(
        self, query: str, k: int | None = None, *, fetch_k: int | None = None
    ) -> list[RetrievedChunk]:
        """Retrieve `k` passages through the configured retrieval pipeline.

        Up to three stages: dense search (optionally fused with a lexical BM25
        channel), then cross-encoder reranking, then truncation to `k`.
        `fetch_k` is keyword-only so existing two-argument call sites keep working.
        """
        k = k or self.settings.top_k
        fetch_k = fetch_k or candidate_count(self.settings, k)

        if self.settings.hybrid_enabled:
            candidates = self._hybrid(query, fetch_k)[:fetch_k]
        else:
            candidates = self._dense(query, fetch_k)

        if self.reranker is None or not candidates:
            # With reranking off, candidate_count returns k, so this slice is a
            # no-op; it only bites when a caller passes fetch_k explicitly.
            return renumber(candidates[:k])

        scores = self.reranker(query, [c.text for c in candidates])
        return rerank_chunks(candidates, scores, k)


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as the numbered context block sent to the model."""
    if not chunks:
        return "(no passages retrieved)"
    blocks = [
        f"[{c.index}] source: {c.source}\n{c.text.strip()}" for c in chunks
    ]
    return "\n\n---\n\n".join(blocks)
