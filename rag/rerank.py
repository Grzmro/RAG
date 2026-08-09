"""Second-stage reranking: reorder a wide candidate set with a cross-encoder.

Vector search uses a bi-encoder: query and passage are embedded separately, so a
match means "these are about the same thing", not "this answers that". It is
fast, works over a whole collection, and is recall-oriented — the right passage
is usually *somewhere* in the top 20, but not reliably in the top 5.

A cross-encoder reads the query and the passage together in one forward pass and
is far better at that final judgement, at a cost that rules out running it over
the whole collection. So the store fetches `RERANK_CANDIDATES` passages and the
reranker orders them down to `TOP_K`.

Reranker scores are model-specific logits — unbounded, and comparable only within
a single query. They are recorded in `RetrievedChunk.rerank_score` rather than
overwriting the 0..1 relevance score that `SCORE_THRESHOLD` and the citation
table are calibrated against.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cached_property
from typing import TYPE_CHECKING, Callable, Sequence

from rag.config import Settings

if TYPE_CHECKING:  # avoids a cycle: retriever imports this module at runtime
    from rag.retriever import RetrievedChunk

# query, passage texts -> one score per passage, higher is more relevant
Reranker = Callable[[str, list[str]], list[float]]


def candidate_count(settings: Settings, k: int) -> int:
    """How many passages to pull from the store in order to rerank down to `k`.

    `max` rather than a bare `rerank_candidates`: `rag ask -k 30` must not return
    fewer than 30 passages just because the configured candidate pool is 20.
    """
    return max(k, settings.rerank_candidates) if settings.rerank_enabled else k


def renumber(chunks: list["RetrievedChunk"]) -> list["RetrievedChunk"]:
    """Reassign `index` 1..N, preserving order.

    Any stage that reorders or filters has to end here. `format_context` renders
    `[{index}]` and citation resolution looks markers back up by index, so a gap
    or a duplicate turns a real citation into a dangling one.
    """
    return [replace(chunk, index=rank) for rank, chunk in enumerate(chunks, start=1)]


def rerank_chunks(
    chunks: list["RetrievedChunk"], scores: Sequence[float], k: int
) -> list["RetrievedChunk"]:
    """Order candidates by reranker score, truncate to `k`, renumber from 1.

    Renumbering is not cosmetic: `format_context` renders `[{index}]` and
    `_resolve_citations` looks markers back up by index, so a gap or a duplicate
    here turns a real citation into a dangling one.

    Pure and model-free — the scores come from whichever backend produced them,
    which is what makes the ordering contract testable without a network or a
    model download. The sort is stable, so a reranker that returns a constant is
    a no-op rather than a shuffle, and ties keep vector-search order.
    """
    if len(scores) != len(chunks):
        raise ValueError(
            f"Reranker returned {len(scores)} scores for {len(chunks)} candidates."
        )
    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return [
        replace(chunks[i], index=rank, rerank_score=float(scores[i]))
        for rank, i in enumerate(order[:k], start=1)
    ]


def ranking_to_scores(ranking: Sequence[int], n: int) -> list[float]:
    """Turn a listwise 1-based ordering into descending pointwise scores.

    Numbers that are out of range or repeated are ignored, and candidates the
    model omitted keep 0.0 — with a stable sort they fall to the bottom in
    vector-search order rather than being dropped or randomised.
    """
    scores = [0.0] * n
    seen: set[int] = set()
    for rank, number in enumerate(ranking):
        try:
            i = int(number) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= i < n and i not in seen:
            seen.add(i)
            scores[i] = float(n - rank)
    return scores


class FastEmbedReranker:
    """Local ONNX cross-encoder. Downloads ~90 MB on first use, then cached."""

    def __init__(self, model_name: str, threads: int | None = None) -> None:
        self.model_name = model_name
        self.threads = threads

    @cached_property
    def _model(self):
        # NB: not exported from the `fastembed` package root.
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        return TextCrossEncoder(model_name=self.model_name, threads=self.threads)

    def __call__(self, query: str, texts: list[str]) -> list[float]:
        # `rerank` returns a lazy iterable; materialise it.
        return [float(score) for score in self._model.rerank(query, texts)]


class VoyageReranker:
    """Hosted reranking API. Needs VOYAGE_API_KEY."""

    def __init__(self, model: str) -> None:
        import voyageai

        self.model = model
        self._client = voyageai.Client()  # reads VOYAGE_API_KEY

    def __call__(self, query: str, texts: list[str]) -> list[float]:
        result = self._client.rerank(query=query, documents=list(texts), model=self.model)
        # Voyage returns results already sorted, each carrying a back-pointer to
        # its input position. Scatter them so scores[i] lines up with texts[i]:
        # ordering and truncation stay owned by rerank_chunks, in one place.
        scores = [0.0] * len(texts)
        for item in result.results:
            scores[item.index] = float(item.relevance_score)
        return scores


class CrossEncoderReranker:
    """sentence-transformers cross-encoder (torch, runs locally)."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def __call__(self, query: str, texts: list[str]) -> list[float]:
        pairs = [(query, text) for text in texts]
        return [float(score) for score in self._model.predict(pairs)]


class LLMReranker:
    """Listwise reranking by Claude — no extra dependency, no model download.

    Costs one additional API round-trip per question and is not deterministic,
    so it undermines the eval loop's ability to A/B two retrieval configurations.
    Useful as a demonstration; not recommended for `rag eval`.
    """

    #: passage text sent to the ranker, per candidate — the whole candidate set
    #: goes in one prompt, so full chunks would be wasteful for a relevance call
    SNIPPET_CHARS = 1200

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @cached_property
    def _llm(self):
        # Deferred: build_chat_model calls require_api_key(), and the Retriever is
        # also what GET /status uses — building eagerly would make /status fail
        # without an Anthropic key, which it does not today.
        from rag.llm import build_chat_model

        return build_chat_model(self.settings, self.settings.rerank_model or None)

    def __call__(self, query: str, texts: list[str]) -> list[float]:
        from langchain_core.messages import HumanMessage, SystemMessage

        from rag.llm import message_text, parse_json_object
        from rag.prompts import RERANK_SYSTEM_PROMPT, RERANK_USER_TEMPLATE

        listing = "\n\n".join(
            f"[{i + 1}] {text.strip()[: self.SNIPPET_CHARS]}"
            for i, text in enumerate(texts)
        )
        response = self._llm.invoke(
            [
                SystemMessage(content=RERANK_SYSTEM_PROMPT),
                HumanMessage(
                    content=RERANK_USER_TEMPLATE.format(question=query, passages=listing)
                ),
            ]
        )
        try:
            ranking = parse_json_object(message_text(response)).get("ranking") or []
        except ValueError:
            # A parse failure must not cost the user their answer. All-zero
            # scores plus a stable sort means "keep vector order".
            return [0.0] * len(texts)
        return ranking_to_scores(ranking, len(texts))


def build_reranker(settings: Settings) -> Reranker | None:
    """Return a scoring callable, or None when reranking is disabled."""
    provider = settings.rerank_provider

    if provider == "none":
        return None

    if provider == "fastembed":
        return FastEmbedReranker(model_name=settings.rerank_model)

    if provider == "voyage":
        try:
            import voyageai  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "RERANK_PROVIDER=voyage requires: pip install voyageai"
            ) from exc

        return VoyageReranker(model=settings.rerank_model)

    if provider == "huggingface":
        try:
            from sentence_transformers import CrossEncoder  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "RERANK_PROVIDER=huggingface requires: pip install sentence-transformers"
            ) from exc

        return CrossEncoderReranker(model_name=settings.rerank_model)

    if provider == "llm":
        return LLMReranker(settings)

    raise ValueError(f"Unsupported rerank provider: {provider!r}")
