"""HTTP API over the RAG pipeline — OpenAPI schema and Swagger UI included.

    uv run rag serve        # Swagger UI on http://127.0.0.1:8000/docs

Every expensive object (embedding model, Chroma collection, Anthropic client)
is built once per process, on first use. Construction is deliberately *not*
done at startup: a missing API key or an empty index must surface as a readable
HTTP error on the affected endpoint, not prevent the server from booting.
"""

from __future__ import annotations

import threading
from dataclasses import asdict

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from rag.config import Settings
from rag.pipeline import RAGPipeline
from rag.retriever import Retriever
from rag.store import collection_size

API_DESCRIPTION = """
Grounded question answering over the indexed documents.

Answers are generated **strictly from retrieved passages**, carry inline `[n]`
citation markers resolved back to their source files, and abstain with
`INSUFFICIENT_CONTEXT: …` rather than guessing. Markers pointing outside the
retrieved context are reported separately as `dangling_citations` — a hard,
non-LLM signal of a fabricated reference.
"""

TAGS_METADATA = [
    {"name": "rag", "description": "Ask questions against the indexed corpus."},
    {"name": "index", "description": "Inspect and rebuild the vector index."},
    {"name": "meta", "description": "Liveness."},
]


# --- Wire schemas -----------------------------------------------------------


class Citation(BaseModel):
    """A resolved `[n]` marker from the answer."""

    index: int = Field(description="Matches the [n] marker in the answer text")
    source: str = Field(description="Source file the passage came from")
    chunk_id: str
    snippet: str
    score: float = Field(description="Vector relevance score of the cited passage, 0..1")
    # Mirrors rag.pipeline.Citation exactly — the handler builds these with
    # Citation(**asdict(c)), so a missing field here is a validation error.
    rerank_score: float | None = Field(
        default=None,
        description="Cross-encoder score when a reranker is configured; scale is model-specific",
    )
    cited_text: str | None = Field(
        default=None,
        description="Exact span reported by the API; native citation mode only",
    )
    start_char_index: int | None = None
    end_char_index: int | None = None


class Passage(BaseModel):
    """One retrieved passage, as it was shown to the model."""

    index: int
    source: str
    chunk_id: str
    score: float = Field(description="Dense relevance, 0..1; 0.0 for a lexical-only hit")
    lexical_score: float | None = Field(
        default=None, description="BM25 score when the lexical channel found this chunk"
    )
    fusion_score: float | None = Field(
        default=None, description="Reciprocal-rank-fusion score in hybrid mode"
    )
    rerank_score: float | None = None
    channel: str = Field(description="Which retrieval channel found it: dense, lexical or both")
    text: str


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="Natural-language question about the indexed documents",
        examples=["Who has to approve an expense claim of £350?"],
    )
    k: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Passages to retrieve; falls back to the configured TOP_K",
    )
    include_context: bool = Field(
        default=False, description="Return the retrieved passages alongside the answer"
    )


class AskResponse(BaseModel):
    question: str
    answer: str
    abstained: bool = Field(description="True when the corpus could not answer the question")
    sources: list[str] = Field(description="Unique source files actually cited")
    citations: list[Citation]
    dangling_citations: list[int] = Field(
        description="Markers pointing outside the retrieved context — fabricated references"
    )
    context: list[Passage] | None = Field(
        default=None, description="Populated only when include_context is true"
    )
    usage: dict = Field(default_factory=dict, description="Token usage reported by the model")


class StatusResponse(BaseModel):
    docs_dir: str
    persist_dir: str
    collection: str
    indexed_chunks: int
    embeddings: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    score_threshold: float
    retrieval_mode: str
    reranker: str
    rerank_candidates: int
    citation_mode: str
    answer_model: str
    judge_model: str


class IngestRequest(BaseModel):
    reset: bool = Field(
        default=False,
        description="Drop the collection first. Required after changing the embedding model.",
    )


class FailedFile(BaseModel):
    file: str
    reason: str


class IngestResponse(BaseModel):
    files: int
    chunks: int
    collection_count: int
    skipped: list[str] = Field(description="Files with an unsupported extension")
    failed: list[FailedFile] = Field(description="Files that could not be read")


class HealthResponse(BaseModel):
    status: str


# --- Lazily built, process-wide runtime --------------------------------------


class _Runtime:
    """Holds the objects that are expensive to build, so they are built once.

    Opening the collection loads the embedding model; building the pipeline
    additionally requires an Anthropic key — so `/status` shares the retriever
    but never needs the pipeline.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._retriever: Retriever | None = None
        self._pipeline: RAGPipeline | None = None

    def retriever(self, settings: Settings) -> Retriever:
        with self._lock:
            if self._retriever is None:
                self._retriever = Retriever(settings)
            return self._retriever

    def pipeline(self, settings: Settings) -> RAGPipeline:
        with self._lock:
            if self._pipeline is None:
                self._pipeline = RAGPipeline(settings, retriever=self.retriever(settings))
            return self._pipeline

    def invalidate(self) -> None:
        """Drop cached handles — a reset ingestion replaces the collection."""
        with self._lock:
            self._retriever = None
            self._pipeline = None


_runtime = _Runtime()


def get_settings() -> Settings:
    try:
        return Settings.from_env()
    except ValueError as exc:  # unusable configuration, e.g. unknown provider
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc


def get_retriever(settings: Settings = Depends(get_settings)) -> Retriever:
    try:
        return _runtime.retriever(settings)
    except (RuntimeError, ImportError, ValueError) as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


def get_pipeline(settings: Settings = Depends(get_settings)) -> RAGPipeline:
    """The shared pipeline. Overridden in tests to avoid model construction."""
    try:
        return _runtime.pipeline(settings)
    except (RuntimeError, ImportError, ValueError) as exc:
        # Missing ANTHROPIC_API_KEY lands here — the server is up, the
        # dependency it needs is not.
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


# --- Application -------------------------------------------------------------

app = FastAPI(
    title="RAG Document Assistant",
    version="0.1.0",
    description=API_DESCRIPTION,
    openapi_tags=TAGS_METADATA,
)


@app.get("/health", tags=["meta"], summary="Liveness probe")
def health() -> HealthResponse:
    """Answer without touching the index, the models or the API key."""
    return HealthResponse(status="ok")


@app.get("/status", tags=["index"], summary="Resolved configuration and index size")
def read_status(
    settings: Settings = Depends(get_settings),
    retriever: Retriever = Depends(get_retriever),
) -> StatusResponse:
    return StatusResponse(
        docs_dir=str(settings.docs_dir),
        persist_dir=str(settings.persist_dir),
        collection=settings.collection_name,
        indexed_chunks=collection_size(retriever.store),
        embeddings=f"{settings.embedding_provider}:{settings.embedding_model}",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
        score_threshold=settings.score_threshold,
        retrieval_mode=settings.retrieval_mode,
        reranker=settings.reranker_label,
        rerank_candidates=settings.rerank_candidates,
        citation_mode=settings.citation_mode,
        answer_model=settings.answer_model,
        judge_model=settings.judge_model,
    )


@app.post(
    "/ask",
    tags=["rag"],
    summary="Grounded answer with citations",
    responses={
        409: {"description": "The vector store is empty — ingest documents first"},
        503: {"description": "The pipeline cannot be built (e.g. missing ANTHROPIC_API_KEY)"},
    },
)
def ask(
    payload: AskRequest, pipeline: RAGPipeline = Depends(get_pipeline)
) -> AskResponse:
    try:
        pipeline.retriever.ensure_indexed()
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    result = pipeline.answer(payload.question, k=payload.k)

    return AskResponse(
        question=result.question,
        answer=result.answer,
        abstained=result.abstained,
        sources=result.sources,
        citations=[Citation(**asdict(c)) for c in result.citations],
        dangling_citations=result.dangling_citations,
        context=(
            [
                Passage(
                    index=chunk.index,
                    source=chunk.source,
                    chunk_id=chunk.chunk_id,
                    score=chunk.score,
                    lexical_score=chunk.lexical_score,
                    fusion_score=chunk.fusion_score,
                    rerank_score=chunk.rerank_score,
                    channel=chunk.channel,
                    text=chunk.text,
                )
                for chunk in result.retrieved
            ]
            if payload.include_context
            else None
        ),
        usage=result.usage,
    )


@app.post(
    "/ingest",
    tags=["index"],
    summary="Load, chunk, embed and persist every document in the docs directory",
    responses={404: {"description": "No supported documents found"}},
)
def run_ingest(
    payload: IngestRequest, settings: Settings = Depends(get_settings)
) -> IngestResponse:
    from rag.ingest import ingest

    try:
        report = ingest(settings, reset=payload.reset)
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (RuntimeError, ImportError) as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    finally:
        # `--reset` replaces the collection behind any cached handle.
        _runtime.invalidate()

    return IngestResponse(
        files=report.files,
        chunks=report.chunks,
        collection_count=report.collection_count,
        skipped=[p.name for p in report.skipped],
        failed=[FailedFile(file=p.name, reason=reason) for p, reason in report.failed],
    )
