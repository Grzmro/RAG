"""Central configuration, resolved from environment variables (.env supported)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

_DEFAULT_EMBEDDING_MODELS = {
    "fastembed": "BAAI/bge-small-en-v1.5",
    "voyage": "voyage-3",
    "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
}

_DEFAULT_RERANK_MODELS = {
    "none": "",
    "fastembed": "Xenova/ms-marco-MiniLM-L-6-v2",
    "voyage": "rerank-2-lite",
    "huggingface": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "llm": "",  # empty -> ANSWER_MODEL
}

_RETRIEVAL_MODES = ("dense", "hybrid")
_CITATION_MODES = ("markers", "native")


def _path(value: str) -> Path:
    """Resolve a possibly-relative path against the project root."""
    p = Path(value).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _number(name: str, default: str, low: float, high: float | None = None) -> float:
    """Read a numeric setting and reject values outside its meaningful range.

    Provider names are already validated strictly; numbers were not, so a
    `BM25_B=5` or a negative `TOP_K` was accepted and then quietly produced
    nonsense scores or empty results with nothing to point at.
    """
    raw = os.getenv(name, default)
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a number.") from None
    if value < low or (high is not None and value > high):
        bound = f"{low}..{high}" if high is not None else f"at least {low}"
        raise ValueError(f"{name}={raw} is out of range (expected {bound}).")
    return value


@dataclass(frozen=True)
class Settings:
    # paths
    docs_dir: Path
    persist_dir: Path
    collection_name: str
    # embeddings
    embedding_provider: str
    embedding_model: str
    # chunking / retrieval
    chunk_size: int
    chunk_overlap: int
    top_k: int
    score_threshold: float
    # generation
    answer_model: str
    judge_model: str
    max_tokens: int
    # reranking — appended rather than grouped with retrieval above so these can
    # carry defaults: a dataclass cannot place a default-free field after a
    # defaulted one, and every other field is deliberately default-free.
    rerank_provider: str = "none"
    rerank_model: str = ""
    rerank_candidates: int = 20
    # hybrid retrieval and citation style — appended for the same reason
    retrieval_mode: str = "dense"
    rrf_k: int = 60
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    citation_mode: str = "markers"

    @property
    def rerank_enabled(self) -> bool:
        return self.rerank_provider != "none"

    @property
    def hybrid_enabled(self) -> bool:
        return self.retrieval_mode == "hybrid"

    @property
    def native_citations(self) -> bool:
        return self.citation_mode == "native"

    @property
    def reranker_label(self) -> str:
        """`provider:model` for display, or `none` — used by the CLI, API and eval."""
        if not self.rerank_enabled:
            return "none"
        return f"{self.rerank_provider}:{self.rerank_model or self.answer_model}"

    @classmethod
    def from_env(cls) -> Settings:
        provider = os.getenv("EMBEDDING_PROVIDER", "fastembed").strip().lower()
        if provider not in _DEFAULT_EMBEDDING_MODELS:
            raise ValueError(
                f"Unknown EMBEDDING_PROVIDER={provider!r}. "
                f"Expected one of {sorted(_DEFAULT_EMBEDDING_MODELS)}."
            )
        rerank_provider = os.getenv("RERANK_PROVIDER", "none").strip().lower()
        if rerank_provider not in _DEFAULT_RERANK_MODELS:
            raise ValueError(
                f"Unknown RERANK_PROVIDER={rerank_provider!r}. "
                f"Expected one of {sorted(_DEFAULT_RERANK_MODELS)}."
            )
        retrieval_mode = os.getenv("RETRIEVAL_MODE", "dense").strip().lower()
        if retrieval_mode not in _RETRIEVAL_MODES:
            raise ValueError(
                f"Unknown RETRIEVAL_MODE={retrieval_mode!r}. "
                f"Expected one of {sorted(_RETRIEVAL_MODES)}."
            )
        citation_mode = os.getenv("CITATION_MODE", "markers").strip().lower()
        if citation_mode not in _CITATION_MODES:
            raise ValueError(
                f"Unknown CITATION_MODE={citation_mode!r}. "
                f"Expected one of {sorted(_CITATION_MODES)}."
            )
        return cls(
            docs_dir=_path(os.getenv("DOCS_DIR", "data/docs")),
            persist_dir=_path(os.getenv("PERSIST_DIR", ".chroma")),
            collection_name=os.getenv("COLLECTION_NAME", "rag_documents"),
            embedding_provider=provider,
            embedding_model=os.getenv("EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODELS[provider]),
            chunk_size=int(_number("CHUNK_SIZE", "900", 1)),
            chunk_overlap=int(_number("CHUNK_OVERLAP", "150", 0)),
            top_k=int(_number("TOP_K", "5", 1)),
            # Chroma's relevance scores are normalised to 0..1, so a threshold
            # outside that range either filters nothing or filters everything.
            score_threshold=_number("SCORE_THRESHOLD", "0.0", 0.0, 1.0),
            answer_model=os.getenv("ANSWER_MODEL", "claude-opus-5"),
            judge_model=os.getenv("JUDGE_MODEL", "claude-opus-5"),
            max_tokens=int(_number("MAX_TOKENS", "8000", 1)),
            rerank_provider=rerank_provider,
            # `or` rather than a two-arg getenv: .env.example ships an empty
            # `RERANK_MODEL=` line, which must resolve to the provider default.
            rerank_model=(os.getenv("RERANK_MODEL") or _DEFAULT_RERANK_MODELS[rerank_provider]),
            rerank_candidates=int(_number("RERANK_CANDIDATES", "20", 1)),
            retrieval_mode=retrieval_mode,
            rrf_k=int(_number("RRF_K", "60", 1)),
            # Okapi BM25 is only defined for a non-negative k1, and b is a
            # blend factor between no length normalisation and full.
            bm25_k1=_number("BM25_K1", "1.5", 0.0),
            bm25_b=_number("BM25_B", "0.75", 0.0, 1.0),
            citation_mode=citation_mode,
        )

    def require_api_key(self) -> None:
        """Fail early with a readable message instead of deep inside the SDK."""
        if os.getenv("ANTHROPIC_API_KEY"):
            return
        env_file = PROJECT_ROOT / ".env"
        hint = (
            f"Set ANTHROPIC_API_KEY in {env_file}"
            if env_file.exists()
            else "Copy .env.example to .env and set ANTHROPIC_API_KEY"
        )
        raise RuntimeError(f"ANTHROPIC_API_KEY is empty. {hint}.")
