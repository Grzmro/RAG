"""Embedding backend factory.

Anthropic does not serve an embeddings endpoint, so the semantic-search half of
the pipeline is pluggable. The default is a local ONNX model (no extra API key,
no torch dependency); Voyage AI and HuggingFace are available as alternatives.
"""

from __future__ import annotations

from functools import cached_property

from langchain_core.embeddings import Embeddings

from rag.config import Settings


class FastEmbedEmbeddings(Embeddings):
    """LangChain adapter over `fastembed.TextEmbedding` (ONNX, runs locally).

    Uses the model's asymmetric query/passage encoders where the checkpoint
    defines them (BGE and E5 families do), which measurably improves retrieval
    over encoding both sides identically.
    """

    def __init__(self, model_name: str, threads: int | None = None) -> None:
        self.model_name = model_name
        self.threads = threads

    @cached_property
    def _model(self):
        from fastembed import TextEmbedding

        return TextEmbedding(model_name=self.model_name, threads=self.threads)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed([text]))).tolist()


def build_embeddings(settings: Settings) -> Embeddings:
    provider = settings.embedding_provider

    if provider == "fastembed":
        return FastEmbedEmbeddings(model_name=settings.embedding_model)

    if provider == "voyage":
        try:
            from langchain_voyageai import VoyageAIEmbeddings  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "EMBEDDING_PROVIDER=voyage requires: pip install langchain-voyageai voyageai"
            ) from exc

        return VoyageAIEmbeddings(model=settings.embedding_model)

    if provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "EMBEDDING_PROVIDER=huggingface requires: "
                "pip install langchain-huggingface sentence-transformers"
            ) from exc

        return HuggingFaceEmbeddings(model_name=settings.embedding_model)

    raise ValueError(f"Unsupported embedding provider: {provider!r}")
