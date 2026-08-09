"""End-to-end RAG pipeline: retrieve -> ground -> answer with citations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from rag.citations import answer_text, build_document_blocks, extract_citations
from rag.config import Settings
from rag.llm import build_chat_model, message_text
from rag.prompts import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_SYSTEM_PROMPT_NATIVE,
    ANSWER_USER_TEMPLATE,
    ANSWER_USER_TEMPLATE_NATIVE,
    INSUFFICIENT_MARKER,
)
from rag.retriever import RetrievedChunk, Retriever, format_context

CITATION_RE = re.compile(r"\[(\d{1,2})\]")


@dataclass
class Citation:
    index: int
    source: str
    chunk_id: str
    snippet: str
    score: float
    rerank_score: float | None = None
    # Populated only in native citation mode: the exact span the API reported.
    cited_text: str | None = None
    start_char_index: int | None = None
    end_char_index: int | None = None


@dataclass
class RAGAnswer:
    question: str
    answer: str
    retrieved: list[RetrievedChunk]
    citations: list[Citation] = field(default_factory=list)
    dangling_citations: list[int] = field(default_factory=list)
    abstained: bool = False
    usage: dict = field(default_factory=dict)

    @property
    def sources(self) -> list[str]:
        """Unique source files actually cited, in citation order."""
        seen: list[str] = []
        for citation in self.citations:
            if citation.source not in seen:
                seen.append(citation.source)
        return seen

    @property
    def retrieved_sources(self) -> list[str]:
        seen: list[str] = []
        for chunk in self.retrieved:
            if chunk.source not in seen:
                seen.append(chunk.source)
        return seen


class RAGPipeline:
    def __init__(self, settings: Settings, retriever: Retriever | None = None) -> None:
        self.settings = settings
        self.retriever = retriever or Retriever(settings)
        self.llm = build_chat_model(settings, settings.answer_model)

    def answer(self, question: str, k: int | None = None) -> RAGAnswer:
        chunks = self.retriever.search(question, k=k)

        if not chunks:
            return RAGAnswer(
                question=question,
                answer=(
                    f"{INSUFFICIENT_MARKER}: no passage in the indexed documents "
                    "was relevant to this question."
                ),
                retrieved=[],
                abstained=True,
            )

        if self.settings.native_citations:
            response = self.llm.invoke(
                [
                    SystemMessage(content=ANSWER_SYSTEM_PROMPT_NATIVE),
                    HumanMessage(
                        content=[
                            *build_document_blocks(chunks),
                            {
                                "type": "text",
                                "text": ANSWER_USER_TEMPLATE_NATIVE.format(
                                    question=question
                                ),
                            },
                        ]
                    ),
                ]
            )
            text = answer_text(response.content)
            cited, dangling = self._resolve_native_citations(response.content, chunks)
        else:
            response = self.llm.invoke(
                [
                    SystemMessage(content=ANSWER_SYSTEM_PROMPT),
                    HumanMessage(
                        content=ANSWER_USER_TEMPLATE.format(
                            context=format_context(chunks), question=question
                        )
                    ),
                ]
            )
            text = message_text(response)
            cited, dangling = self._resolve_citations(text, chunks)

        return RAGAnswer(
            question=question,
            answer=text,
            retrieved=chunks,
            citations=cited,
            dangling_citations=dangling,
            abstained=text.upper().startswith(INSUFFICIENT_MARKER),
            usage=getattr(response, "usage_metadata", None) or {},
        )

    @staticmethod
    def _resolve_native_citations(
        content: object, chunks: list[RetrievedChunk]
    ) -> tuple[list[Citation], list[int]]:
        """Map API-reported citation spans back to retrieved chunks.

        `document_index` is the position of the document block we sent, so it
        indexes `chunks` directly. The returned dangling list is always empty:
        the API can only cite a document that was attached, which is the point
        of this mode — fabricated references stop being expressible rather than
        being detected after the fact.
        """
        cited: list[Citation] = []
        for citation in extract_citations(content):
            position = citation.get("document_index")
            if not isinstance(position, int) or not 0 <= position < len(chunks):
                continue
            chunk = chunks[position]
            cited.append(
                Citation(
                    index=chunk.index,
                    source=chunk.source,
                    chunk_id=chunk.chunk_id,
                    snippet=chunk.snippet(),
                    score=chunk.score,
                    rerank_score=chunk.rerank_score,
                    cited_text=citation.get("cited_text"),
                    start_char_index=citation.get("start_char_index"),
                    end_char_index=citation.get("end_char_index"),
                )
            )
        return cited, []

    @staticmethod
    def _resolve_citations(
        text: str, chunks: list[RetrievedChunk]
    ) -> tuple[list[Citation], list[int]]:
        """Map inline [n] markers back to retrieved chunks.

        Markers that point outside the context block are reported separately —
        they are a hard, non-LLM signal of a fabricated reference.
        """
        by_index = {chunk.index: chunk for chunk in chunks}
        cited: list[Citation] = []
        dangling: list[int] = []
        seen: set[int] = set()

        for match in CITATION_RE.finditer(text):
            index = int(match.group(1))
            if index in seen:
                continue
            seen.add(index)
            chunk = by_index.get(index)
            if chunk is None:
                dangling.append(index)
                continue
            cited.append(
                Citation(
                    index=chunk.index,
                    source=chunk.source,
                    chunk_id=chunk.chunk_id,
                    snippet=chunk.snippet(),
                    score=chunk.score,
                    rerank_score=chunk.rerank_score,
                )
            )
        return cited, dangling
