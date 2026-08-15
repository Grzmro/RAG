"""Native citations: let the API report spans instead of parsing `[n]` markers.

In marker mode the model writes `[2]` into its prose and we regex it back out.
That works, but the citation is the model's own claim about what it used — it
can attach the wrong number to a true sentence, and nothing downstream can tell.
`dangling_citations` catches only the crude failure of citing a passage that was
never in the context block.

With `citations: {enabled: true}` on each document block, the API returns the
citation as structured data: which document, and the exact character span within
it. The span is produced by the serving layer, not written by the model, so a
citation cannot point at text that isn't there — it is checkable against the
source rather than taken on trust. That is the whole reason to prefer this mode.

The tradeoff: the answer carries no inline markers, so the prose reads without
`[n]` and the mapping lives in the citation list instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rag.retriever import RetrievedChunk


def build_document_blocks(chunks: Sequence[RetrievedChunk]) -> list[dict[str, Any]]:
    """Render retrieved chunks as citation-enabled `document` content blocks.

    Block order is what `document_index` refers to in the response, so these
    must stay aligned with `chunks` and be sent before the question block.
    """
    return [
        {
            "type": "document",
            "source": {
                "type": "text",
                "media_type": "text/plain",
                "data": chunk.text,
            },
            "title": chunk.source,
            "context": f"chunk {chunk.chunk_id}",
            "citations": {"enabled": True},
        }
        for chunk in chunks
    ]


def answer_text(content: Any) -> str:
    """Flatten the response into prose, ignoring citation metadata.

    With citations on, the reply arrives as several text blocks — cited ones and
    uncited ones — which concatenate back into the answer.
    """
    if isinstance(content, str):
        return content.strip()

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts).strip()


def verify_spans(citations: Sequence[Any], chunks: Sequence[RetrievedChunk]) -> list[Any]:
    """Return the citations whose span does not match the source text.

    This is the check that the marker mode cannot perform at all and that the
    judge should not be asked to perform: `chunk.text[start:end]` either equals
    `cited_text` or it does not, and no model opinion is involved. A non-empty
    result means the answer's evidence does not survive being looked up.
    """
    by_id = {chunk.chunk_id: chunk.text for chunk in chunks}
    mismatched: list[Any] = []
    for citation in citations:
        start, end = citation.start_char_index, citation.end_char_index
        source = by_id.get(citation.chunk_id)
        if source is None or start is None or end is None:
            mismatched.append(citation)
            continue
        if source[start:end] != citation.cited_text:
            mismatched.append(citation)
    return mismatched


def extract_citations(content: Any) -> list[dict[str, Any]]:
    """Pull the citation objects out of the response, in order of appearance.

    Duplicate spans are dropped: a model that cites the same sentence three
    times has used one piece of evidence, and the citation table should say so.
    """
    if isinstance(content, str):
        return []

    seen: set[tuple] = set()
    found: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        for citation in block.get("citations") or []:
            if not isinstance(citation, dict):
                continue
            key = (
                citation.get("document_index"),
                citation.get("start_char_index"),
                citation.get("end_char_index"),
                citation.get("cited_text"),
            )
            if key in seen:
                continue
            seen.add(key)
            found.append(citation)
    return found
