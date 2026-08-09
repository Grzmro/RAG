"""API tests (no API calls, no model construction, no network).

The pipeline dependency is overridden with a stub, so these exercise the HTTP
layer itself: schemas, status codes and the mapping from pipeline state to
responses.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from rag.api import app, get_pipeline
from rag.pipeline import Citation, RAGAnswer
from rag.retriever import RetrievedChunk


def make_chunk(
    index: int, source: str = "handbook.md", rerank_score: float | None = None
) -> RetrievedChunk:
    return RetrievedChunk(
        index=index,
        document=Document(
            page_content=f"passage {index}",
            metadata={"source": source, "chunk_id": f"{source}::{index - 1}"},
        ),
        score=0.71,
        rerank_score=rerank_score,
    )


class StubRetriever:
    def __init__(self, indexed: bool = True) -> None:
        self.indexed = indexed

    def ensure_indexed(self) -> None:
        if not self.indexed:
            raise RuntimeError("The vector store is empty. Run `python -m rag.cli ingest` first.")


class StubPipeline:
    """Stands in for RAGPipeline; records the arguments it was called with."""

    def __init__(self, answer: RAGAnswer, indexed: bool = True) -> None:
        self._answer = answer
        self.retriever = StubRetriever(indexed)
        self.calls: list[tuple[str, int | None]] = []

    def answer(self, question: str, k: int | None = None) -> RAGAnswer:
        self.calls.append((question, k))
        return self._answer


def grounded_answer() -> RAGAnswer:
    chunk = make_chunk(1)
    return RAGAnswer(
        question="Who approves a £350 expense claim?",
        answer="The department head [1].",
        retrieved=[chunk],
        citations=[
            Citation(
                index=1,
                source=chunk.source,
                chunk_id=chunk.chunk_id,
                snippet=chunk.snippet(),
                score=chunk.score,
            )
        ],
        usage={"input_tokens": 900, "output_tokens": 40},
    )


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def use_pipeline(pipeline: StubPipeline) -> None:
    app.dependency_overrides[get_pipeline] = lambda: pipeline


class TestHealth:
    def test_health_needs_no_index_or_key(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestAsk:
    def test_returns_answer_with_resolved_citations(self, client):
        pipeline = StubPipeline(grounded_answer())
        use_pipeline(pipeline)

        response = client.post("/ask", json={"question": "Who approves £350?"})
        assert response.status_code == 200

        body = response.json()
        assert body["answer"] == "The department head [1]."
        assert body["abstained"] is False
        assert body["sources"] == ["handbook.md"]
        assert body["citations"][0]["index"] == 1
        assert body["citations"][0]["source"] == "handbook.md"
        assert body["dangling_citations"] == []
        assert body["context"] is None
        assert pipeline.calls == [("Who approves £350?", None)]

    def test_k_is_forwarded_and_context_is_opt_in(self, client):
        pipeline = StubPipeline(grounded_answer())
        use_pipeline(pipeline)

        response = client.post(
            "/ask", json={"question": "Who approves £350?", "k": 3, "include_context": True}
        )
        assert response.status_code == 200

        context = response.json()["context"]
        assert [p["index"] for p in context] == [1]
        assert context[0]["text"] == "passage 1"
        assert pipeline.calls == [("Who approves £350?", 3)]

    def test_rerank_score_is_null_when_no_reranker_ran(self, client):
        use_pipeline(StubPipeline(grounded_answer()))

        body = client.post(
            "/ask", json={"question": "Who approves £350?", "include_context": True}
        ).json()
        assert body["citations"][0]["rerank_score"] is None
        assert body["context"][0]["rerank_score"] is None

    def test_rerank_score_is_carried_through(self, client):
        # Guards the Citation(**asdict(c)) mirror: if the Pydantic model and the
        # dataclass drift apart, that call raises rather than dropping a field.
        chunk = make_chunk(1, rerank_score=-2.5)
        answer = RAGAnswer(
            question="Who approves £350?",
            answer="The department head [1].",
            retrieved=[chunk],
            citations=[
                Citation(
                    index=1,
                    source=chunk.source,
                    chunk_id=chunk.chunk_id,
                    snippet=chunk.snippet(),
                    score=chunk.score,
                    rerank_score=chunk.rerank_score,
                )
            ],
        )
        use_pipeline(StubPipeline(answer))

        body = client.post(
            "/ask", json={"question": "Who approves £350?", "include_context": True}
        ).json()
        assert body["citations"][0]["rerank_score"] == -2.5
        assert body["context"][0]["rerank_score"] == -2.5
        assert body["citations"][0]["score"] == 0.71  # vector score preserved

    def test_abstention_and_dangling_markers_are_reported(self, client):
        answer = RAGAnswer(
            question="What does Atlas cost?",
            answer="INSUFFICIENT_CONTEXT: pricing is not in the documents [4].",
            retrieved=[make_chunk(1)],
            dangling_citations=[4],
            abstained=True,
        )
        use_pipeline(StubPipeline(answer))

        body = client.post("/ask", json={"question": "What does Atlas cost?"}).json()
        assert body["abstained"] is True
        assert body["dangling_citations"] == [4]
        assert body["sources"] == []

    def test_empty_index_is_a_conflict_not_a_crash(self, client):
        use_pipeline(StubPipeline(grounded_answer(), indexed=False))

        response = client.post("/ask", json={"question": "anything"})
        assert response.status_code == 409
        assert "empty" in response.json()["detail"]

    @pytest.mark.parametrize(
        "payload",
        [{}, {"question": ""}, {"question": "ok", "k": 0}, {"question": "ok", "k": 99}],
    )
    def test_invalid_payloads_are_rejected(self, client, payload):
        use_pipeline(StubPipeline(grounded_answer()))
        assert client.post("/ask", json=payload).status_code == 422


class TestOpenAPI:
    def test_schema_documents_every_route(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        assert set(paths) == {"/health", "/status", "/ask", "/ingest"}

    def test_swagger_ui_is_served(self, client):
        response = client.get("/docs")
        assert response.status_code == 200
        assert "swagger" in response.text.lower()
