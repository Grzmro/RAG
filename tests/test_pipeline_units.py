"""Unit tests for the deterministic parts of the pipeline (no API calls)."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from rag.config import Settings
from rag.ingest import chunk_documents, load_and_chunk
from rag.llm import parse_json_object
from rag.loaders import discover_files
from rag.citations import answer_text, build_document_blocks, extract_citations
from rag.hybrid import BM25Index, fuse_rankings, reciprocal_rank_fusion, tokenize
from rag.pipeline import RAGPipeline
from rag.rerank import build_reranker, candidate_count, ranking_to_scores, rerank_chunks
from rag.retriever import RetrievedChunk, format_context


def make_settings(**overrides) -> Settings:
    base = dict(
        docs_dir=Path("data/docs"),
        persist_dir=Path(".chroma"),
        collection_name="test",
        embedding_provider="fastembed",
        embedding_model="BAAI/bge-small-en-v1.5",
        chunk_size=200,
        chunk_overlap=20,
        top_k=3,
        score_threshold=0.0,
        answer_model="claude-opus-5",
        judge_model="claude-opus-5",
        max_tokens=4000,
        rerank_provider="none",
        rerank_model="",
        rerank_candidates=20,
        retrieval_mode="dense",
        rrf_k=60,
        bm25_k1=1.5,
        bm25_b=0.75,
        citation_mode="markers",
    )
    base.update(overrides)
    return Settings(**base)


def make_chunk(
    index: int,
    source: str = "doc.md",
    text: str = "passage text",
    score: float = 0.9,
) -> RetrievedChunk:
    return RetrievedChunk(
        index=index,
        document=Document(
            page_content=text,
            metadata={"source": source, "chunk_id": f"{source}::{index - 1}"},
        ),
        score=score,
    )


class TestChunking:
    def test_chunks_get_stable_ids_and_indices(self):
        docs = [
            Document(
                page_content="A" * 500 + "\n\n" + "B" * 500,
                metadata={"source": "doc.md"},
            )
        ]
        chunks = chunk_documents(docs, make_settings())

        assert len(chunks) > 1
        assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
        assert chunks[0].metadata["chunk_id"] == "doc.md::0"
        assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)

    def test_indices_restart_per_source(self):
        docs = [
            Document(page_content="x" * 400, metadata={"source": "a.md"}),
            Document(page_content="y" * 400, metadata={"source": "b.md"}),
        ]
        chunks = chunk_documents(docs, make_settings())
        assert {c.metadata["chunk_id"] for c in chunks} >= {"a.md::0", "b.md::0"}


class TestLoadResilience:
    def test_a_corrupt_file_does_not_abort_the_run(self, tmp_path: Path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "good.md").write_text("# Good\n\n" + "content " * 60, encoding="utf-8")
        (docs_dir / "broken.pdf").write_bytes(b"this is not a pdf")

        supported, _ = discover_files(docs_dir)
        chunks, ingested, failed = load_and_chunk(
            supported, make_settings(docs_dir=docs_dir)
        )

        assert [p.name for p in ingested] == ["good.md"]
        assert [p.name for p, _ in failed] == ["broken.pdf"]
        assert chunks and all(c.metadata["source"] == "good.md" for c in chunks)

    def test_unsupported_extensions_are_not_offered_to_loaders(self, tmp_path: Path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "notes.md").write_text("# Notes", encoding="utf-8")
        (docs_dir / "sheet.xlsx").write_bytes(b"binary")

        supported, skipped = discover_files(docs_dir)
        assert [p.name for p in supported] == ["notes.md"]
        assert [p.name for p in skipped] == ["sheet.xlsx"]


class TestCitationResolution:
    def test_maps_markers_to_chunks(self):
        chunks = [make_chunk(1, "a.md"), make_chunk(2, "b.md")]
        cited, dangling = RAGPipeline._resolve_citations(
            "First fact [1]. Second fact [2]. Restated [1].", chunks
        )
        assert [c.index for c in cited] == [1, 2]
        assert [c.source for c in cited] == ["a.md", "b.md"]
        assert dangling == []

    def test_flags_markers_outside_the_context_block(self):
        chunks = [make_chunk(1)]
        cited, dangling = RAGPipeline._resolve_citations("Claim [1] and claim [7].", chunks)
        assert [c.index for c in cited] == [1]
        assert dangling == [7]

    def test_uncited_answer_yields_no_citations(self):
        cited, dangling = RAGPipeline._resolve_citations("No markers here.", [make_chunk(1)])
        assert cited == []
        assert dangling == []


class TestContextFormatting:
    def test_numbers_passages_and_labels_sources(self):
        rendered = format_context([make_chunk(1, "a.md", "alpha"), make_chunk(2, "b.md", "beta")])
        assert "[1] source: a.md" in rendered
        assert "[2] source: b.md" in rendered
        assert "alpha" in rendered and "beta" in rendered

    def test_empty_retrieval(self):
        assert format_context([]) == "(no passages retrieved)"


class TestReranking:
    """The reranker's ordering contract — pure, no model and no network."""

    @staticmethod
    def _candidates() -> list[RetrievedChunk]:
        return [
            make_chunk(1, "a.md", "alpha", score=0.80),
            make_chunk(2, "b.md", "beta", score=0.70),
            make_chunk(3, "c.md", "gamma", score=0.60),
        ]

    def test_reorders_by_score_and_renumbers(self):
        out = rerank_chunks(self._candidates(), [0.1, 9.0, 3.0], k=3)
        assert [c.source for c in out] == ["b.md", "c.md", "a.md"]
        assert [c.index for c in out] == [1, 2, 3]

    def test_truncates_to_k(self):
        chunks = [make_chunk(i, f"s{i}.md") for i in range(1, 6)]
        out = rerank_chunks(chunks, [1.0, 5.0, 2.0, 4.0, 3.0], k=2)
        assert [c.source for c in out] == ["s2.md", "s4.md"]
        assert [c.index for c in out] == [1, 2]

    def test_ties_keep_vector_order(self):
        # A degenerate reranker must be a no-op, not a shuffle.
        out = rerank_chunks(self._candidates(), [0.0, 0.0, 0.0], k=3)
        assert [c.source for c in out] == ["a.md", "b.md", "c.md"]

    def test_keeps_vector_score_and_records_rerank_score(self):
        out = rerank_chunks(self._candidates(), [0.1, 9.0, 3.0], k=3)
        assert out[0].score == 0.70  # still the Chroma relevance score
        assert out[0].rerank_score == 9.0
        assert out[0].ranking_score == 9.0

    def test_does_not_mutate_input(self):
        chunks = self._candidates()
        rerank_chunks(chunks, [0.1, 9.0, 3.0], k=3)
        assert [c.index for c in chunks] == [1, 2, 3]
        assert all(c.rerank_score is None for c in chunks)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            rerank_chunks(self._candidates(), [1.0, 2.0], k=3)

    def test_renumbering_keeps_citations_resolvable(self):
        out = rerank_chunks(self._candidates(), [0.1, 9.0, 3.0], k=3)
        assert "[1] source: b.md" in format_context(out)

        cited, dangling = RAGPipeline._resolve_citations("Fact [1]. Other [2].", out)
        assert [c.source for c in cited] == ["b.md", "c.md"]
        assert dangling == []

    @pytest.mark.parametrize(
        "provider,k,candidates,expected",
        [
            ("none", 5, 20, 5),  # disabled: fetch exactly what is asked for
            ("fastembed", 5, 20, 20),  # fan out, then narrow back down
            ("fastembed", 30, 20, 30),  # never return fewer than k
        ],
    )
    def test_candidate_count(self, provider, k, candidates, expected):
        settings = make_settings(rerank_provider=provider, rerank_candidates=candidates)
        assert candidate_count(settings, k) == expected

    def test_build_reranker_disabled_returns_none(self):
        assert build_reranker(make_settings(rerank_provider="none")) is None

    def test_build_reranker_rejects_unknown_provider(self):
        with pytest.raises(ValueError):
            build_reranker(make_settings(rerank_provider="nope"))

    def test_ranking_to_scores(self):
        assert ranking_to_scores([3, 1], 3) == [2.0, 0.0, 3.0]

    def test_ranking_to_scores_ignores_junk(self):
        # Duplicates, out-of-range numbers and omissions must not raise: the
        # ranking comes from a model and is not trustworthy input.
        assert ranking_to_scores([2, 2, 99, 0, -1], 3) == [0.0, 3.0, 0.0]


class TestTokenizer:
    def test_folds_currency_and_punctuation_into_bare_numbers(self):
        # The whole point of a lexical channel is matching figures verbatim.
        assert tokenize("£200") == tokenize("(200)") == ["200"]

    def test_lowercases_and_splits(self):
        assert tokenize("Atlas Ingest-Service") == ["atlas", "ingest", "service"]

    def test_empty_input(self):
        assert tokenize("   ") == []


class TestBM25:
    CORPUS = {
        "expenses.md::0": "Expenses at or above 200 pounds need department head approval.",
        "kitchen.md::0": "The office kitchen is cleaned every Friday by facilities.",
        "leave.md::0": "Employees get 25 days of annual leave each year.",
    }

    def _index(self) -> BM25Index:
        return BM25Index.build(list(self.CORPUS), list(self.CORPUS.values()))

    def test_ranks_the_lexically_matching_chunk_first(self):
        hits = self._index().search("expense approval above 200", k=3)
        assert hits[0][0] == "expenses.md::0"

    def test_exact_number_match_beats_topical_similarity(self):
        # The case dense retrieval loses: a bare figure with no semantic content.
        hits = self._index().search("200", k=3)
        assert [chunk_id for chunk_id, _ in hits] == ["expenses.md::0"]

    def test_drops_zero_scoring_chunks(self):
        # A channel with nothing to say contributes nothing to the fusion.
        assert self._index().search("quantum chromodynamics", k=3) == []

    def test_empty_query(self):
        assert self._index().search("", k=3) == []

    def test_respects_k(self):
        assert len(self._index().search("the", k=1)) <= 1


class TestReciprocalRankFusion:
    def test_agreement_between_channels_wins(self):
        # `b` is second in both lists; `a` and `c` are first in one and absent
        # from the other. Consensus should beat a single first place.
        scores = reciprocal_rank_fusion([["a", "b"], ["c", "b"]], k=60)
        assert scores["b"] > scores["a"]
        assert scores["b"] > scores["c"]

    def test_single_ranking_preserves_order(self):
        fused = fuse_rankings([["a", "b", "c"]], k=60)
        assert [doc_id for doc_id, _ in fused] == ["a", "b", "c"]

    def test_union_of_both_channels_is_kept(self):
        fused = fuse_rankings([["a"], ["b"]], k=60)
        assert {doc_id for doc_id, _ in fused} == {"a", "b"}

    def test_ties_break_on_first_ranking(self):
        # `a` and `b` each rank first in one list, so their RRF scores tie;
        # the dense channel is passed first and must win the tiebreak.
        fused = fuse_rankings([["a"], ["b"]], k=60)
        assert [doc_id for doc_id, _ in fused] == ["a", "b"]

    def test_empty_input(self):
        assert fuse_rankings([], k=60) == []


class TestNativeCitations:
    def _response(self) -> list[dict]:
        return [
            {"type": "text", "text": "Plain sentence with no citation. "},
            {
                "type": "text",
                "text": "The department head signs off.",
                "citations": [
                    {
                        "type": "char_location",
                        "cited_text": "requires department head approval",
                        "document_index": 1,
                        "document_title": "handbook.md",
                        "start_char_index": 10,
                        "end_char_index": 43,
                    }
                ],
            },
        ]

    def test_builds_one_citation_enabled_block_per_chunk(self):
        blocks = build_document_blocks([make_chunk(1, "a.md"), make_chunk(2, "b.md")])
        assert [b["type"] for b in blocks] == ["document", "document"]
        assert all(b["citations"] == {"enabled": True} for b in blocks)
        # document_index refers to block order, so titles must stay aligned.
        assert [b["title"] for b in blocks] == ["a.md", "b.md"]

    def test_answer_text_concatenates_all_blocks(self):
        assert answer_text(self._response()) == (
            "Plain sentence with no citation. The department head signs off."
        )

    def test_answer_text_accepts_a_plain_string(self):
        assert answer_text("just text") == "just text"

    def test_extracts_spans_in_order(self):
        found = extract_citations(self._response())
        assert len(found) == 1
        assert found[0]["cited_text"] == "requires department head approval"
        assert (found[0]["start_char_index"], found[0]["end_char_index"]) == (10, 43)

    def test_duplicate_spans_are_collapsed(self):
        blocks = self._response()
        found = extract_citations(blocks + [blocks[1]])
        assert len(found) == 1

    def test_resolves_document_index_to_the_retrieved_chunk(self):
        chunks = [make_chunk(1, "a.md"), make_chunk(2, "handbook.md")]
        cited, dangling = RAGPipeline._resolve_native_citations(self._response(), chunks)
        assert [c.source for c in cited] == ["handbook.md"]
        assert cited[0].cited_text == "requires department head approval"
        # Structurally impossible in this mode — the API can only cite an
        # attached document, so there is nothing to report as fabricated.
        assert dangling == []

    def test_out_of_range_document_index_is_ignored(self):
        cited, dangling = RAGPipeline._resolve_native_citations(
            self._response(), [make_chunk(1, "a.md")]
        )
        assert cited == []
        assert dangling == []


class TestJudgeOutputParsing:
    def test_plain_json(self):
        assert parse_json_object('{"groundedness": 5}')["groundedness"] == 5

    def test_fenced_json(self):
        text = 'Here you go:\n```json\n{"verdict": "grounded"}\n```'
        assert parse_json_object(text)["verdict"] == "grounded"

    def test_json_with_trailing_prose(self):
        text = '{"a": {"b": "}"}} and some commentary afterwards'
        assert parse_json_object(text) == {"a": {"b": "}"}}

    def test_raises_when_absent(self):
        with pytest.raises(ValueError):
            parse_json_object("no json at all")
