"""Evaluation loop: score answer groundedness and flag hallucinations.

Combines two kinds of signal per question:

* deterministic checks — did retrieval surface the expected source? do the inline
  [n] markers point at real passages? did the assistant abstain when it should?
* an LLM judge — decomposes the answer into atomic claims and marks each one
  supported / partially supported / unsupported against the retrieved context.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from rag.citations import verify_spans
from rag.config import Settings
from rag.llm import build_chat_model, message_text, parse_json_object
from rag.pipeline import RAGAnswer, RAGPipeline
from rag.prompts import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT_NATIVE,
    JUDGE_USER_TEMPLATE,
)
from rag.retriever import format_context

HALLUCINATION_VERDICTS = {"hallucinated", "partially_grounded"}


@dataclass
class EvalCase:
    id: str
    question: str
    answerable: bool = True
    expected_sources: list[str] = field(default_factory=list)
    must_contain: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class CaseResult:
    id: str
    question: str
    answer: str
    answerable: bool
    abstained: bool
    # deterministic checks
    retrieved_sources: list[str]
    cited_sources: list[str]
    dangling_citations: list[int]
    retrieval_hit: bool | None
    abstention_correct: bool
    must_contain_missing: list[str]
    # judge
    groundedness: int | None
    verdict: str
    citations_valid: bool | None
    # Native citation mode only: every reported span matched its source text.
    citation_spans_ok: bool | None
    unsupported_claims: list[str]
    judge_reasoning: str
    flagged: bool
    flags: list[str]


def load_cases(path: Path) -> list[EvalCase]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a YAML list of questions.")
    cases = []
    for i, item in enumerate(raw):
        cases.append(
            EvalCase(
                id=str(item.get("id", f"q{i + 1}")),
                question=item["question"],
                answerable=bool(item.get("answerable", True)),
                expected_sources=list(item.get("expected_sources", [])),
                must_contain=list(item.get("must_contain", [])),
                note=str(item.get("note", "")),
            )
        )
    return cases


class GroundednessJudge:
    """LLM-as-judge scoring an answer strictly against its retrieved context."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm = build_chat_model(settings, settings.judge_model)
        # Native-mode answers carry no inline markers, so the default prompt's
        # "check the [n] citations" step has nothing to check and its scoring
        # rubric would penalise their absence. Attribution is verified in code
        # there instead — see `verify_spans`.
        self.system_prompt = (
            JUDGE_SYSTEM_PROMPT_NATIVE
            if settings.native_citations
            else JUDGE_SYSTEM_PROMPT
        )

    def score(self, result: RAGAnswer) -> dict:
        response = self.llm.invoke(
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(
                    content=JUDGE_USER_TEMPLATE.format(
                        question=result.question,
                        context=format_context(result.retrieved),
                        answer=result.answer,
                    )
                ),
            ]
        )
        try:
            return parse_json_object(message_text(response))
        except (ValueError, json.JSONDecodeError) as exc:
            return {
                "groundedness": None,
                "verdict": "judge_error",
                "citations_valid": None,
                "unsupported_claims": [],
                "claims": [],
                "reasoning": f"Could not parse judge output: {exc}",
            }


def evaluate_case(case: EvalCase, pipeline: RAGPipeline, judge: GroundednessJudge) -> CaseResult:
    result = pipeline.answer(case.question)
    verdict = judge.score(result)

    retrieval_hit: bool | None = None
    if case.expected_sources:
        retrieval_hit = any(src in result.retrieved_sources for src in case.expected_sources)

    abstention_correct = result.abstained == (not case.answerable)

    missing = []
    if case.answerable and not result.abstained:
        lowered = result.answer.lower()
        missing = [s for s in case.must_contain if s.lower() not in lowered]

    unsupported = [str(c) for c in verdict.get("unsupported_claims", []) or []]
    groundedness = verdict.get("groundedness")
    # The native judge is not asked about attribution, so it reports no verdict
    # on it — the deterministic span check below covers that instead.
    citations_valid = verdict.get("citations_valid")

    spans_ok: bool | None = None
    if result.citations and any(c.cited_text is not None for c in result.citations):
        spans_ok = not verify_spans(result.citations, result.retrieved)

    flags: list[str] = []
    if unsupported:
        flags.append("unsupported_claims")
    if verdict.get("verdict") in HALLUCINATION_VERDICTS:
        flags.append(f"judge_verdict={verdict.get('verdict')}")
    if isinstance(groundedness, int) and groundedness <= 3:
        flags.append(f"low_groundedness={groundedness}")
    if result.dangling_citations:
        flags.append("dangling_citation")
    if citations_valid is False:
        flags.append("misattributed_citation")
    if spans_ok is False:
        flags.append("citation_span_mismatch")
    if not abstention_correct:
        flags.append(
            "should_have_abstained" if not case.answerable else "abstained_unexpectedly"
        )
    if missing:
        flags.append("missing_expected_content")
    if retrieval_hit is False:
        flags.append("retrieval_miss")
    if not result.citations and not result.abstained:
        flags.append("uncited_answer")

    return CaseResult(
        id=case.id,
        question=case.question,
        answer=result.answer,
        answerable=case.answerable,
        abstained=result.abstained,
        retrieved_sources=result.retrieved_sources,
        cited_sources=result.sources,
        dangling_citations=result.dangling_citations,
        retrieval_hit=retrieval_hit,
        abstention_correct=abstention_correct,
        must_contain_missing=missing,
        groundedness=groundedness if isinstance(groundedness, int) else None,
        verdict=str(verdict.get("verdict", "unknown")),
        citations_valid=citations_valid if isinstance(citations_valid, bool) else None,
        citation_spans_ok=spans_ok,
        unsupported_claims=unsupported,
        judge_reasoning=str(verdict.get("reasoning", "")),
        flagged=bool(flags),
        flags=flags,
    )


def summarize(results: list[CaseResult]) -> dict:
    n = len(results)
    scored = [r.groundedness for r in results if r.groundedness is not None]
    with_recall = [r for r in results if r.retrieval_hit is not None]
    cite_checked = [r for r in results if r.citations_valid is not None]
    span_checked = [r for r in results if r.citation_spans_ok is not None]

    def pct(count: int, total: int) -> float:
        return round(100.0 * count / total, 1) if total else 0.0

    return {
        "questions": n,
        "mean_groundedness": round(sum(scored) / len(scored), 2) if scored else None,
        "grounded_rate_pct": pct(sum(r.verdict == "grounded" for r in results), n),
        "hallucination_rate_pct": pct(sum(bool(r.unsupported_claims) for r in results), n),
        "flagged_rate_pct": pct(sum(r.flagged for r in results), n),
        "citation_validity_pct": pct(
            sum(bool(r.citations_valid) for r in cite_checked), len(cite_checked)
        ),
        # Native mode only; 0.0 when the judge, not the API, reported citations.
        "citation_span_integrity_pct": pct(
            sum(bool(r.citation_spans_ok) for r in span_checked), len(span_checked)
        ),
        "dangling_citation_count": sum(len(r.dangling_citations) for r in results),
        "abstention_accuracy_pct": pct(sum(r.abstention_correct for r in results), n),
        "retrieval_recall_pct": pct(
            sum(bool(r.retrieval_hit) for r in with_recall), len(with_recall)
        ),
        "must_contain_pass_pct": pct(
            sum(not r.must_contain_missing for r in results), n
        ),
    }


def render_markdown(summary: dict, results: list[CaseResult], settings: Settings) -> str:
    lines = [
        "# RAG evaluation report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Answer model: `{settings.answer_model}` · Judge model: `{settings.judge_model}` · "
        f"top_k={settings.top_k} · chunk={settings.chunk_size}/{settings.chunk_overlap} · "
        f"embeddings=`{settings.embedding_provider}:{settings.embedding_model}` · "
        f"rerank=`{settings.reranker_label}` · retrieval=`{settings.retrieval_mode}` · "
        f"citations=`{settings.citation_mode}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for key, value in summary.items():
        lines.append(f"| {key.replace('_', ' ')} | {value} |")

    lines += [
        "",
        "## Per-question results",
        "",
        "| ID | Groundedness | Verdict | Cited sources | Flags |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in results:
        flags = ", ".join(r.flags) if r.flags else "—"
        sources = ", ".join(r.cited_sources) if r.cited_sources else "—"
        lines.append(
            f"| {r.id} | {r.groundedness if r.groundedness is not None else '—'} "
            f"| {r.verdict} | {sources} | {flags} |"
        )

    flagged = [r for r in results if r.flagged]
    if flagged:
        lines += ["", "## Flagged answers", ""]
        for r in flagged:
            lines += [
                f"### {r.id} — {r.question}",
                "",
                f"**Flags:** {', '.join(r.flags)}",
                "",
                f"**Answer:** {r.answer}",
                "",
                f"**Judge:** {r.judge_reasoning}",
            ]
            if r.unsupported_claims:
                lines += ["", "**Unsupported claims:**"]
                lines += [f"- {c}" for c in r.unsupported_claims]
            lines.append("")

    return "\n".join(lines) + "\n"


def run_evaluation(
    settings: Settings, questions_path: Path, report_dir: Path
) -> tuple[dict, list[CaseResult], Path]:
    cases = load_cases(questions_path)
    pipeline = RAGPipeline(settings)
    pipeline.retriever.ensure_indexed()
    judge = GroundednessJudge(settings)

    results = [evaluate_case(case, pipeline, judge) for case in cases]
    summary = summarize(results)

    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = report_dir / f"eval-{stamp}.json"
    md_path = report_dir / f"eval-{stamp}.md"

    json_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "settings": {
                    "answer_model": settings.answer_model,
                    "judge_model": settings.judge_model,
                    "embedding_provider": settings.embedding_provider,
                    "embedding_model": settings.embedding_model,
                    "top_k": settings.top_k,
                    "chunk_size": settings.chunk_size,
                    "chunk_overlap": settings.chunk_overlap,
                    "rerank_provider": settings.rerank_provider,
                    "rerank_model": settings.rerank_model,
                    "rerank_candidates": settings.rerank_candidates,
                    "retrieval_mode": settings.retrieval_mode,
                    "rrf_k": settings.rrf_k,
                    "citation_mode": settings.citation_mode,
                },
                "results": [asdict(r) for r in results],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(summary, results, settings), encoding="utf-8")

    return summary, results, md_path
