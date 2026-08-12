"""A scripted walkthrough of what this pipeline does and why.

Demos fail live for boring reasons: a typo, a cold model download, an empty
index, a missing key. This runs the whole narrative from one command, checks
its own preconditions first, and degrades to the offline half rather than
dying when there is no API key.

The order is deliberate. Answering a question is table stakes and goes first
to get it out of the way; everything after it is about the system being able
to show its working — declining when it should, citing spans that can be
checked against the source, and retrieving what a single channel would miss.
"""

from __future__ import annotations

import os
from dataclasses import replace

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rag.config import Settings
from rag.rerank import renumber
from rag.retriever import Retriever, RetrievedChunk

# Drawn from eval/questions.yaml, so every question here is one the evaluation
# loop already scores — the demo and the metrics talk about the same cases.
GROUNDED_QUESTION = "Who has to approve an expense claim of £350?"
UNANSWERABLE_QUESTION = "How much does Atlas cost per million ingested records?"
CITED_QUESTION = "How long are production database credentials valid?"
# A bare figure: no semantic content for an embedding to hold on to, which is
# exactly the shape of query the lexical channel exists for.
LEXICAL_QUESTION = "45 days"

# The questions above are about these documents specifically. Pointed at any
# other corpus the walkthrough does not fail — it abstains its way through
# every step while the commentary still claims a grounded answer was given.
# Confidently narrating something that did not happen is the exact failure
# this project exists to prevent, so the demo refuses to do it.
REQUIRED_SOURCES = {
    "employee-handbook.md",
    "security-policy.md",
    "atlas-product-spec.md",
}


def _heading(console: Console, number: int, title: str, why: str) -> None:
    console.print()
    console.print(Panel(f"[bold]{title}[/]\n[dim]{why}[/]", title=f"Step {number}", border_style="cyan"))


def _takeaway(console: Console, text: str) -> None:
    console.print(f"\n  [bold green]→[/] {text}")


def _wait(console: Console, pause: bool) -> None:
    if not pause:
        return
    console.print("\n[dim]  (enter to continue)[/]", end="")
    try:
        input()
    except EOFError:  # piped stdin — behave like --no-pause
        console.print()


def preflight(settings: Settings, console: Console) -> bool:
    """Check what the demo needs before it needs it.

    Returns whether the API-backed steps can run. A missing key is not fatal:
    retrieval is the half that runs offline, and it is also the half with the
    most to show.
    """
    from rag.store import all_chunks, collection_size, get_vectorstore

    table = Table(title="Pre-flight", header_style="bold", show_lines=False)
    table.add_column("Check")
    table.add_column("Result")

    store = get_vectorstore(settings)
    indexed = collection_size(store)
    table.add_row("indexed chunks", str(indexed) if indexed else "[red]0 — run `rag ingest` first[/]")

    missing: set[str] = set()
    if indexed:
        _, _, metadatas = all_chunks(store)
        present = {m.get("source") for m in metadatas}
        missing = REQUIRED_SOURCES - present
    table.add_row(
        "collection",
        settings.collection_name
        if not missing
        else f"[red]{settings.collection_name} — not the corpus these questions are about[/]",
    )

    has_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    table.add_row("ANTHROPIC_API_KEY", "set" if has_key else "[yellow]unset — offline steps only[/]")

    model_ok, model_detail = (True, "") if not has_key else _check_model(settings.answer_model)
    table.add_row(
        "answer model",
        settings.answer_model if model_ok else f"[red]{settings.answer_model} — {model_detail}[/]",
    )
    table.add_row("retrieval", f"{settings.retrieval_mode} [dim](the demo varies this)[/]")
    table.add_row(
        "cross-encoder",
        "[dim]downloaded on first use (~90 MB), then cached[/]",
    )
    console.print(table)

    if not indexed:
        raise RuntimeError("The vector store is empty. Run `rag ingest` before the demo.")
    if missing:
        raise RuntimeError(
            f"Collection {settings.collection_name!r} is missing "
            f"{', '.join(sorted(missing))}. The walkthrough asks scripted questions about "
            "those documents; against another corpus it abstains through every step while "
            "still narrating a grounded answer. Point DOCS_DIR / COLLECTION_NAME / "
            "PERSIST_DIR back at the sample corpus (or unset them) and re-run."
        )
    if has_key and not model_ok:
        raise RuntimeError(
            f"ANSWER_MODEL={settings.answer_model!r} is not a model this API key can reach "
            f"({model_detail}). Fix it in .env before demoing — for example "
            f"ANSWER_MODEL=claude-opus-5 or claude-haiku-4-5-20251001."
        )
    return has_key


def _check_model(model: str) -> tuple[bool, str]:
    """Ask the Models API whether this ID resolves.

    Catching a bad ANSWER_MODEL here rather than three steps in is the whole
    point of a pre-flight: a 404 mid-demo is a wall of traceback in front of an
    audience. Any failure that is not a clean "no such model" is treated as
    fine, so a network blip does not block a demo that would otherwise run.
    """
    try:
        import anthropic

        anthropic.Anthropic().models.retrieve(model)
        return True, ""
    except Exception as exc:  # noqa: BLE001 - any SDK error type
        if type(exc).__name__ == "NotFoundError":
            return False, "no such model"
        return True, ""


def _citation_table(result, title: str = "Citations") -> Table:
    table = Table(title=title, header_style="bold")
    table.add_column("#", justify="right", width=3)
    table.add_column("Source")
    table.add_column("Score", justify="right", width=6)
    table.add_column("Snippet", overflow="fold")
    for citation in result.citations:
        table.add_row(
            f"[{citation.index}]", citation.source, f"{citation.score:.3f}", citation.snippet
        )
    return table


def step_grounded(console: Console, settings: Settings, number: int) -> None:
    from rag.pipeline import RAGPipeline

    _heading(
        console,
        number,
        "A grounded answer, with its sources",
        "Every factual sentence is drawn from the retrieved passages and cited back to them.",
    )
    console.print(f"  [cyan]?[/] {GROUNDED_QUESTION}\n")
    with console.status("[bold]retrieving and generating…"):
        result = RAGPipeline(settings).answer(GROUNDED_QUESTION)
    console.print(Panel(result.answer, border_style="dim"))
    console.print(_citation_table(result))
    _takeaway(console, "The answer is traceable: each claim points at the passage behind it.")


def step_abstention(console: Console, settings: Settings, number: int) -> None:
    from rag.pipeline import RAGPipeline

    _heading(
        console,
        number,
        "Declining is a first-class outcome",
        "The corpus has no pricing in it. A RAG system that never declines is not grounded — it is lucky.",
    )
    console.print(f"  [cyan]?[/] {UNANSWERABLE_QUESTION}\n")
    with console.status("[bold]retrieving and generating…"):
        result = RAGPipeline(settings).answer(UNANSWERABLE_QUESTION)
    console.print(Panel(result.answer, border_style="yellow"))
    console.print(f"  abstained: [bold]{result.abstained}[/]")
    _takeaway(
        console,
        "Retrieval still returned passages — the model was asked to answer only from them, and refused.",
    )


def step_native_citations(console: Console, settings: Settings, number: int) -> None:
    from rag.citations import verify_spans
    from rag.pipeline import RAGPipeline

    _heading(
        console,
        number,
        "Citations that can be checked, not trusted",
        "The Citations API returns the character span behind each claim. The span comes from the "
        "serving layer, not from the model.",
    )
    native = replace(settings, citation_mode="native")
    console.print(f"  [cyan]?[/] {CITED_QUESTION}\n")
    with console.status("[bold]retrieving and generating…"):
        result = RAGPipeline(native).answer(CITED_QUESTION)
    console.print(Panel(result.answer, border_style="dim"))

    table = Table(title="Reported spans", header_style="bold")
    table.add_column("Source")
    table.add_column("Chars", justify="right", width=11)
    table.add_column("Cited text", overflow="fold")
    for citation in result.citations:
        table.add_row(
            citation.source,
            f"{citation.start_char_index}–{citation.end_char_index}",
            citation.cited_text,
        )
    console.print(table)

    # The point of the step: re-derive every span from the source ourselves.
    console.print("\n  [bold]Verifying each span against the source text[/]")
    by_id = {chunk.chunk_id: chunk.text for chunk in result.retrieved}
    for citation in result.citations:
        source = by_id[citation.chunk_id]
        span = source[citation.start_char_index : citation.end_char_index]
        ok = span == citation.cited_text
        mark = "[green]match[/]" if ok else "[red]MISMATCH[/]"
        console.print(
            f"    {citation.chunk_id}"
            f"[{citation.start_char_index}:{citation.end_char_index}] == cited_text  {mark}"
        )
    mismatched = verify_spans(result.citations, result.retrieved)
    console.print(f"\n  dangling citations: [bold]{result.dangling_citations}[/]  "
                  f"span mismatches: [bold]{len(mismatched)}[/]")
    _takeaway(
        console,
        "A fabricated reference stops being expressible here, rather than being detected afterwards.",
    )


def _ranking_table(
    title: str, chunks: list[RetrievedChunk], highlight: set[str], show_rerank: bool = False
) -> Table:
    table = Table(title=title, header_style="bold")
    table.add_column("#", justify="right", width=3)
    table.add_column("Source")
    table.add_column("Channel", width=8)
    if show_rerank:
        table.add_column("Rerank", justify="right", width=7)
    table.add_column("Snippet", overflow="fold")
    for chunk in chunks:
        row = [str(chunk.index), chunk.source, chunk.channel]
        if show_rerank:
            row.append("—" if chunk.rerank_score is None else f"{chunk.rerank_score:+.2f}")
        table.add_row(
            *row,
            chunk.snippet(70),
            style="green" if chunk.chunk_id in highlight else None,
        )
    return table


def step_hybrid(console: Console, settings: Settings, number: int) -> None:
    _heading(
        console,
        number,
        "The retrieval pipeline, one stage at a time",
        "Same query, three configurations. Embeddings match on meaning, which is why a bare "
        "figure slips past them; BM25 cannot generalise but pins an exact term; the "
        "cross-encoder then reads query and passage together. No LLM calls in this step.",
    )
    hybrid = replace(settings, retrieval_mode="hybrid")
    # One Retriever per configuration would load the embedding model twice, so
    # the first two rankings come from one instance. `_dense` is the same first
    # stage `search` runs before fusion.
    retriever = Retriever(hybrid)
    retriever.ensure_indexed()

    console.print(f"  [cyan]?[/] {LEXICAL_QUESTION}   [dim](a bare figure — no semantics to match on)[/]\n")
    with console.status("[bold]searching…"):
        dense = renumber(retriever._dense(LEXICAL_QUESTION, hybrid.top_k))
        fused = retriever.search(LEXICAL_QUESTION)

    dense_ids = {chunk.chunk_id for chunk in dense}
    rescued = {chunk.chunk_id for chunk in fused if chunk.chunk_id not in dense_ids}

    console.print(_ranking_table("1. Dense only", dense, set()))
    console.print(_ranking_table("2. Hybrid — dense + BM25, fused by RRF", fused, rescued))

    if rescued:
        console.print(
            f"\n  [green]{len(rescued)} passage(s) highlighted above were never returned by the "
            "dense channel.[/]"
        )

    with console.status("[bold]reranking (first run downloads the cross-encoder)…"):
        reranked = Retriever(
            replace(
                hybrid,
                rerank_provider="fastembed",
                rerank_model="Xenova/ms-marco-MiniLM-L-6-v2",
            )
        ).search(LEXICAL_QUESTION)
    console.print(
        _ranking_table("3. Hybrid + cross-encoder rerank", reranked, rescued, show_rerank=True)
    )
    _takeaway(
        console,
        "The passage the dense channel missed entirely ends up ranked first — "
        "each stage fixes a different failure.",
    )


def step_evaluation(console: Console, settings: Settings, number: int) -> None:
    from rag.config import PROJECT_ROOT

    _heading(
        console,
        number,
        "None of this is claimed — it is measured",
        "An evaluation loop scores groundedness with an LLM judge and checks retrieval, "
        "abstention and citations deterministically.",
    )
    reports = sorted((PROJECT_ROOT / "eval" / "reports").glob("eval-*.md"))
    table = Table(header_style="bold")
    table.add_column("Metric")
    table.add_column("What it catches")
    for metric, meaning in [
        ("mean_groundedness", "claims not supported by the retrieved passages"),
        ("retrieval_recall_pct", "the expected source never reaching the model"),
        ("abstention_accuracy_pct", "answering when it should have declined"),
        ("dangling_citation_count", "markers pointing outside the context block"),
        ("citation_span_integrity_pct", "a reported span not matching its source"),
    ]:
        table.add_row(metric, meaning)
    console.print(table)

    if reports:
        console.print(f"\n  latest report: [bold]{reports[-1].relative_to(PROJECT_ROOT)}[/]")
    else:
        console.print("\n  [dim]no report yet — run `rag eval` to generate one[/]")
    _takeaway(
        console,
        "Every retrieval setting is off by default, so each one can be A/B'd against the baseline.",
    )


def run_demo(settings: Settings, console: Console, pause: bool = True) -> int:
    console.print(
        Panel(
            "[bold]RAG document assistant[/]\n"
            "[dim]Grounded answers over a private corpus, with citations that can be verified "
            "and retrieval that is measured rather than asserted.[/]",
            border_style="cyan",
        )
    )
    online = preflight(settings, console)

    steps = [step_hybrid, step_evaluation]
    if online:
        steps = [step_grounded, step_abstention, step_native_citations, *steps]
    else:
        console.print(
            "\n[yellow]No ANTHROPIC_API_KEY — running the retrieval steps only.[/]"
        )

    failed = 0
    for number, step in enumerate(steps, start=1):
        _wait(console, pause)
        try:
            step(console, settings, number)
        except Exception as exc:  # noqa: BLE001 - a demo must not end in a traceback
            # Pre-flight catches the predictable failures; anything left is a
            # surprise, and a wall of stack trace in front of an audience is
            # worse than a one-line apology and the next step.
            failed += 1
            console.print(
                f"\n  [red]This step could not run:[/] {type(exc).__name__}: {exc}"
            )
            console.print("  [dim]continuing with the remaining steps[/]")

    console.print()
    console.print(
        Panel(
            "[bold]Done.[/] [dim]Source: rag/ — tests: `uv run pytest`[/]",
            border_style="red" if failed else "cyan",
        )
    )
    return 1 if failed else 0
