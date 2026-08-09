"""Command line entry point.

    python -m rag.cli ingest [--reset]
    python -m rag.cli ask "your question" [-k 5] [--show-context]
    python -m rag.cli eval [--questions eval/questions.yaml]
    python -m rag.cli status
    python -m rag.cli serve [--host 127.0.0.1] [--port 8000] [--reload]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from rag.config import PROJECT_ROOT, Settings

console = Console()


def cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    from rag.ingest import ingest

    with console.status("[bold]Loading, chunking and embedding documents…"):
        report = ingest(settings, reset=args.reset)

    console.print(
        Panel.fit(
            f"Files ingested: [bold]{report.files}[/]\n"
            f"Chunks written: [bold]{report.chunks}[/]\n"
            f"Collection size: [bold]{report.collection_count}[/]\n"
            f"Store: [dim]{settings.persist_dir}[/]",
            title="Ingestion complete",
            border_style="green",
        )
    )
    if report.skipped:
        console.print(
            f"[yellow]Skipped {len(report.skipped)} unsupported file(s):[/] "
            + ", ".join(p.name for p in report.skipped[:10])
        )
    if report.failed:
        console.print(f"[red]{len(report.failed)} file(s) could not be read:[/]")
        for path, reason in report.failed:
            console.print(f"  [red]•[/] {path.name} — {reason}")
        return 2
    return 0


def cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    from rag.pipeline import RAGPipeline

    pipeline = RAGPipeline(settings)
    pipeline.retriever.ensure_indexed()

    with console.status("[bold]Retrieving and generating…"):
        result = pipeline.answer(args.question, k=args.k)

    console.print()
    console.print(Panel(Markdown(result.answer), title="Answer", border_style="cyan"))

    if result.citations:
        # The extra column appears only when a reranker actually ran, so the
        # default output stays exactly as it was.
        reranked = any(c.rerank_score is not None for c in result.citations)
        # In native mode the API reports the exact span, which is far more
        # useful than our own snippet of the whole chunk — show that instead.
        native = any(c.cited_text is not None for c in result.citations)
        table = Table(title="Citations", show_lines=False, header_style="bold")
        table.add_column("#", justify="right", width=3)
        table.add_column("Source")
        table.add_column("Score", justify="right", width=6)
        if reranked:
            table.add_column("Rerank", justify="right", width=7)
        if native:
            table.add_column("Chars", justify="right", width=11)
        table.add_column("Cited text" if native else "Snippet", overflow="fold")
        for c in result.citations:
            row = [f"[{c.index}]", c.source, f"{c.score:.3f}"]
            if reranked:
                # Signed: cross-encoder logits are commonly negative.
                row.append("—" if c.rerank_score is None else f"{c.rerank_score:+.2f}")
            if native:
                row.append(
                    "—"
                    if c.start_char_index is None
                    else f"{c.start_char_index}–{c.end_char_index}"
                )
            table.add_row(*row, c.cited_text if native else c.snippet)
        console.print(table)
    elif not result.abstained:
        console.print("[yellow]Warning: the answer contains no citations.[/]")

    if result.dangling_citations:
        console.print(
            "[red]Fabricated citation marker(s):[/] "
            + ", ".join(f"[{i}]" for i in result.dangling_citations)
        )

    if args.show_context:
        from rag.retriever import format_context

        console.print(
            Panel(format_context(result.retrieved), title="Retrieved context", border_style="dim")
        )

    if result.usage:
        console.print(
            f"[dim]tokens in/out: {result.usage.get('input_tokens', '?')}"
            f"/{result.usage.get('output_tokens', '?')}[/]"
        )
    return 0


def cmd_eval(args: argparse.Namespace, settings: Settings) -> int:
    from rag.evaluate import run_evaluation

    questions_path = Path(args.questions)
    if not questions_path.is_absolute():
        questions_path = PROJECT_ROOT / questions_path
    report_dir = PROJECT_ROOT / args.report_dir

    console.print(f"[bold]Running evaluation over[/] {questions_path}")
    with console.status("[bold]Answering and judging…"):
        summary, results, md_path = run_evaluation(settings, questions_path, report_dir)

    table = Table(title="Evaluation summary", header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in summary.items():
        table.add_row(key.replace("_", " "), str(value))
    console.print(table)

    detail = Table(title="Per question", header_style="bold")
    detail.add_column("ID")
    detail.add_column("Ground.", justify="right")
    detail.add_column("Verdict")
    detail.add_column("Flags", overflow="fold")
    for r in results:
        style = "red" if r.flagged else "green"
        detail.add_row(
            r.id,
            str(r.groundedness if r.groundedness is not None else "—"),
            r.verdict,
            ", ".join(r.flags) if r.flags else "—",
            style=style,
        )
    console.print(detail)
    console.print(f"[dim]Report written to {md_path}[/]")

    if args.fail_under is not None:
        mean = summary.get("mean_groundedness")
        if mean is None or mean < args.fail_under:
            console.print(
                f"[red]Mean groundedness {mean} is below threshold {args.fail_under}.[/]"
            )
            return 1
    return 0


def cmd_status(_: argparse.Namespace, settings: Settings) -> int:
    from rag.store import collection_size, get_vectorstore

    store = get_vectorstore(settings)
    table = Table(title="Configuration", header_style="bold")
    table.add_column("Setting")
    table.add_column("Value")
    table.add_row("docs dir", str(settings.docs_dir))
    table.add_row("persist dir", str(settings.persist_dir))
    table.add_row("collection", settings.collection_name)
    table.add_row("indexed chunks", str(collection_size(store)))
    table.add_row("embeddings", f"{settings.embedding_provider}:{settings.embedding_model}")
    table.add_row("chunking", f"{settings.chunk_size} / {settings.chunk_overlap}")
    table.add_row("top_k", str(settings.top_k))
    table.add_row("retrieval", settings.retrieval_mode)
    if settings.hybrid_enabled:
        table.add_row("rrf k", str(settings.rrf_k))
    table.add_row("reranker", settings.reranker_label)
    if settings.rerank_enabled:
        table.add_row("rerank candidates", str(settings.rerank_candidates))
    table.add_row("citations", settings.citation_mode)
    table.add_row("answer model", settings.answer_model)
    table.add_row("judge model", settings.judge_model)
    console.print(table)
    return 0


def cmd_serve(args: argparse.Namespace, _: Settings) -> int:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on the installed extras
        raise ImportError("uvicorn is not installed. Run `uv sync`.") from exc

    console.print(
        Panel.fit(
            f"API:        [bold]http://{args.host}:{args.port}[/]\n"
            f"Swagger UI: [bold]http://{args.host}:{args.port}/docs[/]\n"
            f"OpenAPI:    [dim]http://{args.host}:{args.port}/openapi.json[/]",
            title="Serving",
            border_style="cyan",
        )
    )
    # Import string rather than the app object: --reload needs a re-importable target.
    uvicorn.run("rag.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag", description="RAG Document Assistant — grounded Q&A with citations"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Load, chunk and embed documents into Chroma")
    p_ingest.add_argument(
        "--reset", action="store_true", help="Drop the collection before ingesting"
    )
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="Ask a question against the indexed documents")
    p_ask.add_argument("question")
    p_ask.add_argument("-k", type=int, default=None, help="Number of passages to retrieve")
    p_ask.add_argument(
        "--show-context", action="store_true", help="Print the retrieved passages"
    )
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("eval", help="Run the groundedness evaluation loop")
    p_eval.add_argument("--questions", default="eval/questions.yaml")
    p_eval.add_argument("--report-dir", default="eval/reports")
    p_eval.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="Exit non-zero if mean groundedness falls below this (for CI)",
    )
    p_eval.set_defaults(func=cmd_eval)

    p_status = sub.add_parser("status", help="Show configuration and index size")
    p_status.set_defaults(func=cmd_status)

    p_serve = sub.add_parser("serve", help="Serve the HTTP API with Swagger UI at /docs")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true", help="Reload on source changes")
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = Settings.from_env()
        return args.func(args, settings)
    except (RuntimeError, FileNotFoundError, ValueError, ImportError) as exc:
        console.print(f"[red]Error:[/] {exc}")
        return 1
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted.[/]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
