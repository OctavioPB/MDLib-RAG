"""`biblio` command-line entry point.

Only light modules are imported at the top level. ML libraries are imported inside the commands that
need them, after `apply_offline_env` has run (`offline-guarantee` skill, layer 1).
"""

from __future__ import annotations

import io
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from biblio.config import TUTOR_LEVELS, Config, ConfigError, load_config
from biblio.offline import allow_model_downloads, apply_offline_env

if TYPE_CHECKING:
    from http.server import ThreadingHTTPServer

    from biblio.index.embedder import DeviceChoice
    from biblio.index.indexer import IndexStats
    from biblio.index.watch import Watcher
    from biblio.launcher import LLMServer
    from biblio.retrieve.types import Retriever
    from biblio.web.engine import Engine

app = typer.Typer(
    help="100% local search, tutor and writing assistant over a Markdown library.",
    no_args_is_help=True,
)
models_app = typer.Typer(help="Manage local model weights.", no_args_is_help=True)
app.add_typer(models_app, name="models")
library_app = typer.Typer(help="Where the library is, and moving it (ADR-039).")
app.add_typer(library_app, name="library")

console = Console()

# Subcommands allowed to reach the network. Everything else runs with the offline env applied.
_NETWORKED = {("models", "pull")}


def _force_utf8_output() -> None:
    """Legacy Windows code pages cannot print breadcrumb separators or check marks."""
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper) and stream.encoding.lower() != "utf-8":
            stream.reconfigure(encoding="utf-8")


def _config(ctx: typer.Context) -> Config:
    config = ctx.find_root().obj
    assert isinstance(config, Config)
    return config


@app.callback()
def main(ctx: typer.Context) -> None:
    _force_utf8_output()
    try:
        config = load_config()
    except ConfigError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(2) from exc
    ctx.obj = config
    if ctx.invoked_subcommand != "models" and config.runtime.offline:
        apply_offline_env()
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=Console(stderr=True), show_time=False, show_path=False)],
    )


@app.command()
def chunks(
    ctx: typer.Context,
    file: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Markdown file to chunk.")
    ],
    summary: Annotated[
        bool, typer.Option("--summary", help="Only the table, without chunk texts.")
    ] = False,
) -> None:
    """Print the chunks of one file for manual inspection. Offline."""
    from biblio.ingest.chunker import CHUNKER_VERSION, chunk_document
    from biblio.ingest.parser import parse_file
    from biblio.ingest.tokens import load_token_counter
    from biblio.models import ModelNotAvailableError

    config = _config(ctx)
    try:
        counter = load_token_counter(config)
    except ModelNotAvailableError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    doc = parse_file(file, _doc_path(file, config))
    result = chunk_document(doc, counter, config.chunking)
    c = config.chunking
    console.print(
        f"[bold]{doc.doc_path}[/bold] · {len(result)} chunks · chunker v{CHUNKER_VERSION} · "
        f"tokens min/target/max {c.min_tokens}/{c.target_tokens}/{c.max_tokens}"
    )
    table = Table("#", "lines", "tokens", "heading path", "links")
    for chunk in result:
        size = str(chunk.token_count)
        if chunk.token_count > c.max_tokens:
            size = f"[yellow]{size}[/yellow]"
        elif chunk.token_count < c.min_tokens:
            size = f"[cyan]{size}[/cyan]"
        table.add_row(
            str(chunk.ordinal),
            f"{chunk.start_line}-{chunk.end_line}",
            size,
            chunk.breadcrumb,
            str(len(chunk.links)),
        )
    console.print(table)
    if summary:
        return
    for chunk in result:
        title = f"[{chunk.ordinal}] {chunk.breadcrumb} · L{chunk.start_line}-{chunk.end_line}"
        console.print(Panel(Text(chunk.text), title=title, title_align="left"))


def _doc_path(file: Path, config: Config) -> str:
    """Path relative to the library root when the file is inside it, else the file name."""
    library = config.paths.library
    resolved = file.resolve()
    if library is not None and resolved.is_relative_to(library.resolve()):
        return resolved.relative_to(library.resolve()).as_posix()
    return file.name


@app.command()
def index(
    ctx: typer.Context,
    library: Annotated[
        Path | None,
        typer.Argument(
            exists=True, file_okay=False, help="Library root (default: paths.library in config)."
        ),
    ] = None,
    full: Annotated[
        bool, typer.Option("--full", help="Rebuild the whole index from scratch.")
    ] = False,
    device: Annotated[
        str | None,
        typer.Option("--device", help="Override embedding.index_device (e.g. cpu, cuda)."),
    ] = None,
    ignore_llm: Annotated[
        bool,
        typer.Option("--ignore-llm", help="Use the GPU even if an LLM server is running."),
    ] = False,
) -> None:
    """Index the library incrementally (only changed files are re-embedded). Offline."""
    from biblio.index.embedder import resolve_index_device
    from biblio.index.indexer import LLMServerRunningError, ensure_gpu_free

    config = _config(ctx)
    root = library or config.paths.library
    if root is None:
        console.print("[red]No library path: pass one or set paths.library in config/local.toml.")
        raise typer.Exit(2)

    try:
        choice = resolve_index_device(config, device)
        if not ignore_llm:
            ensure_gpu_free(config, choice)
    except (ValueError, LLMServerRunningError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    from biblio.index.indexer import IndexLockedError, IndexMismatchError
    from biblio.models import ModelNotAvailableError

    try:
        stats = _run_index(config, root, full, choice)
    except (IndexLockedError, IndexMismatchError, ModelNotAvailableError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    if stats.failed:
        raise typer.Exit(1)


def _run_index(
    config: Config, root: Path, full: bool, choice: DeviceChoice, relocate: bool = False
) -> IndexStats:
    """Index with a progress bar and print the summary. Index errors propagate to the caller."""
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    from biblio.index.embedder import load_index_embedder
    from biblio.index.indexer import index_library
    from biblio.ingest.tokens import load_token_counter

    mode = "full rebuild" if full else "incremental"
    console.print(
        f"Indexing [bold]{root}[/bold] → {config.paths.index_db} "
        f"({mode}, embeddings on {choice.device} {choice.dtype})"
    )
    columns = (
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("· {task.fields[chunks]} chunks"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    with Progress(*columns, console=console, transient=False) as bar:
        task = bar.add_task("files", total=None, chunks=0)
        written = 0

        def advance(chunks: int) -> None:
            nonlocal written
            written += chunks
            bar.update(task, advance=1, chunks=written)

        stats = index_library(
            config,
            root,
            full=full,
            dtype=choice.dtype,
            embedder_factory=lambda: load_index_embedder(config, choice),
            counter_factory=lambda: load_token_counter(config),
            progress=advance,
            on_start=lambda n: bar.update(task, total=n),
            relocate=relocate,
        )

    rate = f"{stats.chunks_written / stats.embed_seconds:.1f}" if stats.embed_seconds else "—"
    table = Table("", "value", show_header=False)
    table.add_row(
        "documents",
        f"{stats.added} added, {stats.updated} updated, "
        f"{stats.deleted} deleted, {stats.unchanged} unchanged",
    )
    table.add_row("chunks written", str(stats.chunks_written))
    table.add_row("index now", f"{stats.total_documents} documents, {stats.total_chunks} chunks")
    table.add_row("time", f"{stats.seconds:.1f}s total, {stats.embed_seconds:.1f}s embedding")
    table.add_row("throughput", f"{rate} chunks/s while embedding")
    if stats.failed:
        table.add_row("[red]failed[/red]", ", ".join(stats.failed))
    console.print(table)
    return stats


@app.command()
def watch(
    ctx: typer.Context,
    library: Annotated[
        Path | None,
        typer.Argument(
            exists=True, file_okay=False, help="Library root (default: paths.library in config)."
        ),
    ] = None,
) -> None:
    """Reindex automatically when library files change (polling, ADR-018). Ctrl+C stops it."""
    from biblio.index.embedder import resolve_index_device
    from biblio.index.indexer import IndexMismatchError
    from biblio.models import ModelNotAvailableError

    config = _config(ctx)
    root = library or config.paths.library
    if root is None:
        console.print("[red]No library path: pass one or set paths.library in config/local.toml.")
        raise typer.Exit(2)
    try:
        preferred = resolve_index_device(config)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    s = config.watch
    console.print(
        f"Watching [bold]{root}[/bold] every {s.interval_s:g}s (settle {s.settle_s:g}s). "
        "Ctrl+C to stop."
    )
    watcher = _make_watcher(config, root, preferred)
    try:
        watcher.run_forever()
    except KeyboardInterrupt:
        console.print("Stopped.")
    except (IndexMismatchError, ModelNotAvailableError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc


def _make_watcher(
    config: Config,
    root: Path,
    preferred: DeviceChoice,
    current: Callable[[], Config] | None = None,
) -> Watcher:
    """The ADR-018 watcher: GPU when free, CPU while the LLM server runs; prints each run.
    `current` gives the app's config at each check, so a library moved from the page (ADR-039)
    is watched at its new place."""

    def now() -> tuple[Config, Path]:
        cfg = current() if current is not None else config
        return cfg, cfg.paths.library or root

    from datetime import datetime

    from biblio.index.embedder import load_index_embedder
    from biblio.index.indexer import index_library
    from biblio.index.watch import Watcher, WatchEvent, choose_device, snapshot
    from biblio.ingest.tokens import load_token_counter
    from biblio.offline import loopback_port_open

    def reindex() -> tuple[DeviceChoice | None, IndexStats | None]:
        cfg, where = now()
        running = loopback_port_open(cfg.llm.base_url, cfg.llm.probe_timeout_s)
        choice = choose_device(preferred, running)
        if choice is None:
            return None, None
        stats = index_library(
            cfg,
            where,
            full=False,
            dtype=choice.dtype,
            embedder_factory=lambda: load_index_embedder(config, choice),
            counter_factory=lambda: load_token_counter(config),
        )
        return choice, stats

    def report(event: WatchEvent) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        seen = f"{event.change} → " if event.change else "catch-up → "
        if event.kind == "locked":
            console.print(f"[yellow]{stamp} {seen}{event.detail} Retrying later.[/yellow]")
        elif event.kind == "deferred":
            console.print(
                f"[yellow]{stamp} {seen}deferred: the LLM server holds the GPU and a float16 "
                "index cannot be updated on CPU.[/yellow]"
            )
        elif event.stats is not None and event.device is not None:
            s = event.stats
            console.print(
                f"{stamp} {seen}{s.added} added, {s.updated} updated, {s.deleted} deleted "
                f"({s.chunks_written} chunks on {event.device.device}, {s.seconds:.1f}s)"
                + (f" [red]failed: {', '.join(s.failed)}[/red]" if s.failed else "")
            )

    return Watcher(config.watch, lambda: snapshot(*now()), reindex, report)


@models_app.callback()
def models_main(ctx: typer.Context) -> None:
    if ("models", ctx.invoked_subcommand) not in _NETWORKED and _config(ctx).runtime.offline:
        apply_offline_env()


@models_app.command("pull")
def models_pull(
    ctx: typer.Context,
    force: bool = typer.Option(False, "--force", help="Download again even if already present."),
) -> None:
    """Download the embedding model and the reranker at their pinned revisions. Uses the network."""
    from biblio.models import model_specs, model_status, pull_model, validate_revision

    config = _config(ctx)
    models_dir = config.paths.models_dir
    specs = model_specs(config)
    for spec in specs:
        try:
            validate_revision(spec)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(2) from exc

    allow_model_downloads()
    for spec in specs:
        if not force and model_status(spec, models_dir).available:
            console.print(f"[green]✓[/green] {spec.repo_id} already at {spec.revision[:12]}")
            continue
        console.print(f"Downloading {spec.repo_id} @ {spec.revision[:12]} → {spec.local_dir}")
        entry = pull_model(spec, models_dir)
        console.print(
            f"[green]✓[/green] {spec.repo_id}: {entry.files} files, "
            f"{entry.size_bytes / 1024**3:.2f} GiB, license {entry.license}"
        )
    console.print(f"Manifest: {models_dir / 'manifest.json'}")


@models_app.command("status")
def models_status(ctx: typer.Context) -> None:
    """Show which models are present locally. Offline."""
    from biblio.models import model_specs, model_status

    config = _config(ctx)
    table = Table("role", "model", "available", "detail")
    for spec in model_specs(config):
        status = model_status(spec, config.paths.models_dir)
        mark = "[green]yes[/green]" if status.available else "[red]no[/red]"
        table.add_row(spec.role, spec.repo_id, mark, status.detail)
    console.print(table)


@app.command()
def doctor(
    ctx: typer.Context,
    load_models: bool = typer.Option(
        False, "--load-models", help="Also load each model from disk and run one inference (slow)."
    ),
) -> None:
    """Check hardware, models, SQLite extensions and offline mode."""
    from biblio.doctor import Status, collect_hardware, run_checks

    checks = run_checks(_config(ctx), collect_hardware(), load_models=load_models)
    styles = {Status.OK: "[green]ok[/green]", Status.WARN: "[yellow]warn[/yellow]"}
    table = Table("group", "check", "status", "detail")
    for check in checks:
        table.add_row(
            check.group, check.name, styles.get(check.status, "[red]FAIL[/red]"), check.detail
        )
    console.print(table)

    failures = [c for c in checks if c.status is Status.FAIL]
    warnings = [c for c in checks if c.status is Status.WARN]
    if failures:
        console.print(f"[red]{len(failures)} check(s) failed[/red], {len(warnings)} warning(s).")
        raise typer.Exit(1)
    console.print(f"[green]All checks passed[/green] ({len(warnings)} warning(s)).")


def _excerpt(text: str, width: int) -> str:
    """The chunk's first lines of prose (heading lines skipped), collapsed to one line."""
    lines = [
        ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")
    ]
    flat = " ".join(lines)
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def _open_retriever(config: Config, mode: str) -> Retriever:
    from biblio.models import ModelNotAvailableError
    from biblio.retrieve.pipeline import build_retriever
    from biblio.retrieve.store import IndexNotFoundError, QueryEmbedderMismatchError

    try:
        return build_retriever(config, mode)
    except (
        ValueError,
        IndexNotFoundError,
        QueryEmbedderMismatchError,
        ModelNotAvailableError,
    ) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc


@app.command()
def search(
    ctx: typer.Context,
    query: Annotated[str, typer.Argument(help="What to look for, in natural language.")],
    mode: Annotated[
        str | None,
        typer.Option("--mode", help="dense, bm25, hybrid, dense-rerank or hybrid-rerank."),
    ] = None,
    k: Annotated[
        int | None, typer.Option("-k", help="Number of results (default final_k).")
    ] = None,
) -> None:
    """Find the sections of the library that answer QUERY. No LLM. Offline."""
    import time

    config = _config(ctx)
    retriever = _open_retriever(config, mode or config.retrieval.default_mode)
    try:
        t0 = time.perf_counter()
        hits = retriever.search(query, k or config.retrieval.final_k)
        elapsed = time.perf_counter() - t0
    finally:
        retriever.close()

    if not hits:
        console.print("No results.")
        return
    table = Table("#", "score", "section", "excerpt", show_lines=True)
    for hit in hits:
        where = Text(hit.citation, style="bold")
        where.append(f"\nL{hit.start_line}-{hit.end_line}", style="dim")
        table.add_row(str(hit.rank), f"{hit.score:.3f}", where, _excerpt(hit.text, 220))
    console.print(table)
    console.print(f"[dim]{retriever.name} · {len(hits)} results · {elapsed * 1000:.0f} ms[/dim]")


@app.command("eval")
def eval_command(
    ctx: typer.Context,
    tag: Annotated[
        str | None, typer.Option("--tag", help="Name of this run (default: <mode>-<time>).")
    ] = None,
    mode: Annotated[
        str | None,
        typer.Option("--mode", help="dense, bm25, hybrid, dense-rerank or hybrid-rerank."),
    ] = None,
    questions_file: Annotated[
        Path | None, typer.Option("--questions", help="Default: eval.questions in config.")
    ] = None,
    plan: Annotated[
        bool,
        typer.Option(
            "--plan",
            help="Plan each question as a conversation's first message (ADR-038; needs the LLM).",
        ),
    ] = False,
) -> None:
    """Measure retrieval on eval/questions.jsonl and append the run to eval/results/runs.jsonl."""
    from datetime import datetime

    from biblio.eval.questions import QuestionsError, load_questions
    from biblio.eval.runner import append_run, check_expected, run_eval
    from biblio.retrieve.store import IndexNotFoundError, open_index

    config = _config(ctx)
    mode = mode or config.retrieval.default_mode
    try:
        questions = load_questions(questions_file or config.eval.questions)
        conn = open_index(config)
    except (QuestionsError, IndexNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc

    checked = check_expected(conn, questions)
    for warning in checked.warnings:
        console.print(f"[yellow]warning[/yellow] {warning}")
    if checked.problems:
        console.print("[red]The evaluation set points to sections that are not in the index:[/red]")
        for problem in checked.problems:
            console.print(f"  - {problem}")
        console.print("Fix eval/questions.jsonl first: otherwise the run measures noise.")
        raise typer.Exit(2)

    retriever = _open_retriever(config, mode)
    if plan:
        from biblio.generate.llm import load_llm_client
        from biblio.generate.thread import PlannedRetriever

        retriever = PlannedRetriever(
            retriever,
            load_llm_client(config.llm),
            config.thread,
            config.retrieval.rrf_k,
        )
    try:
        record = run_eval(
            config,
            retriever,
            conn,
            questions,
            tag or f"{mode}-{datetime.now():%Y%m%d-%H%M%S}",
        )
    finally:
        retriever.close()
    path = append_run(config.eval.results_dir, record)
    _print_run(record)
    console.print(f"[dim]Saved to {path}[/dim]")


def _print_run(record: dict[str, object]) -> None:
    metrics = record["metrics"]
    by_type = record["by_type"]
    assert isinstance(metrics, dict) and isinstance(by_type, dict)
    keys = list(metrics["section"])
    table = Table(
        "", *keys, title=f"{record['tag']} · {record['mode']} · {record['n_questions']} questions"
    )
    for gran in ("section", "file"):
        table.add_row(f"[bold]{gran}[/bold]", *(f"{metrics[gran][k]:.3f}" for k in keys))
    for qtype, group in by_type.items():
        row = group["section"]
        table.add_row(
            f"  {qtype} (n={group['n']})", *(f"{row[k]:.3f}" if k in row else "" for k in keys)
        )
    console.print(table)
    latency = record["latency_ms"]
    assert isinstance(latency, dict)
    console.print(f"latency p50 {latency['p50']} ms · p95 {latency['p95']} ms")
    failures = record["failures"]
    assert isinstance(failures, list)
    if failures:
        console.print(f"no expected section in the top results: {', '.join(failures)}")


@app.command()
def compare(
    ctx: typer.Context,
    before: Annotated[str, typer.Argument(help="Tag of the baseline run.")],
    after: Annotated[str, typer.Argument(help="Tag of the run with the change.")],
) -> None:
    """Compare two eval runs by metric, by question type and question by question."""
    from biblio.eval.runner import flips, load_run

    config = _config(ctx)
    try:
        a = load_run(config.eval.results_dir, before)
        b = load_run(config.eval.results_dir, after)
    except KeyError as exc:
        console.print(f"[red]{exc.args[0]}[/red]")
        raise typer.Exit(2) from exc

    k = config.retrieval.final_k
    table = Table("metric", before, after, "Δ", title="Section-level (file-level in dim)")
    for key in a["metrics"]["section"]:
        if key not in b["metrics"]["section"]:
            continue
        va, vb = a["metrics"]["section"][key], b["metrics"]["section"][key]
        fa, fb = a["metrics"]["file"][key], b["metrics"]["file"][key]
        table.add_row(
            key, f"{va:.3f} [dim]{fa:.3f}[/dim]", f"{vb:.3f} [dim]{fb:.3f}[/dim]", _delta(vb - va)
        )
    for qtype in sorted(set(a["by_type"]) | set(b["by_type"])):
        key = f"recall@{k}"
        va = a["by_type"].get(qtype, {}).get("section", {}).get(key)
        vb = b["by_type"].get(qtype, {}).get("section", {}).get(key)
        if va is not None and vb is not None:
            table.add_row(f"  {qtype} {key}", f"{va:.3f}", f"{vb:.3f}", _delta(vb - va))
    console.print(table)

    changed = flips(a, b, k)
    if changed:
        console.print(f"Questions that changed in the top-{k}:")
        for f in changed:
            arrow = (
                "[green]now found[/green]" if f.after and f.after <= k else "[red]now missed[/red]"
            )
            console.print(f"  {f.question_id} ({f.qtype}): rank {f.before} → {f.after}  {arrow}")
    else:
        console.print(f"No question entered or left the top-{k}.")
    n = b["n_questions"]
    console.print(
        f"[dim]With {n} questions, a change of 1-2 questions ({2 / n:.0%}) is noise "
        "(retrieval-eval skill).[/dim]"
    )


def _delta(value: float) -> str:
    if abs(value) < 0.0005:
        return "0"
    color = "green" if value > 0 else "red"
    return f"[{color}]{value:+.3f}[/{color}]"


@app.command()
def candidates(
    ctx: typer.Context,
    tag: Annotated[str, typer.Argument(help="Run whose missed questions to review.")],
    top: Annotated[int, typer.Option("--top", help="Sections to propose per question.")] = 5,
) -> None:
    """Write eval/review/<tag>.md: what the search returned for each missed question, for you to
    judge as alternative answers. Never edits eval/questions.jsonl."""
    from biblio.eval.candidates import collect, missed_questions, to_markdown
    from biblio.eval.questions import QuestionsError, load_questions
    from biblio.eval.runner import load_run

    config = _config(ctx)
    try:
        record = load_run(config.eval.results_dir, tag)
        questions = load_questions(config.eval.questions)
    except (KeyError, QuestionsError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc

    missed = missed_questions(record, questions, config.retrieval.final_k)
    if not missed:
        console.print(f"Run {tag!r} has no missed questions.")
        return
    retriever = _open_retriever(config, record["mode"])
    try:
        reviews = collect(retriever, missed, top)
    finally:
        retriever.close()
    out = config.eval.results_dir.parent / "review" / f"{tag}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_markdown(reviews, tag, retriever.name), encoding="utf-8", newline="\n")
    console.print(f"{len(missed)} missed questions → {out}")


@app.command()
def ask(
    ctx: typer.Context,
    question: Annotated[str, typer.Argument(help="Your question, in natural language.")],
    mode: Annotated[
        str | None, typer.Option("--mode", help="Retriever (default from config).")
    ] = None,
    show_context: Annotated[
        bool, typer.Option("--show-context", help="List every fragment given to the LLM.")
    ] = False,
    map_reduce: Annotated[
        str | None, typer.Option("--map-reduce", help="auto, on or off (default from config).")
    ] = None,
    level: Annotated[
        str | None,
        typer.Option("--level", help="beginner, intermediate or advanced (default from config)."),
    ] = None,
) -> None:
    """Answer QUESTION from your library, citing its sources. Needs the LLM server (ADR-015)."""
    import time

    from biblio.generate.ask import Tutor
    from biblio.generate.llm import LLMError, load_llm_client
    from biblio.ingest.tokens import load_token_counter_safe
    from biblio.offline import loopback_port_open

    config = _config(ctx)
    if not loopback_port_open(config.llm.base_url, config.llm.probe_timeout_s):
        console.print(
            f"[red]No LLM server on {config.llm.base_url}.[/red] "
            "Start it with scripts/serve-llm.ps1."
        )
        raise typer.Exit(2)
    counter = load_token_counter_safe(config, console)
    retriever = _open_retriever(config, mode or config.retrieval.default_mode)
    llm = load_llm_client(config.llm)
    if level is not None and level not in TUTOR_LEVELS:
        console.print(f"[red]--level must be one of {', '.join(TUTOR_LEVELS)}[/red]")
        raise typer.Exit(2)
    tutor = Tutor(config, retriever, llm, counter, map_reduce, level)
    try:
        t0 = time.perf_counter()
        prepared = tutor.prepare(question)
        if prepared.strategy != "direct":
            guides = len(prepared.groups)
            console.print(
                f"[dim]map-reduce over {guides} guides… measured faithfulness is lower than the "
                "direct path (ADR-017): check the answer against its sources.[/dim]"
            )
        t1 = time.perf_counter()
        parts: list[str] = []
        for token in tutor.stream(prepared):
            parts.append(token)
            console.print(Text(token), end="")
        console.print()
        t2 = time.perf_counter()
    except LLMError as exc:
        console.print(f"\n[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    finally:
        retriever.close()
        llm.close()

    answer = tutor.finish(prepared, "".join(parts).strip())
    v = answer.validation
    if v.invalid:
        console.print(
            f"[yellow]Citations to fragments that do not exist: {', '.join(map(str, v.invalid))}"
        )
    if v.placeholders:
        console.print(f"[yellow]{v.placeholders} citations have no number (like [n]).[/yellow]")
    if not v.abstained and not v.grounded:
        console.print("[red]Ungrounded answer: it cites no valid fragment. Do not trust it.[/red]")
    shown = (
        answer.context.fragments
        if show_context
        else [f for f in answer.context.fragments if f.number in v.cited]
    )
    if shown:
        console.print("\n[bold]Sources[/bold]")
        for f in shown:
            mark = "" if f.number in v.cited else " [dim](not cited)[/dim]"
            console.print(
                f"  [{f.number}] {f.citation} [dim]L{f.start_line}-{f.end_line}[/dim]{mark}"
            )
    console.print(
        f"[dim]{answer.retriever} · {answer.strategy} · {len(answer.context.fragments)} fragments, "
        f"{answer.context.tokens} tokens · retrieval {t1 - t0:.1f}s · answer {t2 - t1:.1f}s[/dim]"
    )


@app.command()
def draft(
    ctx: typer.Context,
    topic: Annotated[str, typer.Argument(help="Topic of the new guide.")],
    outline_file: Annotated[
        Path | None,
        typer.Option(
            "--outline",
            exists=True,
            dir_okay=False,
            help="Edited outline: write the draft from it (step 2).",
        ),
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="Output file (default: drafts/<topic>.md).")
    ] = None,
) -> None:
    """Draft a new guide in two steps: propose an outline for you to edit, then write it
    section by section with citations. A skeleton to edit, not a finished guide (ADR-008)."""
    from datetime import date

    from biblio.generate.draft import (
        Drafter,
        SourceRegistry,
        parse_outline,
        render_draft,
        render_outline_file,
        slugify,
    )
    from biblio.generate.llm import LLMError, load_llm_client
    from biblio.ingest.tokens import load_token_counter_safe
    from biblio.offline import loopback_port_open
    from biblio.retrieve.store import open_index

    config = _config(ctx)
    if not loopback_port_open(config.llm.base_url, config.llm.probe_timeout_s):
        console.print(
            f"[red]No LLM server on {config.llm.base_url}.[/red] "
            "Start it with scripts/serve-llm.ps1."
        )
        raise typer.Exit(2)
    slug = slugify(topic)
    drafts = config.paths.drafts_dir
    drafts.mkdir(parents=True, exist_ok=True)
    counter = load_token_counter_safe(config, console)
    retriever = _open_retriever(config, config.retrieval.default_mode)
    llm = load_llm_client(config.llm)
    drafter = Drafter(config, retriever, llm, counter, open_index(config))
    try:
        if outline_file is None:
            console.print(f"[bold]Outline for:[/bold] {topic}\n")
            outline = drafter.propose_outline(topic, lambda t: console.print(Text(t), end=""))
            console.print()
            if not outline.sections:
                console.print("[red]The model did not return any '## ' section; try again.[/red]")
                raise typer.Exit(1)
            path = drafts / f"{slug}.outline.md"
            command = f'uv run biblio draft "{topic}" --outline {path.as_posix()}'
            path.write_text(render_outline_file(outline, command), encoding="utf-8", newline="\n")
            console.print(f"\n{len(outline.sections)} sections → [bold]{path}[/bold]")
            console.print(f"Edit it, then run: {command}")
            return

        outline = parse_outline(outline_file.read_text(encoding="utf-8"), topic)
        if not outline.sections:
            console.print(f"[red]{outline_file} has no '## ' section.[/red]")
            raise typer.Exit(2)
        registry = SourceRegistry()
        results = []
        for i, section in enumerate(outline.sections, start=1):
            console.print(f"[bold]({i}/{len(outline.sections)}) {section.title}[/bold]")
            result = drafter.write_section(outline, section, registry)
            flags = []
            if result.missing_evidence:
                flags.append("[yellow]no evidence[/yellow]")
            if result.uncited:
                flags.append("[red]uncited[/red]")
            if result.invalid_citations:
                flags.append("[red]invalid citations[/red]")
            console.print("   " + (", ".join(flags) if flags else "[green]ok[/green]"))
            results.append(result)
    except LLMError as exc:
        console.print(f"\n[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    finally:
        retriever.close()
        llm.close()

    target = out or drafts / f"{slug}.md"
    target.write_text(
        render_draft(topic, results, registry, date.today()), encoding="utf-8", newline="\n"
    )
    missing = sum(r.missing_evidence for r in results)
    console.print(
        f"\nDraft → [bold]{target}[/bold] · {len(results)} sections "
        f"({missing} without evidence) · {len(registry.fragments)} sources"
    )


ask_eval_app = typer.Typer(
    help="Evaluate generated answers with human review (faithfulness, coverage, abstention).",
    no_args_is_help=True,
)
app.add_typer(ask_eval_app, name="ask-eval")


@ask_eval_app.command("run")
def ask_eval_run(
    ctx: typer.Context,
    tag: Annotated[str, typer.Option("--tag", help="Name of this run.")],
    questions_file: Annotated[
        Path | None, typer.Option("--questions", help="Default: eval.answers in config.")
    ] = None,
    map_reduce: Annotated[
        str | None, typer.Option("--map-reduce", help="auto, on or off (default from config).")
    ] = None,
    level: Annotated[
        str | None,
        typer.Option("--level", help="beginner, intermediate or advanced (default from config)."),
    ] = None,
) -> None:
    """Answer every question and write a review sheet for you to grade."""
    from biblio.eval.questions import QuestionsError, load_questions
    from biblio.eval.runner import _git_state
    from biblio.generate.answer_eval import run_answers, save_run
    from biblio.generate.ask import Tutor
    from biblio.generate.llm import LLMError, load_llm_client
    from biblio.ingest.tokens import load_token_counter_safe
    from biblio.offline import loopback_port_open

    config = _config(ctx)
    if map_reduce not in (None, "auto", "on", "off"):
        console.print("[red]--map-reduce must be auto, on or off[/red]")
        raise typer.Exit(2)
    try:
        questions = load_questions(questions_file or config.eval.answers)
    except QuestionsError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    if not loopback_port_open(config.llm.base_url, config.llm.probe_timeout_s):
        console.print(
            f"[red]No LLM server on {config.llm.base_url}.[/red] Start scripts/serve-llm.ps1."
        )
        raise typer.Exit(2)
    counter = load_token_counter_safe(config, console)
    retriever = _open_retriever(config, config.retrieval.default_mode)
    llm = load_llm_client(config.llm)
    if level is not None and level not in TUTOR_LEVELS:
        console.print(f"[red]--level must be one of {', '.join(TUTOR_LEVELS)}[/red]")
        raise typer.Exit(2)
    tutor = Tutor(config, retriever, llm, counter, map_reduce, level)

    def progress(q: object, answer: object, seconds: float) -> None:
        from biblio.eval.questions import Question
        from biblio.generate.ask import Answer

        assert isinstance(q, Question) and isinstance(answer, Answer)
        v = answer.validation
        flag = (
            "abstained" if v.abstained else ("grounded" if v.grounded else "[red]ungrounded[/red]")
        )
        console.print(f"  {q.id:6s} {answer.strategy:10s} {seconds:5.1f}s  {flag}")

    try:
        records = run_answers(tutor, questions, progress)
    except LLMError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    finally:
        retriever.close()
        llm.close()
    commit, dirty = _git_state(config.root)
    meta = {
        "commit": commit,
        "dirty": dirty,
        "model": config.llm.model,
        "map_reduce": map_reduce or config.llm.map_reduce,
        "retriever": retriever.name,
        "level": tutor.level,
    }
    review_dir = config.eval.results_dir.parent / "review"
    _, sheet = save_run(review_dir, tag, meta, records)
    console.print(f"\nReview sheet → [bold]{sheet}[/bold]")
    console.print(f"Grade it, then run: uv run biblio ask-eval score {tag}")


@ask_eval_app.command("score")
def ask_eval_score(
    ctx: typer.Context,
    tag: Annotated[str, typer.Argument(help="Run to score (its review sheet must be graded).")],
) -> None:
    """Read your grades and append the run's scores to eval/results/answers.jsonl."""
    import json

    from biblio.generate.answer_eval import append_score, parse_grades, review_paths, score

    config = _config(ctx)
    data_path, sheet_path = review_paths(config.eval.results_dir.parent / "review", tag)
    if not data_path.is_file() or not sheet_path.is_file():
        console.print(f"[red]No run {tag!r}: run `biblio ask-eval run --tag {tag}` first.[/red]")
        raise typer.Exit(2)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    grades = parse_grades(sheet_path.read_text(encoding="utf-8"))
    result = score(data["records"], grades)
    meta = {k: v for k, v in data.items() if k not in ("tag", "records")}
    path = append_score(config.eval.results_dir, tag, meta, result)

    m = result["metrics"]
    table = Table("", "value", title=f"{tag} · {m['n']} questions", show_header=False)
    for key in (
        "faithfulness",
        "fully_faithful",
        "coverage",
        "abstained_when_absent",
        "false_abstention",
        "grounded",
        "invalid_citations",
        "placeholder_citations",
        "mean_seconds",
        "map_reduce_share",
    ):
        if key in m:
            table.add_row(key, str(m[key]))
    console.print(table)
    graded = m.get("graded_faithfulness", 0)
    if graded < m["n"]:
        console.print(f"[yellow]{m['n'] - graded} answers have no faithfulness grade yet.[/yellow]")
    console.print(f"[dim]Saved to {path}[/dim]")


@app.command("eval-feynman")
def eval_feynman(
    ctx: typer.Context,
    tag: Annotated[
        str | None, typer.Option("--tag", help="Name of this run (default: feynman-<time>).")
    ] = None,
    items_file: Annotated[
        Path | None, typer.Option("--set", help="Default: eval.feynman in config.")
    ] = None,
) -> None:
    """Measure «Explícamelo tú» on eval/feynman.jsonl: how many expected gaps and unsupported
    claims it reports, and how many it reports beyond them (ADR-027). Needs the LLM server."""
    from datetime import datetime

    from biblio.eval.feynman import (
        FeynmanSetError,
        append_feynman_run,
        load_feynman_set,
        run_feynman_eval,
    )
    from biblio.generate.llm import LLMError, load_llm_client
    from biblio.index.embedder import load_query_embedder
    from biblio.models import ModelNotAvailableError
    from biblio.offline import loopback_port_open
    from biblio.retrieve.pipeline import build_retriever
    from biblio.retrieve.store import IndexNotFoundError, QueryEmbedderMismatchError

    config = _config(ctx)
    try:
        items = load_feynman_set(items_file or config.eval.feynman)
    except FeynmanSetError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    if not loopback_port_open(config.llm.base_url, config.llm.probe_timeout_s):
        console.print(
            f"[red]No LLM server on {config.llm.base_url}.[/red] "
            "Start it with scripts/serve-llm.ps1."
        )
        raise typer.Exit(2)
    try:
        embedder = load_query_embedder(config)
        retriever = build_retriever(
            config, config.retrieval.default_mode, embedder_factory=lambda _c: embedder
        )
    except (IndexNotFoundError, QueryEmbedderMismatchError, ModelNotAvailableError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    llm = load_llm_client(config.llm)

    def show(item: object, row: dict[str, object]) -> None:
        console.print(
            f"[dim]{row['id']}: gaps {row['found_missing']}/{row['expected_missing']} · "
            f"claims {row['found_unsupported']}/{row['expected_unsupported']} · "
            f"{row['seconds']}s[/dim]"
        )

    try:
        record = run_feynman_eval(
            config,
            retriever,
            llm,
            embedder.embed,
            items,
            tag or f"feynman-{datetime.now():%Y%m%d-%H%M%S}",
            show,
        )
    except LLMError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    finally:
        retriever.close()
        llm.close()
    path = append_feynman_run(config.eval.results_dir, record)
    table = Table("metric", "value", title=f"{record['tag']} · {record['n_items']} explanations")
    for key, value in record["metrics"].items():
        table.add_row(key, "—" if value is None else str(value))
    latency = record["latency_s"]
    table.add_row("latency p50 / p95", f"{latency['p50']} s / {latency['p95']} s")
    console.print(table)
    for row in record["items"]:
        for point in row["not_found"]:
            console.print(f"[yellow]not reported[/yellow] {row['id']}: {point}")
    console.print(f"[dim]Saved to {path}[/dim]")


@app.command("map")
def concept_map(
    ctx: typer.Context,
    topic: Annotated[str, typer.Argument(help="Topic of the concept map.")],
) -> None:
    """Build a concept map of TOPIC from your library and print it as an outline (ADR-022).
    Needs the LLM server; the page Mapas draws the same map."""
    import time

    from biblio.generate.llm import LLMError, load_llm_client
    from biblio.index.embedder import load_query_embedder
    from biblio.maps.build import CachedEmbed, generate_map, map_outline
    from biblio.maps.export import map_fragments
    from biblio.models import ModelNotAvailableError
    from biblio.offline import loopback_port_open
    from biblio.retrieve.pipeline import build_retriever
    from biblio.retrieve.store import IndexNotFoundError, QueryEmbedderMismatchError

    config = _config(ctx)
    if not loopback_port_open(config.llm.base_url, config.llm.probe_timeout_s):
        console.print(
            f"[red]No LLM server on {config.llm.base_url}.[/red] "
            "Start it with scripts/serve-llm.ps1."
        )
        raise typer.Exit(2)
    try:
        embedder = load_query_embedder(config)
        retriever = build_retriever(
            config, config.retrieval.default_mode, embedder_factory=lambda _c: embedder
        )
    except (IndexNotFoundError, QueryEmbedderMismatchError, ModelNotAvailableError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    llm = load_llm_client(config.llm)
    t0 = time.perf_counter()
    try:
        fragments = map_fragments(retriever.search(topic, config.maps.fragments * 2), config.maps)
        if not fragments:
            console.print("[yellow]No sections about that topic in the library.[/yellow]")
            raise typer.Exit(1)
        progress = None
        for progress in generate_map(
            llm, config.maps, topic, fragments, CachedEmbed(embedder.embed)
        ):
            done = fragments[progress.processed - 1]
            console.print(f"[dim]({progress.processed}/{len(fragments)}) {done.citation}[/dim]")
    except LLMError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    finally:
        retriever.close()
        llm.close()
    assert progress is not None
    console.print(f"\n[bold]{topic}[/bold]")
    for line in map_outline(progress.map):
        console.print(Text(line))
    console.print(
        f"\n{len(progress.map.nodes) - 1} concepts · {progress.map.omitted} left out "
        "(weak, unrelated or over the caps) · "
        f"{progress.skipped} fragments skipped · {time.perf_counter() - t0:.1f}s"
    )
    for f in fragments:
        console.print(f"[dim]{f.number}. {f.citation} (líneas {f.start_line}-{f.end_line})[/dim]")


@app.command()
def serve(
    ctx: typer.Context,
    port: Annotated[int | None, typer.Option("--port", help="Default: web.port in config.")] = None,
    lan: Annotated[
        bool | None,
        typer.Option("--lan/--no-lan", help="Also serve paired devices on the home network."),
    ] = None,
) -> None:
    """Open the local web UI (loopback only, ADR-016, ADR-019; --lan: ADR-035). Ctrl+C stops it."""
    config = _config(ctx)
    server, engine = _open_web(config, port or config.web.port, _lan_wanted(config, lan))
    host, bound = config.web.host, server.server_address[1]
    console.print(f"[bold]Biblioteca[/bold] → http://{host}:{bound}  [dim](Ctrl+C to stop)[/dim]")
    _print_lan(server)
    _serve_until_stopped(server, engine)


def _lan_wanted(config: Config, flag: bool | None) -> bool:
    return config.web.lan if flag is None else flag


def _print_lan(server: ThreadingHTTPServer) -> None:
    """LAN mode (ADR-035): where to point a phone, and a first pairing code to scan."""
    from biblio.web.lan import PAIR_PATH, qr_text

    lan = getattr(server, "biblio_lan", None)
    ip = getattr(server, "biblio_lan_ip", "")
    if lan is None:
        return
    port = server.server_address[1]
    if not ip:
        console.print(
            "[yellow]LAN mode is on but this PC has no address on a private network.[/yellow]"
        )
        return
    code, expires = lan.new_code()
    link = f"http://{ip}:{port}{PAIR_PATH}?code={code}"
    console.print(
        f"[bold]Red local[/bold] → http://{ip}:{port}  "
        f"[dim](scan within {lan_minutes(expires)} min to pair a phone or tablet; "
        "more codes: «Conectar un dispositivo» in the app)[/dim]"
    )
    console.print(Text(qr_text(link)))
    console.print(f"[dim]{link}[/dim]")


def lan_minutes(expires: object) -> int:
    from datetime import datetime

    assert isinstance(expires, datetime)
    return max(1, round((expires - datetime.now().astimezone()).total_seconds() / 60))


def _open_web(config: Config, port: int, lan: bool = False) -> tuple[ThreadingHTTPServer, Engine]:
    from biblio.index.embedder import load_query_embedder
    from biblio.retrieve.store import (
        IndexNotFoundError,
        QueryEmbedderMismatchError,
        check_query_embedder,
        open_index,
    )
    from biblio.web.engine import Engine
    from biblio.web.server import LoopbackOnlyError, make_server

    try:
        conn = open_index(config, shared_across_threads=True)
        check_query_embedder(conn, config)
        console.print("Loading the query embedder…")
        engine = Engine(config, conn, load_query_embedder(config))
        access, lan_ip = None, ""
        if lan:  # ADR-035
            from biblio.web.lan import LanAccess, lan_address

            web = config.web
            access = LanAccess(web.devices, web.pairing_minutes, web.device_days)
            lan_ip = lan_address() or ""
        server = make_server(engine, config.web.host, port, access, lan_ip)
        server.biblio_lan = access  # type: ignore[attr-defined]
        server.biblio_lan_ip = lan_ip  # type: ignore[attr-defined]
    except (IndexNotFoundError, QueryEmbedderMismatchError, LoopbackOnlyError, OSError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    return server, engine


def _serve_until_stopped(server: ThreadingHTTPServer, engine: Engine) -> None:
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("Stopping…")
    finally:
        server.server_close()
        engine.close()


# --- start: everything in one command (ADR-021) ---------------------------------------------------


def _confirm(question: str, yes: bool) -> bool:
    """Ask before any download; `--yes` agrees, and a non-interactive run declines."""
    if yes:
        return True
    if not sys.stdin.isatty():
        console.print(f"[yellow]{question} Not asked (no terminal): run with --yes to agree.")
        return False
    return typer.confirm(question, default=True)


def _start_models(config: Config, yes: bool) -> None:
    from biblio.models import model_specs, model_status, pull_model

    models_dir = config.paths.models_dir
    missing = [s for s in model_specs(config) if not model_status(s, models_dir).available]
    if not missing:
        console.print("[green]✓[/green] Models present")
        return
    names = ", ".join(s.repo_id for s in missing)
    if not _confirm(f"Download the missing models ({names}, ~4.3 GB in total)?", yes):
        console.print("[red]The models are required.[/red] Run `biblio models pull` when ready.")
        raise typer.Exit(2)
    allow_model_downloads()  # the only networked step of biblio itself (rule 1)
    try:
        for spec in missing:
            console.print(f"Downloading {spec.repo_id} @ {spec.revision[:12]}…")
            pull_model(spec, config.paths.models_dir)
    finally:
        apply_offline_env()
    console.print("[green]✓[/green] Models downloaded")


def _start_library(config: Config) -> Config:
    """Ask for the library once and save it in config/local.toml; returns the reloaded config."""
    from biblio.launcher import looks_like_library, set_local_library

    library = config.paths.library
    if library is not None and looks_like_library(library):
        console.print(f"[green]✓[/green] Library: {library}")
        return config
    if library is not None:
        console.print(f"[yellow]The library {library} has no Markdown notes or does not exist.")
    if not sys.stdin.isatty():
        console.print("[red]No library: set paths.library in config/local.toml.[/red]")
        raise typer.Exit(2)
    while True:
        answer = typer.prompt("Folder of your Markdown library (e.g. your Obsidian vault)")
        path = Path(answer.strip().strip('"')).expanduser()
        if looks_like_library(path):
            break
        console.print(f"[red]{path} is not a folder with .md notes.[/red]")
    local = config.root / "config" / "local.toml"
    set_local_library(local, path)
    console.print(f"[green]✓[/green] Library saved in {local}")
    return load_config(config.root)


def _start_index(config: Config, root: Path, yes: bool) -> None:
    """Bring the index up to date before the LLM starts, so it can use the GPU (ADR-012)."""
    from biblio.index.embedder import resolve_index_device
    from biblio.index.indexer import IndexLockedError, IndexMismatchError
    from biblio.index.watch import choose_device
    from biblio.ingest.discovery import discover
    from biblio.offline import loopback_port_open

    llm_running = loopback_port_open(config.llm.base_url, config.llm.probe_timeout_s)
    choice = choose_device(resolve_index_device(config), llm_running)
    if choice is None:
        console.print("[yellow]Index not updated: the LLM server holds the GPU (float16 index).")
        return
    first = not config.paths.index_db.is_file()
    if first:
        notes = len(discover(root, config.paths.include, config.paths.exclude))
        where = "~1 minute per 60 notes on the GPU" if choice.device != "cpu" else "hours on CPU"
        if not _confirm(f"First index of {notes} notes ({where}). Start it now?", yes):
            console.print("[red]The app needs an index.[/red] Run `biblio start` again when ready.")
            raise typer.Exit(2)
    try:
        _run_index(config, root, first, choice)
    except IndexLockedError as exc:
        console.print(f"[yellow]{exc} Using the index as it is.[/yellow]")
    except IndexMismatchError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        if not _confirm("The index was built with other settings. Rebuild it now?", yes):
            raise typer.Exit(2) from exc
        _run_index(config, root, True, choice)


def _start_llm(config: Config, yes: bool) -> LLMServer | None:
    """Make sure the tutor's LLM server answers with the pinned model (ADR-015); returns the
    server if this command started it (to stop it on exit)."""
    import os
    import subprocess
    import time

    from biblio.launcher import (
        SERVE_LLM_SCRIPT,
        LLMServer,
        find_ollama,
        llm_server_env,
        served_models,
        wait_for_port,
    )
    from biblio.offline import loopback_port_open

    cfg = config.llm
    exe = find_ollama(config.launcher.ollama_path)
    env = llm_server_env(config.root / SERVE_LLM_SCRIPT)
    started = None
    if not loopback_port_open(cfg.base_url, cfg.probe_timeout_s):
        if exe is None:
            console.print(
                "[yellow]Ollama is not installed:[/yellow] «Explicar», Guías and Practicar "
                "need it. Install it from https://ollama.com and run `biblio start` again. "
                "Continuing with search only."
            )
            return None
        log = config.paths.index_db.parent / "ollama.log"
        started = LLMServer.start(exe, env, log)
        alive = lambda: started.process.poll() is None  # noqa: E731
        if not wait_for_port(
            cfg.base_url, config.launcher.llm_start_timeout_s, cfg.probe_timeout_s, alive
        ):
            started.stop()
            console.print(f"[red]The LLM server did not start; see {log}.[/red] Search only.")
            return None
    # a freshly started Ollama opens its port a moment before its API answers
    deadline = time.monotonic() + config.launcher.llm_start_timeout_s
    models = served_models(cfg.base_url)
    while models is None and started is not None and time.monotonic() < deadline:
        time.sleep(0.5)
        models = served_models(cfg.base_url)
    if models is None:
        console.print("[yellow]The LLM server answers but not as Ollama; using it as it is.")
        return started
    if cfg.model not in models:
        if exe is None or not _confirm(f"Download the LLM {cfg.model} (~2.5 GB)?", yes):
            console.print(f"[yellow]Without {cfg.model} only search works.[/yellow]")
            return started
        subprocess.run([str(exe), "pull", cfg.model], env={**os.environ, **env}, check=False)
        models = served_models(cfg.base_url) or {}
    if cfg.model not in models:
        console.print(f"[yellow]{cfg.model} is still missing; only search works.[/yellow]")
    elif models[cfg.model] != cfg.model_digest:
        console.print(
            f"[yellow]{cfg.model} has digest {models[cfg.model][:12]}, not the pinned "
            f"{cfg.model_digest[:12]}; answers may differ from the evaluation (ADR-015).[/yellow]"
        )
    else:
        how = "started" if started else "already running"
        console.print(f"[green]✓[/green] LLM {cfg.model} ({how})")
    return started


@app.command()
def start(
    ctx: typer.Context,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Agree to every download without asking.")
    ] = False,
    port: Annotated[int | None, typer.Option("--port", help="Default: web.port in config.")] = None,
    no_llm: Annotated[bool, typer.Option("--no-llm", help="Do not start the LLM server.")] = False,
    no_browser: Annotated[
        bool, typer.Option("--no-browser", help="Do not open the web UI in the browser.")
    ] = False,
    lan: Annotated[
        bool | None,
        typer.Option("--lan/--no-lan", help="Also serve paired devices on the home network."),
    ] = None,
) -> None:
    """Prepare everything and open the app: models, library, index, LLM server, web UI and
    automatic reindexing (ADR-021). Downloads only what is missing, after asking."""
    import threading
    import webbrowser

    from biblio.index.embedder import resolve_index_device
    from biblio.offline import loopback_port_open

    config = _config(ctx)
    port = port or config.web.port
    url = f"http://{config.web.host}:{port}"
    if loopback_port_open(url, config.llm.probe_timeout_s):
        console.print(f"Biblioteca is already running → {url}")
        if config.launcher.open_browser and not no_browser:
            webbrowser.open(url)
        return

    _start_models(config, yes)
    config = _start_library(config)
    root = config.paths.library
    assert root is not None
    _start_index(config, root, yes)  # before the LLM: the GPU is free now (ADR-012)
    llm = None if no_llm else _start_llm(config, yes)
    try:
        server, engine = _open_web(config, port, _lan_wanted(config, lan))
        if config.launcher.watch:
            watcher = _make_watcher(
                config, root, resolve_index_device(config), lambda: engine.config
            )

            def watch_forever() -> None:
                try:
                    watcher.run_forever()
                except Exception as exc:  # the app keeps working without automatic reindexing
                    console.print(f"[red]Automatic reindexing stopped: {exc}[/red]")

            threading.Thread(target=watch_forever, daemon=True).start()
        console.print(
            f"\n[bold]Biblioteca[/bold] → {url}  [dim](keep this window open; Ctrl+C to stop)[/dim]"
        )
        _print_lan(server)
        if config.launcher.open_browser and not no_browser:
            webbrowser.open(url)
        _serve_until_stopped(server, engine)
    finally:
        if llm is not None:
            console.print("Stopping the LLM server…")
            llm.stop()


@ask_eval_app.command("sheet")
def ask_eval_sheet(
    ctx: typer.Context,
    tag: Annotated[str, typer.Argument(help="Run whose review sheet to regenerate.")],
) -> None:
    """Regenerate a run's review sheet from its saved answers, keeping the grades already filled."""
    import json

    from biblio.generate.answer_eval import carry_over_grades, review_markdown, review_paths

    config = _config(ctx)
    data_path, sheet_path = review_paths(config.eval.results_dir.parent / "review", tag)
    if not data_path.is_file():
        console.print(f"[red]No run {tag!r}.[/red]")
        raise typer.Exit(2)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    sheet = review_markdown(data["records"], tag)
    if sheet_path.is_file():
        sheet = carry_over_grades(sheet, sheet_path.read_text(encoding="utf-8"))
    sheet_path.write_text(sheet, encoding="utf-8", newline="\n")
    console.print(f"Review sheet regenerated → {sheet_path}")


# --- where the library is (ADR-039) ------------------------------------------------------------


@library_app.callback(invoke_without_command=True)
def library_main(ctx: typer.Context) -> None:
    """Show where the library is; `biblio library set <ruta>` moves it."""
    if ctx.invoked_subcommand is not None:
        return
    config = _config(ctx)
    library = config.paths.library
    console.print(f"Library: {library if library is not None else '(not set)'}")


@library_app.command("set")
def library_set(
    ctx: typer.Context,
    path: Annotated[Path, typer.Argument(help="The library folder (e.g. an Obsidian vault).")],
    yes: Annotated[bool, typer.Option("--yes", help="Do not ask before a full index.")] = False,
) -> None:
    """Move the library to another folder: the index is reused when most notes match (the same
    library moved), otherwise a full index is built and replaces the current one if it completes."""
    from biblio.index.embedder import resolve_index_device
    from biblio.index.indexer import IndexLockedError, IndexMismatchError
    from biblio.index.watch import choose_device
    from biblio.library import INVALID, RELOCATE, SAME, check_library, save_library
    from biblio.offline import loopback_port_open
    from biblio.retrieve.store import open_index

    config = _config(ctx)
    conn = open_index(config) if config.paths.index_db.is_file() else None
    try:
        found = check_library(config, path, conn)
    finally:
        if conn is not None:
            conn.close()
    if found.plan == INVALID:
        console.print(f"[red]{found.path}: {found.reason}[/red]")
        raise typer.Exit(2)
    if found.plan == SAME:
        console.print(f"[green]✓[/green] The library is already {found.path}.")
        return
    llm_running = loopback_port_open(config.llm.base_url, config.llm.probe_timeout_s)
    choice = choose_device(resolve_index_device(config), llm_running)
    if choice is None:
        console.print("[red]The index is float16 and the LLM server holds the GPU: stop it first.")
        raise typer.Exit(2)
    if found.plan == RELOCATE:
        console.print(
            f"The same library moved: {found.matching} of {found.notes} notes match the index; "
            "it is reused."
        )
    else:
        minutes = found.minutes_cpu if choice.device == "cpu" else found.minutes_gpu
        console.print(
            f"Another library: {found.notes} notes, {found.matching} in the index now. A full "
            f"index on {choice.device} takes about {minutes} min; the current one stays until "
            "it completes. Practice history, maps and plans refer to notes of the old library."
        )
        if not _confirm("Build it now?", yes):
            raise typer.Exit(1)
    try:
        _run_index(config, Path(found.path), found.plan != RELOCATE, choice, relocate=True)
    except (IndexLockedError, IndexMismatchError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    save_library(config, Path(found.path))
    console.print(f"[green]✓[/green] Library: {found.path} (saved in config/local.toml)")


@app.command("reset")
def reset_command(
    ctx: typer.Context,
    yes: Annotated[bool, typer.Option("--yes", help="Do not ask «¿Estás seguro?».")] = False,
) -> None:
    """Start the app over without its memory (ADR-040): conversations, maps and diagrams,
    practice, plans, dictionaries and drafts are moved to data/backups/. The library, its index,
    the models and the settings are not touched. Close the app first."""
    from biblio.offline import loopback_port_open
    from biblio.reset import memory_items, reset_memory

    config = _config(ctx)
    url = f"http://{config.web.host}:{config.web.port}"
    if loopback_port_open(url, config.llm.probe_timeout_s):
        console.print(f"[red]The app is running at {url}: use «Reset» in Biblioteca, or close it.")
        raise typer.Exit(2)
    labels = {
        "conversations": "conversations",
        "maps": "maps and diagrams",
        "practice": "practice questions (with reviews, plans, dictionaries…)",
        "drafts": "files in drafts/",
    }
    for item in memory_items(config):
        console.print(f"  {item.count:>5} {labels[item.key]}  [dim]{item.path}[/dim]")
    console.print("The library, its index, the models and config/local.toml stay as they are.")
    if not _confirm("¿Estás seguro? Erase the app's memory (a copy goes to data/backups/)", yes):
        raise typer.Exit(1)
    try:
        result = reset_memory(config)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    console.print(f"[green]✓[/green] Memory moved to {result.backup} ({len(result.moved)} items).")
