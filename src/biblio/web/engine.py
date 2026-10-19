"""Warm models shared by the web UI's requests (ADR-016, ADR-019).

The query embedder stays loaded on CPU. The reranker follows ADR-014 *per request*: it is loaded
on the GPU only while no LLM server listens, and released before every LLM call, so it never shares
the GPU with the LLM. One request at a time uses the models (single user, `lock`).
"""

from __future__ import annotations

import json
import queue
import random
import sqlite3
import threading
import time
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import quote

import numpy as np

from biblio.config import Config
from biblio.conversations import ConversationStore, NewTurn
from biblio.generate.ask import Tutor
from biblio.generate.citations import extract_citations
from biblio.generate.context import Fragment
from biblio.generate.draft import (
    Drafter,
    Outline,
    OutlineSection,
    SectionResult,
    SourceRegistry,
    render_draft,
    render_outline_file,
    slugify,
)
from biblio.generate.llm import LLMClient, load_llm_client
from biblio.index.embedder import (
    DeviceChoice,
    Embedder,
    load_index_embedder,
    resolve_index_device,
)
from biblio.index.indexer import IndexLockedError, IndexMismatchError, IndexStats, index_library
from biblio.index.watch import choose_device
from biblio.ingest.tokens import TokenCounter, load_llm_token_counter, load_token_counter
from biblio.maps.build import CachedEmbed, ConceptMap, generate_map
from biblio.maps.export import map_fragments, to_canvas, to_markdown
from biblio.maps.history import MapStore, concept_map_from_page, fragment_from_page
from biblio.offline import loopback_port_open
from biblio.practice.quiz import (
    assign_kinds,
    candidate_sections,
    choose_sections,
    generate_question,
    grade_answer,
)
from biblio.practice.store import PracticeStore, StoredQuestion
from biblio.retrieve.bm25 import BM25Retriever
from biblio.retrieve.dense import DenseRetriever
from biblio.retrieve.fusion import HybridRetriever
from biblio.retrieve.rerank import (
    Reranker,
    RerankingRetriever,
    RerankPlan,
    load_reranker,
    resolve_rerank_plan,
)
from biblio.retrieve.store import library_folders
from biblio.retrieve.types import Hit, Retriever
from biblio.web.drafts import draft_path, outline_path

_EXCERPT_CHARS = 320
T = TypeVar("T")
Event = tuple[str, dict[str, Any]]


@dataclass(frozen=True)
class SearchResult:
    retriever: str
    milliseconds: float
    hits: tuple[Hit, ...]


class BusyError(RuntimeError):
    """Another draft or quiz is being generated; the LLM serves one at a time."""


class _Cancelled(Exception):
    """The page that asked for a stream went away."""


class _LockedRetriever:
    """Takes the engine lock only while searching, so a draft that runs for minutes does not block
    the other pages between its searches."""

    def __init__(self, base: Retriever, lock: threading.Lock) -> None:
        self._base = base
        self._lock = lock
        self.name = base.name

    def search(self, query: str, k: int) -> list[Hit]:
        with self._lock:
            return self._base.search(query, k)

    def close(self) -> None:
        pass  # the shared embedder belongs to the engine


def _threaded(work: Callable[[Callable[[Event], None]], T]) -> Generator[Event, None, T]:
    """Run `work(emit)` in a thread, yield every event it emits, and return its result.

    If the consumer stops (the browser closed the stream), the next `emit` raises inside `work`,
    so the work is abandoned instead of running on unseen.
    """
    events: queue.Queue[Event | None] = queue.Queue()  # None marks the end
    outcome: dict[str, Any] = {}
    cancelled = threading.Event()

    def emit(event: Event) -> None:
        if cancelled.is_set():
            raise _Cancelled
        events.put(event)

    def run() -> None:
        try:
            outcome["value"] = work(emit)
        except BaseException as exc:  # handed to the consumer below
            outcome["error"] = exc
        finally:
            events.put(None)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        while (event := events.get()) is not None:
            yield event
    finally:
        cancelled.set()
    worker.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]  # type: ignore[no-any-return]


def _streamed(
    work: Callable[[Callable[[str], None]], T], event: Callable[[str], Event]
) -> Generator[Event, None, T]:
    """Run `work(on_token)` in a thread and yield each token as an event; return its result."""
    return (yield from _threaded(lambda emit: work(lambda text: emit(event(text)))))


class Engine:
    def __init__(
        self,
        config: Config,
        conn: sqlite3.Connection,
        embedder: Embedder,
        reranker_factory: Callable[[Config, str], Reranker] = load_reranker,
        plan_resolver: Callable[[Config], RerankPlan] = resolve_rerank_plan,
        llm_factory: Callable[[Config], LLMClient] = lambda c: load_llm_client(c.llm),
        counter_factory: Callable[[Config], TokenCounter] = load_llm_token_counter,
        index_embedder_factory: Callable[[Config, DeviceChoice], Embedder] = load_index_embedder,
        index_counter_factory: Callable[[Config], TokenCounter] = load_token_counter,
        index_device: Callable[[Config], DeviceChoice] = resolve_index_device,
    ) -> None:
        self.config = config
        self.lock = threading.Lock()
        self._conn = conn
        self._embedder = embedder
        self._reranker_factory = reranker_factory
        self._plan_resolver = plan_resolver
        self._llm_factory = llm_factory
        self._counter_factory = counter_factory
        self._index_embedder_factory = index_embedder_factory
        self._index_counter_factory = index_counter_factory
        self._index_device = index_device
        self._reranker: Reranker | None = None
        self._reranker_device: str | None = None
        self._counter: TokenCounter | None = None
        self._generating = threading.Lock()  # one draft or quiz generation at a time
        self._llm_lock = threading.Lock()  # quiz generation and answer grading take turns
        self._practice: PracticeStore | None = None
        self._map_history: MapStore | None = None
        self._conversations: ConversationStore | None = None

    # --- retrieval --------------------------------------------------------------------------------

    def _base(self, scope: str | None = None) -> Retriever:
        """The configured first stage, optionally limited to one folder (ADR-019)."""
        dense = DenseRetriever(self._conn, self._embedder, scope)
        bm25 = BM25Retriever(self._conn, scope)
        base_mode = self.config.retrieval.default_mode.partition("-")[0]
        if base_mode == "dense":
            return dense
        if base_mode == "bm25":
            return bm25
        return HybridRetriever(dense, bm25, self.config.retrieval)

    def _release_reranker(self) -> None:
        if self._reranker is not None:
            self._reranker.close()
            self._reranker = None
            self._reranker_device = None

    def _search_retriever(self, precise: bool, scope: str | None) -> Retriever:
        base = self._base(scope)
        if not (precise and self.config.retrieval.default_mode.endswith("-rerank")):
            return base
        plan = self._plan_resolver(self.config)
        if plan.device is None:
            self._release_reranker()  # the GPU belongs to the LLM now (ADR-014)
            return base
        if self._reranker is None or self._reranker_device != plan.device:
            self._release_reranker()
            self._reranker = self._reranker_factory(self.config, plan.device)
            self._reranker_device = plan.device
        return RerankingRetriever(base, self._reranker, plan.candidates, plan.device)

    def search(
        self, query: str, k: int, *, precise: bool = True, scope: str | None = None
    ) -> SearchResult:
        """`precise=False` skips the reranker even when the GPU is free (the "Rápida" setting)."""
        with self.lock:
            retriever = self._search_retriever(precise, scope)
            t0 = time.perf_counter()
            hits = retriever.search(query, k)
            return SearchResult(retriever.name, (time.perf_counter() - t0) * 1000, tuple(hits))

    def folders(self) -> list[tuple[str, int]]:
        with self.lock:
            return library_folders(self._conn)

    # --- ask --------------------------------------------------------------------------------------

    def llm_running(self) -> bool:
        return loopback_port_open(self.config.llm.base_url, self.config.llm.probe_timeout_s)

    def ask(
        self, question: str, *, scope: str | None = None, map_reduce: str = "off"
    ) -> Generator[Event, None, None]:
        """UI events: ("meta", …) once, ("token", …) while generating, ("done", …) at the end."""
        with self.lock:
            self._release_reranker()  # before the LLM loads its model on the GPU (ADR-014)
            if self._counter is None:
                self._counter = self._counter_factory(self.config)
            llm = self._llm_factory(self.config)
            try:
                tutor = Tutor(self.config, self._base(scope), llm, self._counter, map_reduce)
                t0 = time.perf_counter()
                prepared = tutor.prepare(question)
                yield (
                    "meta",
                    {
                        "retriever": prepared.retriever,
                        "strategy": prepared.strategy,
                        "fragments": [fragment_json(f, self) for f in prepared.context.fragments],
                    },
                )
                parts = []
                for token in tutor.stream(prepared):
                    parts.append(token)
                    yield "token", {"text": token}
                answer = tutor.finish(prepared, "".join(parts).strip())
                v = answer.validation
                yield (
                    "done",
                    {
                        "cited": list(v.cited),
                        "invalid": list(v.invalid),
                        "placeholders": v.placeholders,
                        "abstained": v.abstained,
                        "grounded": v.grounded,
                        "seconds": round(time.perf_counter() - t0, 1),
                    },
                )
            finally:
                llm.close()

    # --- drafts (Guías page, ADR-019) ------------------------------------------------------------

    def _llm_turn(self) -> TokenCounter:
        """Free the GPU for the LLM and return the token counter (call with `lock` held)."""
        self._release_reranker()  # ADR-014
        if self._counter is None:
            self._counter = self._counter_factory(self.config)
        return self._counter

    def _claim_generation(self) -> None:
        if not self._generating.acquire(blocking=False):
            raise BusyError(
                "Ya se está generando una guía o un cuestionario; espera a que termine."
            )

    def propose_outline(self, topic: str) -> Generator[Event, None, None]:
        """Events: ("token", …) while the model writes, then ("outline", …). Saves the outline file
        next to the CLI's, so `biblio draft --outline` can continue it."""
        self._claim_generation()
        try:
            with self.lock:  # sibling outlines read the index connection directly
                counter = self._llm_turn()
                llm = self._llm_factory(self.config)
                try:
                    drafter = Drafter(self.config, self._base(), llm, counter, self._conn)
                    outline = yield from _streamed(
                        lambda on_token: drafter.propose_outline(topic, on_token),
                        lambda text: ("token", {"text": text}),
                    )
                finally:
                    llm.close()
            slug = self._save_outline(outline)
            yield "outline", {**outline_json(outline), "slug": slug}
        finally:
            self._generating.release()

    def _save_outline(self, outline: Outline) -> str:
        slug = slugify(outline.topic)
        drafts = self.config.paths.drafts_dir
        drafts.mkdir(parents=True, exist_ok=True)
        path = outline_path(drafts, slug)
        command = f'uv run biblio draft "{outline.topic}" --outline {path.as_posix()}'
        path.write_text(render_outline_file(outline, command), encoding="utf-8", newline="\n")
        return slug

    def write_draft(
        self, topic: str, sections: Sequence[OutlineSection]
    ) -> Generator[Event, None, None]:
        """Events: ("start", …), then per section ("section", …), ("token", …)* and
        ("section_done", …), and finally ("done", …) with the saved Markdown and its sources."""
        self._claim_generation()
        try:
            outline = Outline(topic, tuple(sections))
            slug = self._save_outline(outline)  # the outline as edited in the page
            with self.lock:
                counter = self._llm_turn()
            llm = self._llm_factory(self.config)
            try:
                drafter = Drafter(
                    self.config, _LockedRetriever(self._base(), self.lock), llm, counter, self._conn
                )
                registry = SourceRegistry()
                results = []
                yield "start", {"slug": slug, "sections": len(outline.sections)}
                for index, section in enumerate(outline.sections):
                    yield "section", {"index": index, "title": section.title}

                    def work(
                        on_token: Callable[[str], None], section: OutlineSection = section
                    ) -> SectionResult:
                        return drafter.write_section(outline, section, registry, on_token)

                    def token(text: str, index: int = index) -> Event:
                        return "token", {"index": index, "text": text}

                    result = yield from _streamed(work, token)
                    results.append(result)
                    yield (
                        "section_done",
                        {
                            "index": index,
                            "title": result.title,
                            "body": result.body,
                            "missing_evidence": result.missing_evidence,
                            "uncited": result.uncited,
                            "invalid_citations": list(result.invalid_citations),
                            "cited": sorted(extract_citations(result.body)),
                        },
                    )
            finally:
                llm.close()
            markdown = render_draft(topic, results, registry, date.today())
            target = draft_path(self.config.paths.drafts_dir, slug)
            target.write_text(markdown, encoding="utf-8", newline="\n")
            sources = []
            for number, fragment in enumerate(registry.fragments, start=1):
                sources.append({**fragment_json(fragment, self), "number": number})
            done = {"slug": slug, "path": str(target), "markdown": markdown, "sources": sources}
            yield "done", done
        finally:
            self._generating.release()

    # --- practice (Practicar page, ADR-020) ----------------------------------------------------

    @property
    def practice(self) -> PracticeStore:
        if self._practice is None:
            self._practice = PracticeStore(self.config.practice.db)
        return self._practice

    def practice_guides(self) -> list[tuple[str, str]]:
        """(path, title) of every indexed guide, by title."""
        with self.lock:
            rows = self._conn.execute("SELECT path, title FROM documents ORDER BY title").fetchall()
        return [(str(path), str(title)) for path, title in rows]

    def start_quiz(
        self, kind: str, value: str, count: int, kinds: Sequence[str]
    ) -> Generator[Event, None, None]:
        """Events: ("quiz", …), then ("question", …) or ("skipped", …) per section as each is
        generated, then ("done", …). Raises ValueError before the first event if there is nothing
        to practice in that scope."""
        self._claim_generation()
        try:
            rng = random.Random()
            store = self.practice
            latest = store.latest_results()
            review = [k for k, (result, _) in latest.items() if result != "correct"]
            if kind != "review":
                review = []
            with self.lock:
                retriever = self._base() if kind == "topic" else None
                candidates = candidate_sections(
                    self._conn, self.config, kind, value, retriever, review
                )
            sections = choose_sections(
                candidates, latest, store.flagged_sections(), count, rng, keep_order=kind == "topic"
            )
            if not sections:
                raise ValueError("No encontré secciones con contenido para practicar ahí.")
            plan = list(zip(sections, assign_kinds(kinds, len(sections), rng), strict=True))
            quiz_id = store.new_quiz(kind, value, count)
            yield "quiz", {"quiz_id": quiz_id, "planned": len(plan)}
            with self.lock:
                self._release_reranker()  # the GPU is the LLM's now (ADR-014)
            llm = self._llm_factory(self.config)
            generated = 0
            try:
                for index, (section, question_kind) in enumerate(plan):
                    with self._llm_lock:
                        content = generate_question(llm, self.config, section, question_kind, rng)
                    if content is None:
                        yield "skipped", {"index": index, "citation": section.citation}
                        continue
                    qid = store.add_question(quiz_id, question_kind, section, content)
                    stored = store.question(qid)
                    assert stored is not None
                    generated += 1
                    yield "question", {"index": index, **question_json(stored)}
            finally:
                llm.close()
            yield "done", {"generated": generated, "skipped": len(plan) - generated}
        finally:
            self._generating.release()

    def check_choice(self, question_id: int, choice: int | bool) -> dict[str, Any] | None:
        """Grade a multiple-choice index or a true/false answer by code and record it."""
        q = self.practice.question(question_id)
        if q is None or q.kind not in ("mc", "tf"):
            return None
        if q.kind == "mc":
            ok = not isinstance(choice, bool) and choice == q.content.correct
        else:
            ok = isinstance(choice, bool) and choice == q.content.truth
        result = "correct" if ok else "wrong"
        attempt = self.practice.record_attempt(question_id, json.dumps(choice), result, "code")
        return {"result": result, "attempt_id": attempt, **self.feedback_json(q)}

    def reveal(self, question_id: int) -> dict[str, Any] | None:
        """The reference answer of an open question, for the user to compare (nothing recorded)."""
        q = self.practice.question(question_id)
        if q is None or q.kind not in ("open_self", "open_model"):
            return None
        return self.feedback_json(q)

    def self_grade(self, question_id: int, answer: str, result: str) -> int | None:
        q = self.practice.question(question_id)
        if q is None or q.kind not in ("open_self", "open_model"):
            return None
        return self.practice.record_attempt(question_id, answer, result, "self")

    def model_grade(self, question_id: int, answer: str) -> dict[str, Any] | None:
        """The model's grade of an open answer (experimental); `graded` is False when the model
        gave no usable grade, and the page then asks the user to grade it."""
        q = self.practice.question(question_id)
        if q is None or q.kind != "open_model":
            return None
        with self.lock:
            self._release_reranker()
        llm = self._llm_factory(self.config)
        try:
            with self._llm_lock:
                grade = grade_answer(llm, self.config, q, answer)
        finally:
            llm.close()
        if grade is None:
            return {"graded": False, **self.feedback_json(q)}
        attempt = self.practice.record_attempt(
            question_id, answer, grade.result, "model", grade.reason
        )
        return {
            "graded": True,
            "result": grade.result,
            "reason": grade.reason,
            "attempt_id": attempt,
            **self.feedback_json(q),
        }

    def practice_progress(self) -> dict[str, Any]:
        p = self.practice.progress()
        cfg = self.config.practice
        agreement = p.grader_agreed / p.grader_reviewed if p.grader_reviewed else None
        return {
            "guides": [
                {
                    "doc_path": g.doc_path,
                    "title": g.title,
                    "answered": g.answered,
                    "score": g.score,
                    "last": g.last,
                    "to_review": list(g.to_review),
                }
                for g in p.guides
            ],
            "to_review": p.to_review,
            "questions": p.questions,
            "flagged": p.flagged,
            "grader": {
                "reviewed": p.grader_reviewed,
                "agreed": p.grader_agreed,
                "unreliable": agreement is not None
                and p.grader_reviewed >= cfg.grader_min_samples
                and agreement < cfg.grader_min_agreement,
            },
        }

    def feedback_json(self, q: StoredQuestion) -> dict[str, Any]:
        """What the page shows after an answer: the key, the explanation and the source."""
        s = q.section
        return {
            "explanation": q.content.explanation,
            "correct": q.content.correct,
            "truth": q.content.truth,
            "reference": q.content.reference,
            "source": {
                "citation": s.citation,
                **_place(s.doc_path, s.heading_path),
                "lines": [s.start_line, s.end_line],
                "text": s.text,
                "open_url": self.open_url(s.doc_path),
            },
        }

    # --- updating the index from the page -----------------------------------------------------

    def reindex(self) -> Generator[Event, None, None]:
        """Incremental reindex, as `biblio index` (only changed files are re-embedded). Events:
        ("start", {files, device}), ("progress", …) a few times a second, ("done", {stats}).
        The GPU is used when free, the CPU while the LLM server holds it (ADR-018 rule)."""
        library = self.config.paths.library
        if library is None:
            raise ValueError("No hay biblioteca configurada (paths.library en config/local.toml).")
        choice = choose_device(self._index_device(self.config), self.llm_running())
        if choice is None:
            raise ValueError(
                "El índice está en float16 y el modelo de lenguaje ocupa la GPU: detenlo y vuelve "
                "a intentarlo."
            )

        def work(emit: Callable[[Event], None]) -> IndexStats:
            seen = {"files": 0, "chunks": 0, "total": 0}
            last = [0.0]

            def on_start(total: int) -> None:
                seen["total"] = total
                emit(("start", {"files": total, "device": choice.device}))

            def progress(chunks: int) -> None:
                seen["files"] += 1
                seen["chunks"] += chunks
                now = time.monotonic()
                if now - last[0] >= 0.25 or seen["files"] == seen["total"]:
                    last[0] = now
                    emit(("progress", dict(seen)))

            try:
                return index_library(
                    self.config,
                    library,
                    full=False,
                    dtype=choice.dtype,
                    embedder_factory=lambda: self._index_embedder_factory(self.config, choice),
                    counter_factory=lambda: self._index_counter_factory(self.config),
                    progress=progress,
                    on_start=on_start,
                )
            except IndexLockedError as exc:
                busy = "La biblioteca ya se está actualizando; espera a que termine."
                raise BusyError(busy) from exc
            except IndexMismatchError as exc:
                raise ValueError(
                    "El índice se construyó con otra configuración: hay que reconstruirlo "
                    "con `uv run biblio index --full`."
                ) from exc

        stats = yield from _threaded(work)
        yield (
            "done",
            {
                "added": stats.added,
                "updated": stats.updated,
                "deleted": stats.deleted,
                "unchanged": stats.unchanged,
                "failed": list(stats.failed),
                "chunks_written": stats.chunks_written,
                "documents": stats.total_documents,
                "seconds": round(stats.seconds, 1),
                "device": choice.device,
            },
        )

    # --- conversation history (ADR-023) -------------------------------------------------------

    @property
    def conversations(self) -> ConversationStore:
        if self._conversations is None:
            self._conversations = ConversationStore(self.config.conversations.db)
        return self._conversations

    def ask_recorded(
        self, question: str, *, scope: str | None, strategy: str, conversation_id: int | None
    ) -> Generator[Event, None, None]:
        """`ask`, stored in the history when the answer is complete: the "done" event carries the
        conversation id. An error is stored too and sent as an ("error", …) event with the id."""
        map_reduce = "on" if strategy == "map-reduce" else "off"  # opt-in only (ADR-017)
        meta: dict[str, Any] = {}
        parts: list[str] = []
        try:
            for name, data in self.ask(question, scope=scope, map_reduce=map_reduce):
                if name == "meta":
                    meta = data
                elif name == "token":
                    parts.append(data["text"])
                elif name == "done":
                    turn = NewTurn(
                        question,
                        "explain",
                        scope,
                        strategy,
                        "done",
                        answer="".join(parts).strip(),
                        payload={**meta, "done": data},
                    )
                    cid, _ = self.conversations.add_turn(conversation_id, turn)
                    data = {**data, "conversation_id": cid}
                yield name, data
        except Exception as exc:  # stored, then reported to the page with the conversation id
            turn = NewTurn(question, "explain", scope, strategy, "error", error=str(exc))
            cid, _ = self.conversations.add_turn(conversation_id, turn)
            yield "error", {"error": str(exc), "conversation_id": cid}

    # --- concept maps (Mapas page, ADR-022) ---------------------------------------------------

    def _embed_names(self, names: Sequence[str]) -> np.ndarray:
        with self.lock:  # the query embedder is shared with searches
            return self._embedder.embed(list(names))

    def concept_map(self, topic: str, scope: str | None = None) -> Generator[Event, None, None]:
        """Events: ("start", {fragments}), then ("map", …) after each fragment with the map so far,
        then ("done", …). Raises ValueError before the first event if nothing is found."""
        self._claim_generation()
        try:
            cfg = self.config.maps
            with self.lock:
                hits = self._base(scope).search(topic, cfg.fragments * 2)
            fragments = map_fragments(hits, cfg)
            if not fragments:
                raise ValueError("No encontré secciones sobre ese tema en tus notas.")
            sources = [fragment_json(f, self) for f in fragments]
            yield "start", {"topic": topic, "fragments": sources}
            with self.lock:
                self._release_reranker()  # the GPU is the LLM's now (ADR-014)
            llm = self._llm_factory(self.config)
            progress = None
            try:
                embed = CachedEmbed(self._embed_names)
                turns = lambda: self._llm_lock  # noqa: E731  (generation and grading take turns)
                for progress in generate_map(llm, cfg, topic, fragments, embed, turns):
                    yield (
                        "map",
                        {
                            "processed": progress.processed,
                            "skipped": progress.skipped,
                            "map": map_json(progress.map),
                        },
                    )
            finally:
                llm.close()
            assert progress is not None
            map_id = self.map_history.add(topic, map_json(progress.map), sources)  # ADR-024
            done = {"map_id": map_id, "skipped": progress.skipped, "omitted": progress.map.omitted}
            yield "done", done
        finally:
            self._generating.release()

    @property
    def map_history(self) -> MapStore:
        if self._map_history is None:
            self._map_history = MapStore(self.config.maps.history_db)
        return self._map_history

    def save_map(self, map_id: int) -> dict[str, str] | None:
        """Write a stored map to drafts/maps/ as Markdown and Obsidian Canvas."""
        stored = self.map_history.get(map_id)
        if stored is None:
            return None
        cmap = concept_map_from_page(stored.map, stored.positions)
        fragments = [fragment_from_page(f) for f in stored.fragments]
        slug = slugify(cmap.topic)
        folder = self.config.paths.drafts_dir / "maps"
        folder.mkdir(parents=True, exist_ok=True)
        markdown = to_markdown(cmap, fragments, date.today())
        md_path, canvas_path = folder / f"{slug}.md", folder / f"{slug}.canvas"
        md_path.write_text(markdown, encoding="utf-8", newline="\n")
        canvas = json.dumps(to_canvas(cmap, fragments), ensure_ascii=False, indent=2)
        canvas_path.write_text(canvas + "\n", encoding="utf-8", newline="\n")
        return {"markdown": str(md_path), "canvas": str(canvas_path), "outline": markdown}

    # --- helpers ----------------------------------------------------------------------------------

    def open_url(self, doc_path: str) -> str | None:
        """`obsidian://` link to the note: a local URI scheme, no network involved."""
        library = self.config.paths.library
        if library is None:
            return None
        return "obsidian://open?path=" + quote(str(Path(library) / doc_path), safe="")

    def status(self) -> dict[str, Any]:
        with self.lock:
            docs, chunks = (
                self._conn.execute("SELECT count(*) FROM documents").fetchone()[0],
                self._conn.execute("SELECT count(*) FROM chunks").fetchone()[0],
            )
        llm = self.llm_running()
        mode = self.config.retrieval.default_mode
        rerank = mode.endswith("-rerank") and not llm
        return {
            "documents": docs,
            "chunks": chunks,
            "mode": mode
            if rerank or not mode.endswith("-rerank")
            else mode.removesuffix("-rerank"),
            "llm_running": llm,
            "llm_model": self.config.llm.model,
            "final_k": self.config.retrieval.final_k,
        }

    def close(self) -> None:
        self._release_reranker()
        self._embedder.close()
        if self._practice is not None:
            self._practice.close()
        if self._conversations is not None:
            self._conversations.close()
        if self._map_history is not None:
            self._map_history.close()


def excerpt(text: str) -> str:
    lines = [
        ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")
    ]
    flat = " ".join(lines)
    return flat if len(flat) <= _EXCERPT_CHARS else flat[: _EXCERPT_CHARS - 1] + "…"


def _place(doc_path: str, heading_path: tuple[str, ...]) -> dict[str, Any]:
    """Where a passage lives, split for display: guide title, section path, file."""
    return {
        "doc_path": doc_path,
        "guide": heading_path[0] if heading_path else doc_path,
        "sections": list(heading_path[1:]),
    }


def hit_json(hit: Hit, engine: Engine) -> dict[str, Any]:
    return {
        "rank": hit.rank,
        "score": round(hit.score, 4),
        "citation": hit.citation,
        **_place(hit.doc_path, hit.heading_path),
        "lines": [hit.start_line, hit.end_line],
        "excerpt": excerpt(hit.text),
        "text": hit.text,
        "open_url": engine.open_url(hit.doc_path),
    }


def fragment_json(fragment: Fragment, engine: Engine) -> dict[str, Any]:
    return {
        "number": fragment.number,
        "citation": fragment.citation,
        **_place(fragment.doc_path, fragment.heading_path),
        "lines": [fragment.start_line, fragment.end_line],
        "excerpt": excerpt(fragment.text),
        "text": fragment.text,
        "open_url": engine.open_url(fragment.doc_path),
    }


def outline_json(outline: Outline) -> dict[str, Any]:
    return {
        "topic": outline.topic,
        "sections": [{"title": s.title, "notes": s.notes} for s in outline.sections],
        "siblings": list(outline.siblings),
    }


def question_json(q: StoredQuestion) -> dict[str, Any]:
    """A question as the page sees it before answering: no answer key, no source text."""
    return {
        "id": q.id,
        "kind": q.kind,
        "text": q.content.text,
        "options": list(q.content.options),
        "guide": q.section.guide,
    }


def map_json(cmap: ConceptMap) -> dict[str, Any]:
    return {
        "topic": cmap.topic,
        "nodes": [
            {
                "id": n.id,
                "label": n.label,
                "level": n.level,
                "parent": n.parent,
                "sources": list(n.sources),
                "x": n.x,
                "y": n.y,
            }
            for n in cmap.nodes
        ],
        "edges": [
            {"source": e.source, "target": e.target, "label": e.label, "kind": e.kind}
            for e in cmap.edges
        ],
        "omitted": cmap.omitted,
    }
