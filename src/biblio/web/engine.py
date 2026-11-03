"""Warm models shared by the web UI's requests (ADR-016, ADR-019).

The query embedder stays loaded on CPU. The reranker follows ADR-014 *per request*: it is loaded
on the GPU only while no LLM server listens, and released before every LLM call, so it never shares
the GPU with the LLM. One request at a time uses the models (single user, `lock`).
"""

from __future__ import annotations

import bisect
import json
import math
import os
import queue
import random
import re
import sqlite3
import threading
import time
from collections import Counter
from collections.abc import Callable, Generator, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar
from urllib.parse import quote, unquote

import numpy as np

from biblio.books import Book, book_id, find_books
from biblio.config import Config, reason_llm
from biblio.conversations import ConversationStore, NewTurn
from biblio.generate.ask import PreparedAsk, Tutor
from biblio.generate.citations import extract_citations
from biblio.generate.context import Context, Fragment
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
from biblio.generate.reason import REASON, Gate, ReasonChecks, Reasoner, topic_gate
from biblio.generate.thread import Plan, PriorTurn, gist
from biblio.generate.translate import translate_query
from biblio.index.embedder import (
    DeviceChoice,
    Embedder,
    load_index_embedder,
    resolve_index_device,
)
from biblio.index.indexer import IndexLockedError, IndexMismatchError, IndexStats, index_library
from biblio.index.watch import choose_device
from biblio.ingest.tokens import TokenCounter, load_llm_token_counter, load_token_counter
from biblio.launcher import LLMSwitch, set_local_value
from biblio.library import (
    INVALID,
    RELOCATE,
    SAME,
    LibraryCheck,
    Listing,
    browse,
    check_library,
    save_library,
)
from biblio.maps.build import (
    CachedEmbed,
    ConceptMap,
    fold,
    fragment_focus,
    generate_map,
    in_english,
    topic_in_text,
)
from biblio.maps.diagrams import (
    DIAGRAM_KINDS,
    Fishbone,
    Flow,
    group_causes,
    move_cause,
    move_step,
    suggest_kind,
)
from biblio.maps.diagrams import generate_diagram as run_diagram
from biblio.maps.drawing import (
    diagram_canvas,
    diagram_from_page,
    diagram_markdown,
    diagram_page,
    novak_layout,
    novak_markdown_block,
)
from biblio.maps.export import map_fragments, numbered_fragments, to_canvas, to_markdown
from biblio.maps.history import MapStore, concept_map_from_page, fragment_from_page
from biblio.maps.select import (
    count_general,
    general_first,
    map_candidates,
    order_sections,
    rerank_scores,
)
from biblio.offline import loopback_port_open
from biblio.practice.dictionary import Entry, build_dictionary, to_anki
from biblio.practice.dictionary import to_markdown as dictionary_markdown
from biblio.practice.feynman import (
    Idea,
    check_claims,
    check_coverage,
    extract_ideas,
    judge_general,
    proposed_grade,
)
from biblio.practice.quiz import (
    assign_kinds,
    candidate_sections,
    choose_sections,
    generate_question,
    grade_answer,
    is_content_heading,
)
from biblio.practice.srs import FSRS, SectionSchedule, due_today, schedule
from biblio.practice.store import (
    FeynmanSession,
    PracticeStore,
    QuestionContent,
    Section,
    StoredQuestion,
)
from biblio.reader import note_tree
from biblio.reset import MemoryItem, ResetResult, memory_items, reset_memory
from biblio.retrieve.bilingual import BilingualRetriever
from biblio.retrieve.bm25 import BM25Retriever
from biblio.retrieve.dense import DenseRetriever
from biblio.retrieve.fusion import HybridRetriever
from biblio.retrieve.navigation import SkipNavigation
from biblio.retrieve.rerank import (
    Reranker,
    RerankingRetriever,
    RerankPlan,
    load_reranker,
    resolve_rerank_plan,
)
from biblio.retrieve.store import library_folders, open_index
from biblio.retrieve.types import Hit, Retriever
from biblio.study.gaps import GapStore, extract_topic, same_topic
from biblio.study.plan import (
    PlanItem,
    PlanProgress,
    folder_guides,
    guide_states,
    next_guide,
    plan_folder,
)
from biblio.study.spotter import UNEXPLAINED, Coverage, GuideCard, spot_gaps
from biblio.study.summaries import (
    BOOK,
    CHAPTER,
    ChapterLine,
    Summary,
    SummaryStore,
    book_hash,
    summarize_book,
    summarize_document,
)
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


class _FixedHits:
    """A search whose answer is already known: the topic's general sections (ADR-062)."""

    name = "general"

    def __init__(self, hits: Sequence[Hit]) -> None:
        self._hits = list(hits)

    def search(self, query: str, k: int) -> list[Hit]:
        return self._hits[:k]

    def close(self) -> None:
        pass


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
        self.llm_switch: LLMSwitch | None = None  # set by `biblio start`/`serve` (ADR-043)
        self._asset_files: dict[str, list[str]] | None = None  # file name → paths (ADR-048)
        self._asset_walked = 0.0
        self._english_books: dict[str, bool] = {}  # book id → its chapters are in English (ADR-053)
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
        self._gap_store: GapStore | None = None
        self._conversations: ConversationStore | None = None
        self._summaries: SummaryStore | None = None

    # --- retrieval --------------------------------------------------------------------------------

    def _base(self, scope: str | None = None) -> Retriever:
        """The configured first stage, optionally limited to one folder (ADR-019)."""
        r = self.config.retrieval
        dense = DenseRetriever(self._conn, self._embedder, scope)
        bm25 = BM25Retriever(self._conn, scope, r.bm25_heading_weight)
        base_mode = r.default_mode.partition("-")[0]
        base: Retriever = (
            dense
            if base_mode == "dense"
            else bm25
            if base_mode == "bm25"
            else HybridRetriever(dense, bm25, r)
        )
        # tables of contents are never results, nor reranker candidates (ADR-044)
        return SkipNavigation(base, r.navigation_margin) if r.skip_navigation else base

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

    def is_note(self, path: str) -> bool:
        """Whether `path` is an indexed note (a search inside it, ADR-045)."""
        with self.lock:
            row = self._conn.execute("SELECT 1 FROM documents WHERE path = ?", (path,)).fetchone()
        return row is not None

    def folders(self) -> list[tuple[str, int]]:
        with self.lock:
            return library_folders(self._conn)

    # --- ask --------------------------------------------------------------------------------------

    def llm_running(self) -> bool:
        return loopback_port_open(self.config.llm.base_url, self.config.llm.probe_timeout_s)

    def ask(
        self,
        question: str,
        *,
        scope: str | None = None,
        map_reduce: str = "off",
        passage: Hit | None = None,
        level: str | None = None,
        frame: str = "",
        history: Sequence[PriorTurn] = (),
        force: bool = False,
        reason: bool = False,
    ) -> Generator[Event, None, None]:
        """UI events: ("meta", …) once, ("token", …) while generating, ("done", …) at the end.
        In a conversation (ADR-038) the message is planned first from its topic (`frame`) and
        previous turns; a message on another topic gets ("offtopic", …) instead of an answer,
        unless `force`. `reason` answers in the «Razonar» mode (ADR-056), with a ("gate", …)
        event before "meta"; a question about a passage is always the tutor's."""
        if reason and passage is None:
            yield from self._ask_reasoned(question, scope, frame, history, force)
            return
        with self.lock:
            self._release_reranker()  # before the LLM loads its model on the GPU (ADR-014)
            if self._counter is None:
                self._counter = self._counter_factory(self.config)
            llm = self._llm_factory(self.config)
            try:
                base = self._base(scope)
                if self.config.retrieval.bilingual:  # also searched in English (ADR-052)
                    r = self.config.retrieval
                    base = BilingualRetriever(
                        base, lambda q: translate_query(llm, r, q), r.rrf_k, r.bilingual_head
                    )
                tutor = Tutor(self.config, base, llm, self._counter, map_reduce, level)
                t0 = time.perf_counter()
                plan: Plan | None = None
                if passage is not None:  # a passage of a note is fragment [1] (ADR-031)
                    prepared = tutor.prepare_passage(
                        question, passage, passage.text, self._section_of(passage)
                    )
                elif self.config.thread.plan:
                    plan = tutor.plan(question, frame, history)
                    if plan.relation == "nuevo" and not force:
                        yield "offtopic", {"frame": frame, "question": question}
                        return
                    # answered here anyway: on its own, the conversation's topic unchanged
                    prepared = tutor.prepare_planned(
                        plan, () if plan.relation == "nuevo" else history
                    )
                else:
                    prepared = tutor.prepare(question)
                yield (
                    "meta",
                    {
                        "retriever": prepared.retriever,
                        "strategy": prepared.strategy,
                        "fragments": [fragment_json(f, self) for f in prepared.context.fragments],
                        "plan": plan_json(plan, time.perf_counter() - t0) if plan else None,
                        # the question as it was also searched in English (ADR-052)
                        "translation": getattr(base, "last_translation", None),
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

    def _ask_reasoned(
        self,
        question: str,
        scope: str | None,
        frame: str,
        history: Sequence[PriorTurn],
        force: bool,
    ) -> Generator[Event, None, None]:
        """`ask` in the «Razonar» mode (ADR-056, ADR-057): the plan, the field's names and the
        answer all on the mode's own model, so one model stays loaded; the gate's reranker runs
        on CPU and is released before the answer, when RAM is tightest."""
        config = replace(self.config, llm=reason_llm(self.config))
        with self.lock:
            counter = self._llm_turn()
            llm = self._llm_factory(config)
            try:
                base = self._base(scope)
                tutor = Tutor(config, base, llm, counter, "off")
                t0 = time.perf_counter()
                plan: Plan | None = None
                if self.config.thread.plan:
                    plan = tutor.plan(question, frame, history)
                    if plan.relation == "nuevo" and not force:
                        yield "offtopic", {"frame": frame, "question": question}
                        return
                standalone = plan.question if plan else question
                reranker = self._reranker_factory(self.config, "cpu")
                try:
                    reasoner = Reasoner(config, tutor, llm, base, reranker.score)
                    gate = reasoner.gate(standalone)
                finally:
                    reranker.close()
                yield "gate", gate_json(gate)
                if not gate.open:  # nothing searched: the library does not explain the field
                    prepared = PreparedAsk(standalone, Context((), 0, 0), (), base.name)
                elif plan is not None:
                    prepared = tutor.prepare_planned(
                        plan, () if plan.relation == "nuevo" else history
                    )
                else:
                    prepared = tutor.prepare(question)
                yield (
                    "meta",
                    {
                        "retriever": prepared.retriever,
                        "strategy": REASON,
                        "fragments": [fragment_json(f, self) for f in prepared.context.fragments],
                        "plan": plan_json(plan, time.perf_counter() - t0) if plan else None,
                        "translation": None,
                    },
                )
                parts = []
                for token in reasoner.stream(prepared, gate):
                    parts.append(token)
                    yield "token", {"text": token}
                answer = reasoner.finish(prepared, "".join(parts).strip(), gate)
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
                        "reason": checks_json(answer.reason) if answer.reason else None,
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
                    drafter = Drafter(
                        self.config,
                        self._base(),
                        llm,
                        counter,
                        self._conn,
                        lambda q: translate_query(llm, self.config.retrieval, q),
                    )
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
            # the sections are written by the «Razonar» model (ADR-057, ADR-061)
            config = (
                replace(self.config, llm=reason_llm(self.config))
                if self.config.draft.reason_sections
                else self.config
            )
            llm = self._llm_factory(config)
            try:
                drafter = Drafter(
                    config,
                    _LockedRetriever(self._base(), self.lock),
                    llm,
                    counter,
                    self._conn,
                    lambda q: translate_query(llm, config.retrieval, q),
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
                meta = drafter.frame(outline, results, registry)
            finally:
                llm.close()
            markdown = render_draft(topic, results, registry, date.today(), meta)
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

    def source_scope(self, kind: str | None, value: str) -> str | None:
        """Where retrieval looks for a source chosen on a page (ADR-063): None is the whole
        library; a folder, a book (a folder) or a note (a path ending in `.md`) limits it."""
        if kind in ("guide", "chapter", "book", "folder") and value.strip("/"):
            return value.strip("/")
        return None

    def valid_source(self, kind: str, value: str) -> bool:
        """Whether a source named by a page exists: a note, a book or a folder of the library."""
        if kind == "library":
            return True
        if kind in ("guide", "chapter"):
            return value in dict(self.practice_guides())
        if kind == "book":
            return any(b.id == value for b in self._books())
        return kind == "folder" and value.strip("/") in dict(self.folders())

    def own_guides(self) -> list[dict[str, str]]:
        """The user's own notes (not book chapters) for the «Una guía» picker, by title."""
        folder = self.config.library.books_folder
        return [
            {"path": p, "title": t, "folder": p.rpartition("/")[0]}
            for p, t in self.practice_guides()
            if book_id(p, folder) is None
        ]

    def practice_guides(self) -> list[tuple[str, str]]:
        """(path, title) of every indexed guide, by title."""
        with self.lock:
            rows = self._conn.execute("SELECT path, title FROM documents ORDER BY title").fetchall()
        return [(str(path), str(title)) for path, title in rows]

    def _general_hits(
        self, topic: str, search: Callable[[str, int], list[Hit]], limit: int
    ) -> list[Hit]:
        """The sections that explain a general topic first (ADR-062): the bare topic's and the
        definitional question's hits fused, scored against that question by the reranker on CPU
        (the LLM may hold the GPU), the ones that explain it kept when there are enough. «dirección
        IP» had brought a router's DHCP settings before its definition."""
        cfg = self.config.maps
        with self.lock:
            hits = map_candidates(search, topic, cfg, self.config.retrieval.rrf_k)
        if not cfg.rerank or not hits:
            return hits[:limit]
        scores = self._map_rerank(topic, hits)
        return general_first(hits, scores, cfg.general_score, cfg.min_general)[:limit]

    def _topic_in_english(self, topic: str) -> str | None:
        """A quiz topic translated by the LLM (ADR-060): «estadística» is not written «statistics»
        and most of the library is English. None when it already is English or nothing came back."""
        if not self.config.practice.topic_match or in_english(topic):
            return None
        with self.lock:
            self._release_reranker()  # the GPU is the LLM's (ADR-014)
        llm = self._llm_factory(self.config)
        try:
            with self._llm_lock:
                text = translate_query(llm, self.config.retrieval, topic)
        finally:
            llm.close()
        return text if text and fold(text) != fold(topic) else None

    def start_quiz(
        self,
        kind: str,
        value: str,
        count: int,
        kinds: Sequence[str],
        within: tuple[str, str] | None = None,
    ) -> Generator[Event, None, None]:
        """Events: ("quiz", …), then ("question", …) or ("skipped", …) per section as each is
        generated, then ("done", …). Raises ValueError before the first event if there is nothing
        to practice in that scope."""
        self._claim_generation()
        try:
            rng = random.Random()
            store = self.practice
            latest = store.latest_results()
            flagged = store.flagged_sections()
            if kind in ("srs", "sections"):
                # explicit sections, in priority order: today's reviews (ADR-025) or a map node's
                if kind == "srs":
                    due, _, remaining = self._today_queue()
                    keys = [s.key for s in due][:remaining]
                else:
                    keys = parse_section_keys(value)
                with self.lock:
                    found = candidate_sections(self._conn, self.config, "review", "", review=keys)
                first: dict[tuple[str, str], Section] = {}
                for c in found:
                    first.setdefault(c.key, c.section)
                sections = [first[k] for k in keys if k in first and k not in flagged][:count]
            else:
                review = [k for k, (result, _) in latest.items() if result != "correct"]
                if kind != "review":
                    review = []
                english = self._topic_in_english(value) if kind == "topic" else None
                with self.lock:
                    where = self.source_scope(*within) if within else None
                    retriever: Retriever | None = self._base(where) if kind == "topic" else None
                    if retriever is not None and english:  # the English search joins the pool
                        r = self.config.retrieval
                        # both searches with equal weight: a quiz wants notes and books alike
                        retriever = BilingualRetriever(retriever, lambda _q: english, r.rrf_k)
                if retriever is not None and self.config.practice.topic_general:
                    # the sections that explain the topic, not those that only apply it (ADR-062)
                    retriever = _FixedHits(
                        self._general_hits(
                            value, retriever.search, self.config.practice.max_questions * 3
                        )
                    )
                with self.lock:
                    candidates = candidate_sections(
                        self._conn, self.config, kind, value, retriever, review, english
                    )
                sections = choose_sections(
                    candidates, latest, flagged, count, rng, keep_order=kind == "topic"
                )
            if not sections:
                if kind == "srs":
                    raise ValueError("No tienes repasos pendientes hoy.")
                raise ValueError("No encontré secciones con contenido para practicar ahí.")
            plan = list(zip(sections, assign_kinds(kinds, len(sections), rng), strict=True))
            # a review without the LLM asks the section's last stored question again (ADR-025)
            reuse = kind == "srs" and not self.llm_running()
            quiz_id = store.new_quiz(kind, value, count)
            yield "quiz", {"quiz_id": quiz_id, "planned": len(plan), "reused": reuse}
            llm = None
            if not reuse:
                with self.lock:
                    self._release_reranker()  # the GPU is the LLM's now (ADR-014)
                llm = self._llm_factory(self.config)
            generated = 0
            try:
                for index, (section, question_kind) in enumerate(plan):
                    if llm is None:
                        old = store.latest_question(_section_key(section))
                        qid = (
                            store.add_question(quiz_id, old.kind, section, old.content)
                            if old
                            else None
                        )
                    else:
                        with self._llm_lock:
                            content = generate_question(
                                llm, self.config, section, question_kind, rng
                            )
                        qid = (
                            store.add_question(quiz_id, question_kind, section, content)
                            if content
                            else None
                        )
                    if qid is None:
                        yield "skipped", {"index": index, "citation": section.citation}
                        continue
                    stored = store.question(qid)
                    assert stored is not None
                    generated += 1
                    yield "question", {"index": index, **question_json(stored)}
            finally:
                if llm is not None:
                    llm.close()
            yield "done", {"generated": generated, "skipped": len(plan) - generated}
        finally:
            self._generating.release()

    # --- spaced repetition (ADR-025) and mastery (ADR-026) ------------------------------------

    def _fsrs(self) -> FSRS:
        p = self.config.practice
        return FSRS(p.fsrs_weights, p.desired_retention, p.max_interval_days)

    def _schedules(self) -> dict[tuple[str, str], SectionSchedule]:
        now = datetime.now().astimezone()
        return schedule(self.practice.section_histories(), self._fsrs(), now)

    def _today_queue(self) -> tuple[list[SectionSchedule], int, int]:
        """(sections due today, most at risk first; reviews done today; reviews left today)."""
        now = datetime.now().astimezone()
        flagged = self.practice.flagged_sections()
        due = [s for s in due_today(self._schedules(), now) if s.key not in flagged]
        done = self.practice.reviews_done_on(now.date().isoformat())
        return due, done, max(self.config.practice.daily_reviews - done, 0)

    def practice_today(self) -> dict[str, Any]:
        due, done, remaining = self._today_queue()
        today = min(len(due), remaining)
        minutes = math.ceil(today * self.config.practice.review_minutes_per_question)
        return {
            "due": len(due),
            "today": today,
            "done": done,
            "limit": self.config.practice.daily_reviews,
            "minutes": minutes,
        }

    def map_mastery(self, map_id: int) -> dict[str, Any] | None:
        """Per node: how well its practised source sections are remembered today (ADR-026)."""
        stored = self.map_history.get(map_id)
        if stored is None:
            return None
        p = self.config.practice
        schedules = self._schedules()
        keys = {
            f["number"]: (
                f["doc_path"],
                json.dumps([f["guide"], *f["sections"]], ensure_ascii=False),
            )
            for f in stored.fragments
        }
        nodes = {}
        for n in stored.map.get("nodes", []):  # the diagrams of ADR-036 have no mastery
            sections = [keys[s] for s in n["sources"] if s in keys]
            recalls = [schedules[k].recall_now for k in sections if k in schedules]
            recall = sum(recalls) / len(recalls) if recalls else None
            if recall is None:
                state = "new"
            elif recall >= p.mastered_at:
                state = "mastered"
            elif recall >= p.partial_at:
                state = "partial"
            else:
                state = "weak"
            nodes[n["id"]] = {
                "state": state,
                "recall": round(recall, 2) if recall is not None else None,
                "practiced": len(recalls),
                "sections": [list(k) for k in sections],
            }
        return {"nodes": nodes, "mastered_at": p.mastered_at, "partial_at": p.partial_at}

    def note_mastery(self, doc_path: str) -> dict[str, Any] | None:
        """Per entry of the reader's index: how well its sections are remembered today (ADR-050).

        Practice is keyed by a chunk's heading path, which includes headings the index leaves
        out (lines of code, levels below `outline_depth`). A section is placed at its own
        heading: the last heading, up to the end of its first chunk, written as the path's last
        title (a chunk may start at a parent heading with no text of its own, and a merged
        chunk carries its parent's path). That heading belongs to the last index entry at or
        before it; a heading out of the index belongs to the entry above it.
        """
        view = self.note_view(doc_path)
        if view is None:
            return None
        lines = [h["line"] for h in view["outline"]]
        heads = [
            (b["lines"][0], b["title"])
            for b in view["blocks"]
            if b["type"] == "heading" and b.get("lines")
        ]
        with self.lock:
            rows = self._conn.execute(
                "SELECT c.heading_path, c.start_line, c.end_line FROM chunks c "
                "JOIN documents d ON d.id = c.document_id WHERE d.path = ? ORDER BY c.start_line",
                (doc_path,),
            ).fetchall()
        entry_of: dict[tuple[str, ...], int] = {}  # a section → its index entry
        for heading_path, start, end in rows:
            path = tuple(json.loads(heading_path))
            if path in entry_of or not lines:
                continue
            at = start  # the text before the first heading: the first entry
            if len(path) > 1:
                at = next(
                    (ln for ln, title in reversed(heads) if ln <= end and title == path[-1]), start
                )
            entry_of[path] = max(bisect.bisect_right(lines, at) - 1, 0)
        p = self.config.practice
        schedules = {
            (doc, tuple(json.loads(path))): s
            for (doc, path), s in self._schedules().items()
            if doc == doc_path
        }
        flagged = self.practice.flagged_sections()
        end_of_day = datetime.now().astimezone().replace(hour=23, minute=59, second=59)
        recalls: list[list[float]] = [[] for _ in lines]
        due: list[list[str]] = []
        for (_, path), s in schedules.items():
            if path not in entry_of:  # a section the note no longer has
                continue
            recalls[entry_of[path]].append(s.recall_now)
            if s.due <= end_of_day and s.key not in flagged:
                due.append(list(s.key))
        entries = []
        for i, (line, values) in enumerate(zip(lines, recalls, strict=True)):
            recall = sum(values) / len(values) if values else None
            if recall is None:
                state = "new"
            elif recall >= p.mastered_at:
                state = "mastered"
            elif recall >= p.partial_at:
                state = "partial"
            else:
                state = "weak"
            entries.append(
                {
                    "line": line,
                    "state": state,
                    "recall": round(recall, 2) if recall is not None else None,
                    "practiced": len(values),
                    "sections": sum(1 for e in entry_of.values() if e == i),
                }
            )
        return {
            "entries": entries,
            "due": due,
            "mastered_at": p.mastered_at,
            "partial_at": p.partial_at,
        }

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

    def _feynman_agreement(self) -> dict[str, Any]:
        votes, fair = self.practice.feynman_agreement()
        cfg = self.config.feynman
        return {
            "votes": votes,
            "fair": fair,
            "unreliable": votes >= cfg.min_samples and fair / votes < cfg.min_agreement,
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
            "feynman": self._feynman_agreement(),
            "grader": {
                "reviewed": p.grader_reviewed,
                "agreed": p.grader_agreed,
                "unreliable": agreement is not None
                and p.grader_reviewed >= cfg.grader_min_samples
                and agreement < cfg.grader_min_agreement,
            },
        }

    # --- «Explícamelo tú» (ADR-027) -------------------------------------------------------------

    def feynman_start(
        self, concept: str, within: tuple[str, str] | None = None
    ) -> Generator[Event, None, None]:
        """Step 1, while the user writes: find the notes and take their key ideas. One event,
        ("ready", {session_id, …}); the ideas stay on the server until the explanation arrives.
        Raises ValueError (before any event) when the notes say too little."""
        self._claim_generation()
        try:
            cfg = self.config.feynman
            config = self._feynman_config()
            where = self.source_scope(*within) if within else None  # ADR-063
            if cfg.general:  # what explains the concept, not where it is configured (ADR-062)
                hits = self._general_hits(concept, self._base(where).search, cfg.fragments * 2)
            else:
                with self.lock:
                    hits = self._base(where).search(concept, cfg.fragments * 2)
            fragments = numbered_fragments(
                hits, cfg.fragments, cfg.max_fragment_chars, openings=cfg.general
            )
            if len(fragments) < cfg.min_fragments:
                raise ValueError(f"Tus notas no tienen suficiente sobre «{concept}».")
            with self.lock:
                self._release_reranker()  # the GPU is the LLM's now (ADR-014)
            llm = self._llm_factory(config)
            try:
                if cfg.reason:  # the notes must explain the topic (ADR-058, the gate of ADR-056)
                    gate = self._feynman_gate(llm, config, concept, where)
                    if not gate.open:
                        raise ValueError(
                            f"Tus notas no explican «{concept}», así que no puedo comparar tu "
                            "explicación con ellas."
                        )
                with self._llm_lock:
                    ideas = extract_ideas(llm, cfg, concept, fragments)
            finally:
                llm.close()
            if not ideas:
                raise ValueError(
                    f"No encontré en tus notas ideas clave sobre «{concept}» que pueda comprobar."
                )
            session_id = self.practice.new_feynman(
                concept,
                [_fragment_record(f) for f in fragments],
                [{"text": i.text, "fragment": i.fragment, "quote": i.quote} for i in ideas],
            )
            yield "ready", {"session_id": session_id, "ideas": len(ideas), "notes": len(fragments)}
        finally:
            self._generating.release()

    def feynman_check(
        self, session: FeynmanSession, explanation: str
    ) -> Generator[Event, None, None]:
        """Step 2: ("coverage", {ideas}), ("claims", {claims}), then ("done", {proposed, …})."""
        self._claim_generation()
        try:
            cfg = self.config.feynman
            fragments = [_fragment_from_record(r) for r in session.fragments]
            ideas = [Idea(i["text"], i["fragment"], i["quote"]) for i in session.ideas]
            with self.lock:
                self._release_reranker()
            llm = self._llm_factory(self._feynman_config())
            try:
                with self._llm_lock:
                    coverage = check_coverage(llm, cfg, session.concept, ideas, explanation)
                if coverage is None:
                    raise ValueError("No pude comprobar tu explicación; inténtalo otra vez.")
                ideas_json = [
                    {
                        "text": i.text,
                        "fragment": i.fragment,
                        "quote": i.quote,
                        "verdict": c.verdict,
                        "phrase": c.phrase,
                    }
                    for i, c in zip(ideas, coverage, strict=True)
                ]
                yield "coverage", {"ideas": ideas_json}
                with self._llm_lock:
                    claims = check_claims(llm, cfg, session.concept, fragments, explanation)
                    # the session only started because the notes explain the topic (ADR-058)
                    if claims and cfg.reason:
                        claims = judge_general(llm, cfg, session.concept, claims)
            finally:
                llm.close()
            claims_json = [
                {
                    "text": c.text,
                    "verdict": c.verdict,
                    "fragment": c.fragment,
                    "quote": c.quote,
                    "general": c.general,
                    "general_note": c.general_note,
                }
                for c in claims or []
            ]
            yield "claims", {"claims": claims_json, "failed": claims is None}
            score, proposed = proposed_grade(coverage, claims or [], cfg)
            review = {"ideas": ideas_json, "claims": claims_json, "score": round(score, 2)}
            self.practice.set_feynman_review(session.id, explanation, review, proposed)
            yield (
                "done",
                {
                    "score": round(score, 2),
                    "proposed": proposed,
                    "sources": [fragment_json(f, self) for f in fragments],
                },
            )
        finally:
            self._generating.release()

    def _feynman_config(self) -> Config:
        """«Explícamelo tú» on the «Razonar» model when `feynman.reason` (ADR-058)."""
        if not self.config.feynman.reason:
            return self.config
        return replace(self.config, llm=reason_llm(self.config))

    def _feynman_gate(
        self, llm: LLMClient, config: Config, concept: str, where: str | None = None
    ) -> Gate:
        """The topic gate of ADR-056 on the concept, in the part of the library the user chose
        (ADR-063); its CPU reranker is released at once."""
        reranker = self._reranker_factory(self.config, "cpu")
        try:
            with self.lock:
                base = self._base(where)
            return topic_gate(llm, config, base, reranker.score, concept)
        finally:
            reranker.close()

    def feynman_confirm(self, session_id: int, result: str) -> dict[str, Any] | None:
        """The grade the user kept, recorded as practice on each section that gave an idea, so it
        moves the review schedule (ADR-025) and the mastery map (ADR-026). None if the explanation
        was not checked yet or the grade was already recorded."""
        session = self.practice.feynman(session_id)
        if session is None or session.review is None or session.result is not None:
            return None
        by_number = {r["number"]: r for r in session.fragments}
        ideas_by_fragment: dict[int, list[str]] = {}
        for idea in session.ideas:
            ideas_by_fragment.setdefault(idea["fragment"], []).append(idea["text"])
        store = self.practice
        quiz_id = store.new_quiz("feynman", session.concept, len(ideas_by_fragment))
        for number, texts in ideas_by_fragment.items():
            r = by_number[number]
            section = Section(
                r["doc_path"], tuple(r["heading_path"]), r["start_line"], r["end_line"], r["text"]
            )
            content = QuestionContent(
                f"Explica con tus palabras: {session.concept}",
                "Una buena explicación recoge las ideas clave de esta sección.",
                reference=" ".join(f"• {text}" for text in texts),
            )
            qid = store.add_question(quiz_id, "open_self", section, content)
            store.record_attempt(qid, session.explanation or "", result, "self")
        store.set_feynman_result(session_id, result, quiz_id)
        return {"recorded": len(ideas_by_fragment)}

    def feynman_vote(self, session_id: int, point: str, fair: bool) -> bool:
        """«¿Es justo?» on one reported point: "idea:<i>" or "claim:<i>"."""
        session = self.practice.feynman(session_id)
        if session is None or session.review is None:
            return False
        kind, _, index = point.partition(":")
        items = session.review.get({"idea": "ideas", "claim": "claims"}.get(kind, ""), [])
        if not index.isdigit() or not 0 <= int(index) < len(items):
            return False
        self.practice.vote_feynman(session_id, point, fair)
        return True

    # --- study plan per folder (ADR-028) ------------------------------------------------------

    def generate_plan(self, folder: str) -> Generator[Event, None, None]:
        """Events: ("progress", {done, total, label}) after each LLM ordering, then ("plan", view).
        Raises ValueError before any event when the folder has no guides."""
        self._claim_generation()
        try:
            cfg = self.config.plan
            with self.lock:
                guides = folder_guides(self._conn, folder, cfg.headings_per_guide)
            if not guides:
                raise ValueError("Esa carpeta no tiene guías indexadas.")
            with self.lock:
                self._release_reranker()  # the GPU is the LLM's now (ADR-014)
            llm = self._llm_factory(self.config)
            items: list[PlanItem] = []
            try:
                for step in plan_folder(llm, cfg, folder, guides):
                    if isinstance(step, PlanProgress):
                        yield (
                            "progress",
                            {"done": step.done, "total": step.total, "label": step.label},
                        )
                    else:
                        items = step
            finally:
                llm.close()
            self.practice.save_plan(
                folder,
                [
                    {
                        "doc_path": i.doc_path,
                        "reason": i.reason,
                        "module": i.module,
                        "placed": i.placed,
                    }
                    for i in items
                ],
            )
            view = self.plan_view(folder)
            assert view is not None
            yield "plan", view
        finally:
            self._generating.release()

    def plan_view(self, folder: str) -> dict[str, Any] | None:
        """The stored plan with each guide's state today; guides added to the folder since are
        appended as new, deleted ones dropped."""
        stored = self.practice.plan(folder)
        if stored is None:
            return None
        created, items = stored
        with self.lock:
            titles = {g.doc_path: g.title for g in folder_guides(self._conn, folder, 0)}
        kept = [i for i in items if i["doc_path"] in titles]
        planned = {i["doc_path"] for i in kept}
        new = sorted((p for p in titles if p not in planned), key=lambda p: titles[p])
        rows = [{**i, "new": False} for i in kept] + [
            {"doc_path": p, "reason": "", "module": "", "placed": False, "new": True} for p in new
        ]
        order = [r["doc_path"] for r in rows]
        recall = {k: s.recall_now for k, s in self._schedules().items()}
        with self.lock:
            states = guide_states(self._conn, self.config, order, self.practice.reads(), recall)
        counts = {s: 0 for s in ("unread", "read", "practiced", "mastered")}
        for st in states.values():
            counts[st.state] += 1
        nxt = next_guide(order, states)
        return {
            "folder": folder,
            "created": created,
            "items": [
                {
                    **r,
                    "title": titles[r["doc_path"]],
                    "state": states[r["doc_path"]].state,
                    "sections": states[r["doc_path"]].sections,
                    "practiced": states[r["doc_path"]].practiced,
                    "recall": states[r["doc_path"]].recall,
                    "open_url": self.open_url(r["doc_path"]),
                }
                for r in rows
            ],
            "counts": counts,
            "next": {
                "doc_path": nxt[0],
                "title": titles[nxt[0]],
                "why": nxt[1],
                "open_url": self.open_url(nxt[0]),
            }
            if nxt
            else None,
        }

    def plan_list(self) -> list[dict[str, Any]]:
        out = []
        for folder in self.practice.plans():
            view = self.plan_view(folder)
            if view is not None:
                out.append(
                    {
                        "folder": folder,
                        "created": view["created"],
                        "total": len(view["items"]),
                        "counts": view["counts"],
                        "next": view["next"],
                    }
                )
        return out

    def set_plan_order(self, folder: str, order: Sequence[str]) -> bool:
        """The user's order: must hold exactly the guides the plan shows."""
        view = self.plan_view(folder)
        if view is None:
            return False
        rows = {i["doc_path"]: i for i in view["items"]}
        if len(order) != len(rows) or set(order) != set(rows):
            raise ValueError("El orden debe incluir cada guía del plan una sola vez.")
        keys = ("doc_path", "reason", "module", "placed")
        self.practice.set_plan_items(folder, [{k: rows[p][k] for k in keys} for p in order])
        return True

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
        yield from self._index_stream(library, full=False, relocate=False)

    def _index_stream(
        self, library: Path, *, full: bool, relocate: bool
    ) -> Generator[Event, None, None]:
        """One indexing run with its progress as events (`reindex`, `change_library`)."""
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
                emit(("start", {"files": total, "device": choice.device, "full": full}))

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
                    full=full,
                    dtype=choice.dtype,
                    embedder_factory=lambda: self._index_embedder_factory(self.config, choice),
                    counter_factory=lambda: self._index_counter_factory(self.config),
                    progress=progress,
                    on_start=on_start,
                    relocate=relocate,
                    swap=self._swap_index,
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

    def _swap_index(self, replace_index: Callable[[], None]) -> None:
        """A full rebuild replaces the index file: Windows cannot replace a file that is open, so
        the app's connection is closed around it (ADR-039)."""
        with self.lock:
            self._release_reranker()
            self._conn.close()
            try:
                replace_index()
            finally:
                self._conn = open_index(self.config, shared_across_threads=True)

    # --- where the library is (ADR-039) -------------------------------------------------------

    def library_check(self, path: str) -> LibraryCheck:
        with self.lock:
            return check_library(self.config, Path(path), self._conn)

    def browse_folders(self, path: str | None) -> Listing:
        return browse(path, self.config.library.browse_max_entries)

    def change_library(self, path: str) -> Generator[Event, None, None]:
        """Events: ("check", …) with the plan, then the indexing run's events, then ("library",
        {path}) once the new location is in use and saved. Raises ValueError before any event if
        the folder has no notes."""
        found = self.library_check(path)
        if found.plan == INVALID:
            raise ValueError(found.reason)
        yield "check", asdict(found)
        new = Path(found.path)
        if found.plan != SAME:
            # the library is switched only once its index is ready: a failed run changes nothing
            yield from self._index_stream(new, full=found.plan != RELOCATE, relocate=True)
        save_library(self.config, new)
        self.config = replace(self.config, paths=replace(self.config.paths, library=new))
        yield "library", {"path": str(new)}

    # --- conversation history (ADR-023) -------------------------------------------------------

    @property
    def conversations(self) -> ConversationStore:
        if self._conversations is None:
            self._conversations = ConversationStore(self.config.conversations.db)
        return self._conversations

    def ask_recorded(
        self,
        question: str,
        *,
        scope: str | None,
        strategy: str,
        conversation_id: int | None,
        passage: Hit | None = None,
        level: str | None = None,
        force: bool = False,
        reason: bool = False,
    ) -> Generator[Event, None, None]:
        """`ask`, stored in the history when the answer is complete: the "done" event carries the
        conversation id. An error is stored too and sent as an ("error", …) event with the id.
        The conversation's topic and last turns are the context of the message (ADR-038)."""
        map_reduce = "on" if strategy == "map-reduce" else "off"  # opt-in only (ADR-017)
        meta: dict[str, Any] = {}
        parts: list[str] = []
        abstained = False
        try:
            level = level or self.config.tutor.default_level  # ADR-032
            frame, history = self.thread_context(conversation_id)
            asked = self.ask(
                question,
                scope=scope,
                map_reduce=map_reduce,
                passage=passage,
                level=level,
                frame=frame,
                history=history,
                force=force,
                reason=reason,
            )
            for name, data in asked:
                if name == "offtopic":  # nothing answered, nothing stored: the page asks
                    yield name, {**data, "conversation_id": conversation_id}
                    return
                if name == "gate":  # the «Razonar» mode's topic gate, kept with the turn
                    meta["gate"] = data
                elif name == "meta":
                    meta = {**meta, **data}
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
                        payload={**meta, "done": data, "level": level, "reason": reason},
                    )
                    cid, _ = self.conversations.add_turn(conversation_id, turn)
                    plan = meta.get("plan")
                    if plan and plan["relation"] != "nuevo":
                        self.conversations.set_frame(cid, plan["frame"])
                    data = {**data, "conversation_id": cid}
                yield name, data
                abstained = abstained or (name == "done" and bool(data.get("abstained")))
        except Exception as exc:  # stored, then reported to the page with the conversation id
            turn = NewTurn(question, "explain", scope, strategy, "error", error=str(exc))
            cid, _ = self.conversations.add_turn(conversation_id, turn)
            yield "error", {"error": str(exc), "conversation_id": cid}
            return
        if abstained:  # after `ask` has finished and released the lock (ADR-029)
            gap = self._gap_from_question(question)
            if gap is not None:
                yield "gap", gap

    def thread_context(self, conversation_id: int | None) -> tuple[str, list[PriorTurn]]:
        """A conversation's topic and its last explained turns, as the planner and the tutor see
        them (ADR-038). A search-only turn («Solo ubicar») has no answer: its question is kept."""
        if conversation_id is None:
            return "", []
        found = self.conversations.get(conversation_id)
        if found is None:
            return "", []
        summary, turns = found
        cfg = self.config.thread
        done = [t for t in turns if t.status == "done"][-cfg.history_turns :]
        prior = [
            PriorTurn(
                (t.payload.get("plan") or {}).get("question") or t.question,
                gist(t.answer, cfg.gist_chars) if t.mode == "explain" else "",
            )
            for t in done
        ]
        frame = summary.frame or (prior[0].question if prior else "")
        return frame, prior

    # --- gap detector (ADR-029) ----------------------------------------------------------------

    @property
    def gaps(self) -> GapStore:
        if self._gap_store is None:
            conn, lock = self.practice.shared()
            self._gap_store = GapStore(conn, lock)
        return self._gap_store

    def _record_gap(self, topic: str, source: str, text: str, detail: str) -> dict[str, Any] | None:
        """Record one occurrence; what the page shows («Apuntado… · Deshacer»), or None when it
        could not be recorded or the topic was dismissed before."""
        try:
            vector = self._embed_names([topic])[0]
            into = self._same_gap(topic, text, vector)
            rec = self.gaps.record(topic, vector, source, text, detail, into)
        except Exception:  # a gap is a convenience: never break the answer or the map for it
            return None
        if rec.status == "dismissed":  # the user said no to this subject before
            return None
        return {
            "gap_id": rec.gap_id,
            "event_id": rec.event_id,
            "topic": rec.topic,
            "count": rec.count,
            "merged": rec.merged,
        }

    def _same_gap(self, topic: str, text: str, vector: np.ndarray) -> int | None:
        """The known topic on the same subject: embeddings shortlist, the LLM decides (ADR-030)."""
        cfg = self.config.gaps
        found = self.gaps.candidates(vector, cfg.candidate_similarity, cfg.max_candidates)
        if not found:
            return None
        llm = self._llm_factory(self.config)
        try:
            with self._llm_lock:
                chosen = same_topic(llm, cfg, text, topic, [c.topic for c in found])
        finally:
            llm.close()
        return found[chosen].gap_id if chosen is not None else None

    def _gap_from_question(self, question: str) -> dict[str, Any] | None:
        try:
            llm = self._llm_factory(self.config)
            try:
                with self._llm_lock:
                    topic = extract_topic(llm, self.config.gaps, question)
            finally:
                llm.close()
        except Exception:
            return None
        return self._record_gap(topic, "ask", question, "")

    # --- concept maps (Mapas page, ADR-022) ---------------------------------------------------

    def _embed_names(self, names: Sequence[str]) -> np.ndarray:
        with self.lock:  # the query embedder is shared with searches
            return self._embedder.embed(list(names))

    def _map_rerank(self, topic: str, hits: Sequence[Hit]) -> list[float]:
        """The candidates scored by the reranker on CPU, outside the engine's lock: a map needs
        the LLM, which holds the GPU (ADR-014). Loaded for this and released at once: +2.5 GiB
        RAM while it runs (measured, ADR-051)."""
        with self.lock:
            self._release_reranker()  # a GPU reranker kept from a search goes too
        reranker = self._reranker_factory(self.config, "cpu")
        try:
            return rerank_scores(reranker.score, topic, hits, self.config.maps)
        finally:
            reranker.close()

    def concept_map(
        self, topic: str, scope: str | None = None, kind: str = "concepts"
    ) -> Generator[Event, None, None]:
        """Events: ("selecting", {candidates}) while the reranker orders them, ("start",
        {fragments}), then ("map", …) after each fragment with the map so far, then ("done", …).
        Raises ValueError before the first event if nothing is found. `kind` "novak" lays the
        same map out top-down (ADR-036)."""
        self._claim_generation()
        try:
            cfg = self.config.maps
            with self.lock:  # the sections that explain the topic first (ADR-051)
                hits = map_candidates(
                    self._base(scope).search, topic, cfg, self.config.retrieval.rrf_k
                )
            if not map_fragments(hits, cfg):
                self._record_gap(topic, "map", topic, "sin secciones")  # ADR-029
                raise ValueError(
                    "No encontré secciones sobre ese tema en tus notas. "
                    "Lo apunté como tema pendiente en Guías."
                )
            scores = None
            if cfg.rerank:
                yield "selecting", {"candidates": len(hits)}
                scores = self._map_rerank(topic, hits)
            fragments = map_fragments(order_sections(topic, hits, scores), cfg)
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
                            "stage": progress.stage,  # "organizing": the LLM picks the branches
                            "map": map_json(progress.map, kind),
                        },
                    )
                general = None  # how many fragments explain the topic in general (ADR-051)
                if scores is not None:
                    score_of = {
                        (h.doc_path, h.start_line): s for h, s in zip(hits, scores, strict=True)
                    }

                    def focus(f: Fragment) -> str | None:
                        with self._llm_lock:
                            return fragment_focus(llm, cfg, topic, f)

                    general = count_general(topic, fragments, score_of, cfg.general_score, focus)
            finally:
                llm.close()
            assert progress is not None
            map_id = self.map_history.add(topic, map_json(progress.map, kind), sources)  # ADR-024
            done = {"map_id": map_id, "skipped": progress.skipped, "omitted": progress.map.omitted}
            yield "done", done
            concepts = len(progress.map.nodes) - 1  # the topic is a node too
            # absent: no fragment writes the topic, nor names it in its own language (ADR-042)
            absent = not progress.topic_named and not any(
                topic_in_text(topic, f.text) for f in fragments
            )
            # scarce: the notes name the topic but hardly explain it in general (ADR-051 stage 3)
            scarce = general is not None and general < cfg.min_general
            if absent or scarce or concepts < self.config.gaps.map_min_concepts:  # thin (ADR-029)
                detail = (
                    "el tema no aparece en tus notas"
                    if absent
                    else f"explicado en general en {general} de {len(fragments)} secciones"
                    if scarce
                    else f"{concepts} {'concepto' if concepts == 1 else 'conceptos'}"
                )
                gap = self._record_gap(topic, "map", topic, detail)
                if gap is not None:
                    extra = {"general": general, "read": len(fragments)} if scarce else {}
                    yield "gap", {**gap, "absent": absent, **extra}
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
        fragments = [fragment_from_page(f) for f in stored.fragments]
        kind = stored.summary.kind
        folder = self.config.paths.drafts_dir / "maps"
        folder.mkdir(parents=True, exist_ok=True)
        canvas_data: dict[str, Any]
        if kind in DIAGRAM_KINDS:  # ADR-036
            slug = f"{slugify(stored.summary.topic)}-{_KIND_SLUGS[kind]}"
            markdown = diagram_markdown(stored.map, fragments, date.today())
            canvas_data = diagram_canvas(stored.map, fragments)
        else:
            cmap = concept_map_from_page(stored.map, stored.positions)
            slug = slugify(cmap.topic) + ("-mapa-conceptual" if kind == "novak" else "")
            markdown = to_markdown(cmap, fragments, date.today())
            if kind == "novak":
                block = "\n".join(novak_markdown_block(cmap))
                markdown = markdown.replace("\n## Fuentes\n", f"{block}\n\n## Fuentes\n", 1)
            canvas_data = to_canvas(cmap, fragments)
        md_path, canvas_path = folder / f"{slug}.md", folder / f"{slug}.canvas"
        md_path.write_text(markdown, encoding="utf-8", newline="\n")
        canvas = json.dumps(canvas_data, ensure_ascii=False, indent=2)
        canvas_path.write_text(canvas + "\n", encoding="utf-8", newline="\n")
        return {"markdown": str(md_path), "canvas": str(canvas_path), "outline": markdown}

    # --- flowcharts, sequence and cause-effect diagrams (ADR-036) ---------------------------------

    def diagram(
        self, topic: str, kind: str, scope: str | None = None
    ) -> Generator[Event, None, None]:
        """Events: ("start", {fragments}), ("diagram", …) after each fragment and step, then
        ("done", {map_id}); map_id is None when the notes describe nothing of that kind.
        Raises ValueError before the first event if nothing is found."""
        self._claim_generation()
        try:
            cfg = self.config.diagrams
            with self.lock:
                hits = self._base(scope).search(topic, cfg.fragments * 2)
            fragments = numbered_fragments(hits, cfg.fragments, cfg.max_fragment_chars)
            if not fragments:
                self._record_gap(topic, "map", topic, "sin secciones")  # ADR-029
                raise ValueError(
                    "No encontré secciones sobre ese tema en tus notas. "
                    "Lo apunté como tema pendiente en Guías."
                )
            sources = [fragment_json(f, self) for f in fragments]
            yield "start", {"topic": topic, "kind": kind, "fragments": sources}
            with self.lock:
                self._release_reranker()  # the GPU is the LLM's now (ADR-014)
            llm = self._llm_factory(self.config)
            page: dict[str, Any] = {}
            try:
                turns = lambda: self._llm_lock  # noqa: E731  (generation and grading take turns)
                for progress in run_diagram(llm, cfg, kind, topic, fragments, turns):
                    page = diagram_page(kind, topic, progress.diagram)
                    yield (
                        "diagram",
                        {
                            "processed": progress.processed,
                            "skipped": progress.skipped,
                            "stage": progress.stage,
                            "map": page,
                        },
                    )
            finally:
                llm.close()
            map_id = self.map_history.add(topic, page, sources) if page["count"] else None
            yield "done", {"map_id": map_id, "omitted": page["omitted"]}
        finally:
            self._generating.release()

    def suggest_diagram(self, topic: str, scope: str | None = None) -> dict[str, str]:
        """The diagram type the model proposes for a topic, from the headings retrieval finds."""
        self._claim_generation()
        try:
            cfg = self.config.diagrams
            with self.lock:
                hits = self._base(scope).search(topic, cfg.suggest_fragments * 2)
                self._release_reranker()  # ADR-014
            fragments = numbered_fragments(hits, cfg.suggest_fragments, 200)
            headings = [" › ".join(f.heading_path) for f in fragments]
            llm = self._llm_factory(self.config)
            try:
                with self._llm_lock:
                    found = suggest_kind(llm, cfg, topic, headings)
            finally:
                llm.close()
        finally:
            self._generating.release()
        if found is None:
            return {"kind": "concepts", "reason": ""}
        return {"kind": found.kind, "reason": found.reason}

    def edit_diagram(self, map_id: int, edit: dict[str, Any]) -> dict[str, Any] | None:
        """A change by hand, without the LLM: a step moved, a cause moved to another category, or
        another version of a sequence shown. None if the map or the change does not exist."""
        stored = self.map_history.get(map_id)
        if stored is None or stored.summary.kind not in DIAGRAM_KINDS:
            return None
        kind, diagram, current = diagram_from_page(stored.map)
        changed: object | None = None
        op = edit.get("op")
        if op == "move_step" and kind == "flow" and isinstance(diagram, tuple) and diagram:
            delta = edit.get("delta")
            flow = diagram[current]
            moved = (
                move_step(flow, str(edit.get("step")), delta)
                if isinstance(flow, Flow) and isinstance(delta, int) and not isinstance(delta, bool)
                else None
            )
            if moved is not None:
                changed = (*diagram[:current], moved, *diagram[current + 1 :])
        elif op == "move_cause" and isinstance(diagram, Fishbone):
            changed = move_cause(diagram, str(edit.get("cause")), str(edit.get("category")))
        elif op == "version" and isinstance(diagram, tuple):
            index = edit.get("index")
            if isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(diagram):
                changed, current = diagram, index
        if changed is None:
            return None
        page = diagram_page(kind, stored.summary.topic, changed, current)
        self.map_history.replace_map(map_id, page)
        return page

    def regroup_diagram(
        self, map_id: int, grouping: str, custom: Sequence[str] = ()
    ) -> dict[str, Any] | None:
        """Group a fishbone's causes again (model, 6M or the user's list); the notes are not read
        again (ADR-036). None if the map is not a fishbone."""
        stored = self.map_history.get(map_id)
        if stored is None or stored.summary.kind != "fishbone":
            return None
        _kind, diagram, _current = diagram_from_page(stored.map)
        assert isinstance(diagram, Fishbone)
        self._claim_generation()
        try:
            with self.lock:
                self._release_reranker()  # ADR-014
            llm = self._llm_factory(self.config)
            try:
                with self._llm_lock:
                    grouped = group_causes(llm, self.config.diagrams, diagram, grouping, custom)
            finally:
                llm.close()
        finally:
            self._generating.release()
        page = diagram_page("fishbone", stored.summary.topic, grouped)
        self.map_history.replace_map(map_id, page)
        return page

    # --- helpers ----------------------------------------------------------------------------------

    # --- the built-in reader (ADR-031) ------------------------------------------------------------

    def _note_index(self) -> tuple[set[str], dict[str, str]]:
        """Indexed note paths, and file name (lower case) → path for `[[wikilinks]]`."""
        with self.lock:
            paths = [
                str(p) for (p,) in self._conn.execute("SELECT path FROM documents ORDER BY path")
            ]
        by_name: dict[str, str] = {}
        for p in sorted(paths, key=lambda p: (p.count("/"), p)):  # the shallowest wins a clash
            by_name.setdefault(p.rsplit("/", 1)[-1].lower(), p)
        return set(paths), by_name

    # --- «Diccionario» (ADR-034) ------------------------------------------------------------

    def _book_fragments(self, book: Book) -> tuple[list[Fragment], int]:
        """A book's content sections, chapter after chapter, numbered from 1; when it has more than
        `guide_max_fragments` they are taken evenly across the whole book (ADR-063), so the
        dictionary of a book is of the book and not of its first chapters. Also how many it has."""
        cfg = self.config.dictionary
        sections: list[tuple[str, tuple[str, ...], str, int, int]] = []
        for chapter in book.chapters:
            with self.lock:
                rows = self._conn.execute(
                    "SELECT c.heading_path, c.text, c.start_line, c.end_line FROM chunks c "
                    "JOIN documents d ON d.id = c.document_id WHERE d.path = ? "
                    "ORDER BY c.start_line",
                    (chapter.path,),
                ).fetchall()
            for heading_json, text, start, end in rows:
                heading = tuple(json.loads(heading_json))
                if len(heading) >= 2 and is_content_heading(heading[-1]):
                    sections.append((chapter.path, heading, text, start, end))
        cap = cfg.guide_max_fragments
        if len(sections) > cap:
            picks = sorted({round(i * (len(sections) - 1) / (cap - 1)) for i in range(cap)})
            chosen = [sections[i] for i in picks]
        else:
            chosen = sections
        fragments = [
            Fragment(n, path, heading, text[: cfg.max_fragment_chars], start, end)
            for n, (path, heading, text, start, end) in enumerate(chosen, start=1)
        ]
        return fragments, len(sections)

    def _guide_fragments(self, doc_path: str) -> tuple[list[Fragment], int]:
        """A guide's content sections in order, numbered from 1 (navigation sections skipped),
        at most `guide_max_fragments`; and how many content sections it has in all."""
        cfg = self.config.dictionary
        with self.lock:
            rows = self._conn.execute(
                "SELECT c.heading_path, c.text, c.start_line, c.end_line FROM chunks c "
                "JOIN documents d ON d.id = c.document_id WHERE d.path = ? ORDER BY c.start_line",
                (doc_path,),
            ).fetchall()
        fragments: list[Fragment] = []
        total = 0
        for heading_json, text, start, end in rows:
            heading = tuple(json.loads(heading_json))
            if len(heading) < 2 or not is_content_heading(heading[-1]):
                continue
            total += 1
            if len(fragments) == cfg.guide_max_fragments:
                continue
            fragments.append(
                Fragment(
                    len(fragments) + 1,
                    doc_path,
                    heading,
                    text[: cfg.max_fragment_chars],
                    start,
                    end,
                )
            )
        return fragments, total

    # --- «Gap Spotter» (ADR-047) -----------------------------------------------------------------

    def spot_gaps(self, topic: str) -> Generator[Event, None, None]:
        """Events: ("start", {fragments}), ("step", {stage, done, total, coverage}) as the run goes,
        then ("done", {id, coverage, guides}). Raises ValueError before any event when the notes
        have nothing on the topic."""
        self._claim_generation()
        try:
            cfg = self.config.spotter
            if cfg.general:  # the topic's own sections are read for what they leave unexplained
                hits = self._general_hits(topic, self._base().search, cfg.sections * 2)
            else:
                with self.lock:
                    hits = self._base().search(topic, cfg.sections * 2)
            fragments = numbered_fragments(
                hits, cfg.sections, cfg.max_fragment_chars, openings=cfg.general
            )
            if not fragments:
                raise ValueError(
                    "No encontré secciones sobre ese tema en tus notas: no hay nada que completar."
                )
            yield (
                "start",
                {"topic": topic, "fragments": [fragment_json(f, self) for f in fragments]},
            )
            with self.lock:
                self._release_reranker()  # the GPU is the LLM's now (ADR-014)

            def search(name: str) -> list[Fragment]:
                with self.lock:
                    found = self._base().search(name, cfg.judge_hits * 2)
                return numbered_fragments(found, cfg.judge_hits, cfg.max_fragment_chars)

            llm = self._llm_factory(self.config)
            coverage: tuple[Coverage, ...] = ()
            cards: tuple[GuideCard, ...] = ()
            try:
                turns = lambda: self._llm_lock  # noqa: E731  (generation and grading take turns)
                for step in spot_gaps(llm, cfg, topic, fragments, search, turns):
                    coverage, cards = step.coverage, step.guides
                    yield (
                        "step",
                        {
                            "stage": step.stage,
                            "done": step.done,
                            "total": step.total,
                            "coverage": [_coverage_json(c, fragments, self) for c in coverage],
                        },
                    )
            finally:
                llm.close()
            result = {
                "coverage": [_coverage_json(c, fragments, self) for c in coverage],
                "guides": _guides_json(cards, coverage, fragments, self),
                "fragments": [fragment_json(f, self) for f in fragments],
            }
            run_id = self.practice.add_spotter_run(topic, result)
            yield "done", {"id": run_id, **result}
        finally:
            self._generating.release()

    def spotter_runs(self) -> list[dict[str, Any]]:
        return self.practice.spotter_runs(self.config.spotter.history_limit)

    def spotter_run(self, run_id: int) -> dict[str, Any] | None:
        return self.practice.spotter_run(run_id)

    def delete_spotter_run(self, run_id: int) -> bool:
        return self.practice.delete_spotter_run(run_id)

    def note_spotter_guide(self, run_id: int, index: int) -> dict[str, Any] | None:
        """«Apuntar en Temas pendientes»: a suggested guide becomes a pending topic in Guías."""
        run = self.practice.spotter_run(run_id)
        if run is None or not 0 <= index < len(run["guides"]):
            raise LookupError("unknown run or guide")
        guide = run["guides"][index]
        detail = " · ".join(g["name"] for g in guide["gaps"])
        return self._record_gap(guide["title"], "spotter", run["topic"], detail)

    def build_dictionary(
        self, kind: str, value: str, within: tuple[str, str] | None = None
    ) -> Generator[Event, None, None]:
        """Events: ("start", {fragments}), ("entries", …) after each fragment with the entries so
        far, then ("done", {id}). Raises ValueError before any event when nothing is found."""
        self._claim_generation()
        try:
            cfg = self.config.dictionary
            where = self.source_scope(*within) if within else None  # ADR-063
            if kind == "book" and not value.endswith(".md"):
                book = next((b for b in self._books() if b.id == value), None)
                if book is None:
                    raise ValueError("Ese libro no está en tu biblioteca indexada.")
                fragments, available = self._book_fragments(book)
                title = book.title
                focus = f"Libro: {title}. Incluye las entradas importantes del fragmento."
            elif kind in ("guide", "chapter", "book"):
                titles = dict(self.practice_guides())
                if value not in titles:
                    raise ValueError("Esa guía no está en tu biblioteca indexada.")
                fragments, available = self._guide_fragments(value)
                title = titles[value]
                focus = f"Guía: {title}. Incluye las entradas importantes del fragmento."
            else:
                if cfg.general:  # the sections that explain the topic first (ADR-062)
                    hits = self._general_hits(value, self._base(where).search, cfg.fragments * 2)
                else:
                    with self.lock:
                        hits = self._base(where).search(value, cfg.fragments * 2)
                fragments = numbered_fragments(
                    hits, cfg.fragments, cfg.max_fragment_chars, openings=cfg.general
                )
                available = len(fragments)
                title = value
                focus = (
                    f"Tema del diccionario: {value}. "
                    "Incluye solo entradas relacionadas con el tema."
                )
            if not fragments:
                raise ValueError("No encontré secciones con contenido para ese diccionario.")
            yield (
                "start",
                {
                    "title": title,
                    "fragments": [fragment_json(f, self) for f in fragments],
                    "available": available,  # a guide longer than the cap is read in part
                },
            )
            with self.lock:
                self._release_reranker()  # the GPU is the LLM's now (ADR-014)
            llm = self._llm_factory(self.config)
            entries: tuple[Entry, ...] = ()
            skipped = 0
            try:
                turns = lambda: self._llm_lock  # noqa: E731  (generation and grading take turns)
                for progress in build_dictionary(llm, cfg, focus, fragments, turns):
                    entries, skipped = progress.entries, progress.skipped
                    yield (
                        "entries",
                        {
                            "processed": progress.processed,
                            "skipped": progress.skipped,
                            "entries": [_entry_json(e) for e in entries],
                        },
                    )
            finally:
                llm.close()
            dictionary_id = self.practice.add_dictionary(
                kind,
                value,
                title,
                [_entry_json(e) for e in entries],
                [_fragment_record(f) for f in fragments],
            )
            yield "done", {"id": dictionary_id, "skipped": skipped, "count": len(entries)}
        finally:
            self._generating.release()

    def dictionary_view(self, dictionary_id: int) -> dict[str, Any] | None:
        stored = self.practice.dictionary(dictionary_id)
        if stored is None:
            return None
        fragments = [_fragment_from_record(r) for r in stored["fragments"]]
        return {**stored, "fragments": [fragment_json(f, self) for f in fragments]}

    def dictionary_review(
        self, dictionary_id: int, index: int, result: str, quiz_id: int | None
    ) -> dict[str, Any] | None:
        """One flashcard answered: recorded as a self-graded question on the entry's section, so
        it counts for the review schedule (ADR-025) and mastery (ADR-026)."""
        stored = self.practice.dictionary(dictionary_id)
        if stored is None or not 0 <= index < len(stored["entries"]):
            return None
        entry = stored["entries"][index]
        by_number = {r["number"]: r for r in stored["fragments"]}
        record = by_number.get(entry["sources"][0]) if entry["sources"] else None
        if record is None:
            return None
        section = Section(
            record["doc_path"],
            tuple(record["heading_path"]),
            record["start_line"],
            record["end_line"],
            record["text"],
        )
        if entry["kind"] == "concept":
            question = f"¿Qué es «{entry['term']}»?"
        elif entry["kind"] == "function":
            question = f"¿Qué hace `{entry['detail']}`?"
        else:
            question = f"¿Qué calcula «{entry['term']}»: ${entry['detail']}$?"
        store = self.practice
        if quiz_id is None or store.quiz_scope(quiz_id) != ("card", str(dictionary_id)):
            quiz_id = store.new_quiz("card", str(dictionary_id), len(stored["entries"]))
        content = QuestionContent(
            question,
            entry["detail"] if entry["kind"] == "concept" else "",
            reference=entry["text"],
        )
        qid = store.add_question(quiz_id, "open_self", section, content)
        store.record_attempt(qid, "", result, "self")
        store.mark_card(dictionary_id, index, result)  # the «sabida / a medias» mark per term
        return {"quiz_id": quiz_id}

    def dictionary_export(self, dictionary_id: int, fmt: str) -> tuple[str, str, str] | None:
        """(file name, content type, text) of a Markdown or Anki export."""
        stored = self.practice.dictionary(dictionary_id)
        if stored is None:
            return None
        entries = [
            Entry(
                e["kind"],
                e["term"],
                e["text"],
                e["detail"],
                tuple(e["sources"]),
                e.get("translation", ""),
            )
            for e in stored["entries"]
        ]
        fragments = [_fragment_from_record(r) for r in stored["fragments"]]
        slug = slugify(stored["title"]) or f"diccionario-{dictionary_id}"
        if fmt == "anki":
            text = to_anki(stored["title"], entries, fragments)
            return f"{slug}.anki.txt", "text/plain; charset=utf-8", text
        today = date.today().isoformat()
        text = dictionary_markdown(stored["title"], entries, fragments, today)
        return f"{slug}.md", "text/markdown; charset=utf-8", text

    def note_list(self) -> list[dict[str, Any]]:
        """Every indexed note for the «Biblioteca» page (ADR-033): path, title, read mark, the
        book it is a chapter of, if any (ADR-053), when its text last changed, and whether it
        has practice and sections due today (ADR-055)."""
        with self.lock:
            rows = self._conn.execute(
                "SELECT path, title, indexed_at FROM documents ORDER BY path"
            ).fetchall()
        reads = self.practice.reads()
        practiced = {doc for doc, _ in self.practice.section_histories()}
        due = {s.key[0] for s in self._today_queue()[0]}
        folder = self.config.library.books_folder
        return [
            {
                "path": p,
                "title": t,
                "read": p in reads,
                "book": book_id(p, folder),
                "changed": changed,
                "practiced": p in practiced,
                "due": p in due,
            }
            for p, t, changed in rows
        ]

    # --- summaries of a chapter, note or book (ADR-056 stage E) -------------------------------

    @property
    def summaries(self) -> SummaryStore:
        if self._summaries is None:
            self._summaries = SummaryStore(self.config.understanding.db)
        return self._summaries

    def _document(self, path: str) -> tuple[str, str, list[Fragment]] | None:
        """(SHA-256, title, content sections in order, numbered from 1) of an indexed note."""
        with self.lock:
            row = self._conn.execute(
                "SELECT id, sha256, title FROM documents WHERE path = ?", (path,)
            ).fetchone()
            if row is None:
                return None
            chunks = self._conn.execute(
                "SELECT heading_path, text, start_line, end_line FROM chunks "
                "WHERE document_id = ? ORDER BY ordinal",
                (row[0],),
            ).fetchall()
        fragments: list[Fragment] = []
        for heading, text, start, end in chunks:
            heading_path = tuple(json.loads(heading))
            if is_content_heading(heading_path[-1]):  # tables of contents are not content
                fragments.append(Fragment(len(fragments) + 1, path, heading_path, text, start, end))
        return row[1], row[2], fragments

    def _summarize(
        self,
        config: Config,
        path: str,
        title: str,
        sha: str,
        fragments: Sequence[Fragment],
        emit: Callable[[Event], None],
        chapter: tuple[int, int] | None = None,
    ) -> Summary | None:
        """One document's summary on `config`'s model, stored; progress as ("progress", …)."""
        if self._counter is None:
            self._counter = self._counter_factory(self.config)
        counter = self._counter
        llm = self._llm_factory(config)
        try:

            def progress(done: int, total: int) -> None:
                data: dict[str, Any] = {"done": done, "total": total}
                if chapter is not None:
                    data["chapter"], data["chapters"] = chapter
                emit(("progress", data))

            with self._llm_lock:
                made = summarize_document(
                    llm, self.config.understanding, title, fragments, counter, progress
                )
        finally:
            llm.close()
        if made is None:
            return None
        overview, points = made
        return self.summaries.put(
            Summary(CHAPTER, path, title, sha, config.llm.model, overview, tuple(points))
        )

    def _stale(self, summaries: Sequence[Summary]) -> set[tuple[str, str]]:
        """The summaries whose text changed since they were made (or is no longer indexed)."""
        with self.lock:
            hashes = dict(self._conn.execute("SELECT path, sha256 FROM documents").fetchall())
        books = {b.id: b for b in self._books()} if any(s.kind == BOOK for s in summaries) else {}
        stale = set()
        for s in summaries:
            if s.kind == CHAPTER:
                current = hashes.get(s.key)
            else:
                found = books.get(s.key)
                shas = [hashes[c.path] for c in found.chapters if c.path in hashes] if found else []
                current = book_hash(shas) if found else None
            if current != s.source_hash:
                stale.add((s.kind, s.key))
        return stale

    def summaries_list(self) -> dict[str, Any]:
        """Every kept summary, for the library's «Resúmenes» tab: what it is, where it sits (its
        book or area), when and with which model, an excerpt, and whether its text changed."""
        kept = self.summaries.all()
        stale = self._stale(kept)
        books = {b.id: b for b in self._books()}
        folder = self.config.library.books_folder
        out = []
        for s in kept:
            book = books.get(s.key if s.kind == BOOK else book_id(s.key, folder) or "")
            out.append(
                {
                    "kind": s.kind,
                    "key": s.key,
                    "title": s.title,
                    "model": s.model,
                    "created": s.created,
                    "overview": s.overview,
                    "points": len(s.points),
                    "chapters": len(s.chapters),
                    "book": {"id": book.id, "title": book.title} if book else None,
                    "area": book.area if book else s.key.split("/")[0] if "/" in s.key else "",
                    "stale": (s.kind, s.key) in stale,
                }
            )
        return {"summaries": out}

    def summary(self, kind: str, key: str) -> dict[str, Any] | None:
        """One kept summary as the reader shows it, without the LLM; None if there is none."""
        found = self.summaries.get(kind, key)
        if found is None:
            return None
        return {**summary_json(found, cached=True), "stale": bool(self._stale([found]))}

    def understand_note(self, path: str) -> Generator[Event, None, None]:
        """A chapter's or note's summary, on the «Razonar» model (the user's choice): the kept one
        when its text has not changed, else ("progress", …) per window and then ("summary", …).
        Raises ValueError (before any event) for a note that is not indexed or has no text."""
        found = self._document(path)
        if found is None:
            raise ValueError("Esa nota no está en el índice.")
        sha, title, fragments = found
        config = replace(self.config, llm=reason_llm(self.config))
        kept = self.summaries.get(CHAPTER, path)
        if kept is not None and kept.source_hash == sha and kept.model == config.llm.model:
            yield "summary", summary_json(kept, cached=True)
            return
        if not fragments:
            raise ValueError("Esta nota no tiene texto que resumir.")
        self._claim_generation()
        try:
            with self.lock:
                self._release_reranker()  # the GPU is the LLM's now (ADR-014)
            made = yield from _threaded(
                lambda emit: self._summarize(config, path, title, sha, fragments, emit)
            )
            if made is None:
                raise ValueError("No pude resumir esta nota; inténtalo otra vez.")
            yield "summary", summary_json(made)
        finally:
            self._generating.release()

    def understand_book(self, book: str) -> Generator[Event, None, None]:
        """A book's overview from its chapters' summaries, all on the tutor's model (the user's
        choice: a first book takes minutes, not the ~40 of the «Razonar» model). Chapters already
        summarized, by either model, are reused when their text has not changed."""
        found = next((b for b in self._books() if b.id == book), None)
        if found is None:
            raise ValueError("No encontré ese libro.")
        documents = {c.path: self._document(c.path) for c in found.chapters}
        chapters = [c for c in found.chapters if documents[c.path] is not None]
        source = book_hash([documents[c.path][0] for c in chapters])  # type: ignore[index]
        kept = self.summaries.get(BOOK, book)
        if kept is not None and kept.source_hash == source:
            yield "summary", summary_json(kept, cached=True)
            return
        self._claim_generation()
        try:
            with self.lock:
                self._release_reranker()

            def work(emit: Callable[[Event], None]) -> Summary:
                lines: list[ChapterLine] = []
                for n, chapter in enumerate(chapters, start=1):
                    sha, _, fragments = documents[chapter.path]  # type: ignore[misc]
                    summary = self.summaries.get(CHAPTER, chapter.path)
                    if (summary is None or summary.source_hash != sha) and fragments:
                        summary = self._summarize(
                            self.config,
                            chapter.path,
                            chapter.title,
                            sha,
                            fragments,
                            emit,
                            (n, len(chapters)),
                        )
                    emit(("progress", {"chapter": n, "chapters": len(chapters)}))
                    if summary is not None and summary.source_hash == sha:
                        lines.append(ChapterLine(chapter.path, chapter.title, summary.overview))
                if not lines:
                    raise ValueError("No pude resumir ningún capítulo de este libro.")
                llm = self._llm_factory(self.config)
                try:
                    with self._llm_lock:
                        overview, invalid = summarize_book(
                            llm, self.config.understanding, found.title, lines
                        )
                finally:
                    llm.close()
                return self.summaries.put(
                    Summary(
                        BOOK,
                        book,
                        found.title,
                        source,
                        self.config.llm.model,
                        overview,
                        chapters=tuple(lines),
                        invalid=invalid,
                    )
                )

            made = yield from _threaded(work)
            yield "summary", summary_json(made)
        finally:
            self._generating.release()

    # --- books (ADR-053) --------------------------------------------------------------------

    def _books(self) -> list[Book]:
        with self.lock:
            rows = self._conn.execute("SELECT path, title FROM documents").fetchall()
        return find_books(rows, self.config.library.books_folder)

    def _in_english(self, book: Book) -> bool:
        """Whether a book is in English, from its first chapter's first fragment; kept."""
        if book.id not in self._english_books and book.chapters:
            with self.lock:
                row = self._conn.execute(
                    "SELECT c.text FROM chunks c JOIN documents d ON d.id = c.document_id "
                    "WHERE d.path = ? ORDER BY c.ordinal LIMIT 1",
                    (book.chapters[0].path,),
                ).fetchone()
            self._english_books[book.id] = bool(row) and in_english(row[0])
        return self._english_books.get(book.id, False)

    def books(self) -> dict[str, Any]:
        """The shelf: every book with its area, chapters read and the next one to read."""
        reads = self.practice.reads()
        out = []
        areas: Counter[str] = Counter()
        for b in self._books():
            read = [c for c in b.chapters if c.path in reads]
            following = next((c for c in b.chapters if c.path not in reads), None)
            areas[b.area] += 1
            out.append(
                {
                    "id": b.id,
                    "title": b.title,
                    "area": b.area,
                    "chapters": len(b.chapters),
                    "read": len(read),
                    "english": self._in_english(b),
                    "next": {"path": following.path, "title": following.title}
                    if following
                    else None,
                }
            )
        return {
            "books": out,
            "areas": [{"name": a, "count": n} for a, n in areas.most_common()],
        }

    def book(self, book: str) -> dict[str, Any] | None:
        """One book's chapters, in order, with their read marks."""
        found = next((b for b in self._books() if b.id == book), None)
        if found is None:
            return None
        reads = self.practice.reads()
        return {
            "id": found.id,
            "title": found.title,
            "area": found.area,
            "english": self._in_english(found),
            "chapters": [
                {"path": c.path, "title": c.title, "read": c.path in reads} for c in found.chapters
            ],
        }

    def _book_context(self, doc_path: str) -> dict[str, Any] | None:
        """Where a chapter sits in its book, for the reader: position, previous and next."""
        key = book_id(doc_path, self.config.library.books_folder)
        if key is None:
            return None
        found = next((b for b in self._books() if b.id == key), None)
        if found is None:
            return None
        paths = [c.path for c in found.chapters]
        if doc_path not in paths:  # the cover
            return {
                "id": found.id,
                "title": found.title,
                "position": 0,
                "total": len(paths),
                "prev": None,
                "next": {"path": paths[0], "title": found.chapters[0].title} if paths else None,
                "chapters": [{"path": c.path, "title": c.title} for c in found.chapters],
            }
        i = paths.index(doc_path)

        def chapter(j: int) -> dict[str, str] | None:
            if 0 <= j < len(paths):
                return {"path": found.chapters[j].path, "title": found.chapters[j].title}
            return None

        return {
            "id": found.id,
            "title": found.title,
            "position": i + 1,
            "total": len(paths),
            "prev": chapter(i - 1),
            "next": chapter(i + 1),
            "chapters": [{"path": c.path, "title": c.title} for c in found.chapters],
        }

    # --- images a note references (ADR-048) --------------------------------------------------

    def note_asset(
        self, doc_path: str, src: str | None = None, name: str | None = None
    ) -> tuple[bytes, str] | None:
        """(bytes, content type) of an image the note references: `src` from Markdown, relative to
        the note, or `name` from an Obsidian embed, found by name. None unless the file is a
        browser image inside the library, small enough."""
        library = self.config.paths.library
        paths, _ = self._note_index()
        if library is None or doc_path not in paths or not (src or name):
            return None
        root = library.resolve()
        folder = PurePosixPath(doc_path).parent
        candidates: list[Path] = []
        if src:
            ref = unquote(src.strip().strip("<>").split("#", 1)[0].split("?", 1)[0])
            if not ref or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", ref):
                return None  # a URL: never fetched (rule 1)
            candidates.append(
                root / ref.lstrip("/") if ref.startswith("/") else root / folder / ref
            )
            name = PurePosixPath(ref).name  # Obsidian also writes vault-wide names here
        if name:
            candidates += [root / folder / name, root / name]
            candidates += [
                root / p for p in self._asset_named(PurePosixPath(name).name, str(folder))
            ]
        limit = self.config.reader.asset_max_mb * 1024 * 1024
        for candidate in candidates:
            try:
                file = candidate.resolve()
            except OSError:
                continue
            if not file.is_relative_to(root) or not file.is_file() or file.stat().st_size > limit:
                continue
            body = file.read_bytes()
            kind = _image_type(file.suffix.lower(), body)
            if kind is not None:
                return body, kind
        return None

    def _asset_named(self, name: str, near: str) -> list[str]:
        """Library files called `name`, the closest to the folder `near` first."""
        key = name.lower()
        now = time.monotonic()
        stale = now - self._asset_walked > self.config.reader.asset_index_seconds
        if self._asset_files is None or (key not in self._asset_files and stale):
            library = self.config.paths.library
            found: dict[str, list[str]] = {}
            if library is not None:
                for dirpath, dirnames, filenames in os.walk(library):
                    dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                    for f in filenames:
                        if not f.lower().endswith(".md"):
                            rel = Path(dirpath, f).relative_to(library).as_posix()
                            found.setdefault(f.lower(), []).append(rel)
            self._asset_files, self._asset_walked = found, now
        options = self._asset_files.get(key, [])

        def shared(path: str) -> int:
            """How many leading folders `path` shares with `near`."""
            count = 0
            for a, b in zip(path.split("/")[:-1], near.split("/"), strict=False):
                if a != b:
                    break
                count += 1
            return count

        return sorted(options, key=lambda p: -shared(p))

    def note_view(self, doc_path: str) -> dict[str, Any] | None:
        """An indexed note as a tree of blocks for the reader; None if it is not indexed."""
        library = self.config.paths.library
        paths, by_name = self._note_index()
        if library is None or doc_path not in paths:
            return None
        file = library / doc_path
        if not file.is_file():
            return None

        def resolve(target: str) -> str | None:
            if target in paths:
                return target
            return by_name.get(target.rsplit("/", 1)[-1].lower())

        text = file.read_bytes().decode("utf-8-sig", errors="replace")
        tree = note_tree(
            text,
            doc_path,
            resolve,
            self.config.reader.outline_depth,
            self.config.reader.callout_headings,
        )
        return {
            "path": doc_path,
            **tree,
            "words_per_minute": self.config.reader.words_per_minute,
            "open_url": self.open_url(doc_path),
            "read": doc_path in self.practice.reads(),
            "book": self._book_context(doc_path),  # a chapter of a book (ADR-053)
        }

    def passage(self, doc_path: str, first: int, last: int, text: str) -> Hit | None:
        """The selected text as a hit, under the heading path of the chunk where it starts."""
        with self.lock:
            row = self._conn.execute(
                "SELECT c.id, c.heading_path FROM chunks c "
                "JOIN documents d ON d.id = c.document_id WHERE d.path = ? AND c.start_line <= ? "
                "ORDER BY c.start_line DESC LIMIT 1",
                (doc_path, max(first, 1)),
            ).fetchone()
            if row is None:  # before the first chunk (a title alone): the first chunk
                row = self._conn.execute(
                    "SELECT c.id, c.heading_path FROM chunks c "
                    "JOIN documents d ON d.id = c.document_id WHERE d.path = ? "
                    "ORDER BY c.start_line LIMIT 1",
                    (doc_path,),
                ).fetchone()
        if row is None:
            return None
        heading = tuple(json.loads(row[1]))
        return Hit(row[0], doc_path, heading, text, first, max(last, first), 0, 0.0, "passage")

    def _section_of(self, passage: Hit) -> Hit | None:
        """The indexed chunk the passage was found in, as context for explaining it (ADR-031);
        called with `lock` held."""
        row = self._conn.execute(
            "SELECT d.path, c.heading_path, c.text, c.start_line, c.end_line "
            "FROM chunks c JOIN documents d ON d.id = c.document_id WHERE c.id = ?",
            (passage.chunk_id,),
        ).fetchone()
        if row is None:
            return None
        path, heading, text, start, end = row
        heading_path = tuple(json.loads(heading))
        return Hit(passage.chunk_id, path, heading_path, text, start, end, 0, 0.0, "section")

    def passage_quiz(self, passage: Hit, kinds: Sequence[str]) -> Generator[Event, None, None]:
        """«Pregúntame sobre esto»: questions written from the passage only, kept as practice of
        its section (ADR-031). Events as `start_quiz`."""
        self._claim_generation()
        try:
            rng = random.Random()
            section = Section(
                passage.doc_path,
                passage.heading_path,
                passage.start_line,
                passage.end_line,
                passage.text,
            )
            if _section_key(section) in self.practice.flagged_sections():
                raise ValueError(
                    "Marcaste una pregunta de esta sección como mal hecha: "
                    "no genero más sobre ella."
                )
            plan = assign_kinds(kinds, self.config.reader.passage_questions, rng)
            store = self.practice
            quiz_id = store.new_quiz("passage", passage.doc_path, len(plan))
            yield "quiz", {"quiz_id": quiz_id, "planned": len(plan), "reused": False}
            with self.lock:
                self._release_reranker()  # the GPU is the LLM's now (ADR-014)
            llm = self._llm_factory(self.config)
            generated = 0
            try:
                for index, kind in enumerate(plan):
                    with self._llm_lock:
                        content = generate_question(llm, self.config, section, kind, rng)
                    if content is None:
                        yield "skipped", {"index": index, "citation": section.citation}
                        continue
                    stored = store.question(store.add_question(quiz_id, kind, section, content))
                    assert stored is not None
                    generated += 1
                    yield "question", {"index": index, **question_json(stored)}
            finally:
                llm.close()
            yield "done", {"generated": generated, "skipped": len(plan) - generated}
        finally:
            self._generating.release()

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
            "llm_control": self.llm_control(llm),
            "bilingual": self.config.retrieval.bilingual,  # the chat also searches in English
        }

    def set_bilingual(self, on: bool) -> None:
        """«Buscar también en inglés» in Ajustes (ADR-052): used from the next question, and kept
        in `config/local.toml` so a restart and the terminal see it too."""
        set_local_value(
            self.config.root / "config" / "local.toml",
            "retrieval",
            "bilingual",
            "true" if on else "false",
        )
        self.config = replace(self.config, retrieval=replace(self.config.retrieval, bilingual=on))

    # --- switching the LLM off and on (ADR-043) ---------------------------------------------------

    def llm_control(self, running: bool | None = None) -> dict[str, bool]:
        """What «Ajustes» can do with the LLM: stop it (or unload its model), or start it."""
        switch = self.llm_switch
        return {
            "available": switch is not None,
            "owned": switch is not None and switch.owned,
            "can_start": switch is not None and switch.can_start(),
            "running": self.llm_running() if running is None else running,
        }

    def stop_llm(self) -> str:
        """ "stopped" or "unloaded" (an Ollama opened outside the app keeps running without the
        model in memory). Refused while something is being generated."""
        if self.llm_switch is None:
            raise ValueError("Esta sesión no controla el modelo de lenguaje.")
        self._claim_generation()
        try:
            return self.llm_switch.stop()
        finally:
            self._generating.release()

    def start_llm(self) -> None:
        if self.llm_switch is None:
            raise ValueError("Esta sesión no controla el modelo de lenguaje.")
        if self.llm_running():
            return
        self._claim_generation()
        try:
            self.llm_switch.start()
        finally:
            self._generating.release()

    def _close_stores(self) -> None:
        if self._practice is not None:
            self._practice.close()
        if self._conversations is not None:
            self._conversations.close()
        if self._map_history is not None:
            self._map_history.close()
        if self._summaries is not None:
            self._summaries.close()
        self._practice = self._conversations = self._map_history = None
        self._summaries = None
        self._gap_store = None  # it shares the practice connection

    # --- «Reset» (ADR-040) --------------------------------------------------------------------

    def memory(self) -> list[MemoryItem]:
        return memory_items(self.config)

    def reset_memory(self) -> ResetResult:
        """Move the app's memory aside; the stores reopen empty on their next use. Refused while
        something is being generated."""
        self._claim_generation()
        try:
            with self.lock:
                self._close_stores()
                return reset_memory(self.config)
        finally:
            self._generating.release()

    def close(self) -> None:
        self._release_reranker()
        self._embedder.close()
        self._close_stores()


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


def _coverage_json(c: Coverage, fragments: Sequence[Fragment], engine: Engine) -> dict[str, Any]:
    """One subtopic: where the candidate came from, the verdict and its evidence (ADR-047)."""
    out: dict[str, Any] = {
        "name": c.candidate.name,
        "about": c.candidate.about,
        "source": c.candidate.source,
        "verdict": c.verdict,
    }
    if c.candidate.source == UNEXPLAINED and c.candidate.mention is not None:
        number, quote = c.candidate.mention
        where = next((f for f in fragments if f.number == number), None)
        if where is not None:
            out["mention"] = {
                **_place(where.doc_path, where.heading_path),
                "quote": quote,
                "citation": where.citation,
                "lines": [where.start_line, where.end_line],
            }
    if c.evidence is not None:
        e = c.evidence
        out["evidence"] = {
            **_place(e.doc_path, e.heading_path),
            "quote": c.quote,
            "citation": e.citation,
            "lines": [e.start_line, e.end_line],
            "open_url": engine.open_url(e.doc_path),
        }
    return out


def _guides_json(
    cards: Sequence[GuideCard],
    coverage: Sequence[Coverage],
    fragments: Sequence[Fragment],
    engine: Engine,
) -> list[dict[str, Any]]:
    gaps = [c for c in coverage if c.verdict != "covered"]
    by_number = {f.number: f for f in fragments}
    out = []
    for card in cards:
        notes = [by_number[n] for n in card.notes if n in by_number]
        out.append(
            {
                "title": card.title,
                "why": card.why,
                "gaps": [
                    {
                        "name": gaps[i].candidate.name,
                        "source": gaps[i].candidate.source,
                        "verdict": gaps[i].verdict,
                        "about": gaps[i].candidate.about,
                    }
                    for i in card.gaps
                ],
                "questions": list(card.questions),
                "notes": [
                    {
                        **_place(f.doc_path, f.heading_path),
                        "citation": f.citation,
                        "lines": [f.start_line, f.end_line],
                    }
                    for f in notes
                ],
            }
        )
    return out


_IMAGE_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".avif": "image/avif",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
}


def _image_type(suffix: str, body: bytes) -> str | None:
    """A browser image's content type: by extension, else by its first bytes (`.bin` exports).
    JPEG 2000 and anything else browsers do not draw: None (ADR-048)."""
    if suffix in _IMAGE_TYPES:
        return _IMAGE_TYPES[suffix]
    if body.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if body.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if body[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if body[:4] == b"RIFF" and body[8:12] == b"WEBP":
        return "image/webp"
    return None


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


_KIND_SLUGS = {"flow": "flujo", "sequence": "secuencia", "fishbone": "causa-efecto"}


def summary_json(summary: Summary, cached: bool = False) -> dict[str, Any]:
    """A summary as the reader shows it (ADR-056 stage E)."""
    return {
        "kind": summary.kind,
        "key": summary.key,
        "title": summary.title,
        "model": summary.model,
        "created": summary.created,
        "cached": cached,
        "overview": summary.overview,
        "points": [asdict(p) for p in summary.points],
        "chapters": [asdict(c) for c in summary.chapters],
        "invalid": list(summary.invalid),
    }


def gate_json(gate: Gate) -> dict[str, Any]:
    """The topic gate of the «Razonar» mode (ADR-056), as the page shows it."""
    return {
        "open": gate.open,
        "subject": gate.subject,
        "subjects": list(gate.subjects),
        "sections": gate.sections,
    }


def checks_json(checks: ReasonChecks) -> dict[str, Any]:
    return {
        "gate": gate_json(checks.gate),
        "layers": list(checks.layers),
        "quotes": checks.quotes,
        "quotes_found": checks.quotes_found,
        "general_citations": checks.general_citations,
    }


def plan_json(plan: Plan, seconds: float) -> dict[str, Any]:
    """How a message was read (ADR-038), shown under the answer and kept with the turn."""
    return {
        "relation": plan.relation,
        "question": plan.question,
        "searches": list(plan.searches),
        "frame": plan.frame,
        "relate": plan.relate,
        "planned": plan.planned,
        "seconds": round(seconds, 1),
    }


def map_json(cmap: ConceptMap, kind: str = "concepts") -> dict[str, Any]:
    """A concept map as the page receives it; "novak" is laid out top-down (ADR-036)."""
    if kind == "novak":
        cmap = novak_layout(cmap)
    return {
        "kind": kind,
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
                "translation": n.translation,
            }
            for n in cmap.nodes
        ],
        "edges": [
            {"source": e.source, "target": e.target, "label": e.label, "kind": e.kind}
            for e in cmap.edges
        ],
        "omitted": cmap.omitted,
    }


def _section_key(section: Section) -> tuple[str, str]:
    """How the practice history names a section (ADR-020)."""
    return section.doc_path, json.dumps(list(section.heading_path), ensure_ascii=False)


def parse_section_keys(value: str) -> list[tuple[str, str]]:
    """A quiz on given sections (a map node's, ADR-026): JSON [[doc_path, heading_path JSON], …]."""
    try:
        raw = json.loads(value)
    except ValueError as exc:
        raise ValueError("Las secciones no tienen un formato válido.") from exc
    if not isinstance(raw, list) or not raw or len(raw) > 50:
        raise ValueError("Las secciones no tienen un formato válido.")
    keys = []
    for item in raw:
        if not (
            isinstance(item, list) and len(item) == 2 and all(isinstance(x, str) for x in item)
        ):
            raise ValueError("Las secciones no tienen un formato válido.")
        keys.append((item[0], item[1]))
    return keys


def _fragment_record(f: Fragment) -> dict[str, Any]:
    """A fragment as kept with a «Explícamelo tú» session (ADR-027)."""
    return {
        "number": f.number,
        "doc_path": f.doc_path,
        "heading_path": list(f.heading_path),
        "start_line": f.start_line,
        "end_line": f.end_line,
        "text": f.text,
    }


def _fragment_from_record(r: dict[str, Any]) -> Fragment:
    heading = tuple(r["heading_path"])
    return Fragment(r["number"], r["doc_path"], heading, r["text"], r["start_line"], r["end_line"])


def _entry_json(e: Entry) -> dict[str, Any]:
    return {
        "kind": e.kind,
        "term": e.term,
        "text": e.text,
        "detail": e.detail,
        "sources": list(e.sources),
        "translation": e.translation,
    }
