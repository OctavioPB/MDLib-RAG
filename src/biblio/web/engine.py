"""Warm models shared by the web UI's requests (ADR-016, ADR-019).

The query embedder stays loaded on CPU. The reranker follows ADR-014 *per request*: it is loaded
on the GPU only while no LLM server listens, and released before every LLM call, so it never shares
the GPU with the LLM. One request at a time uses the models (single user, `lock`).
"""

from __future__ import annotations

import json
import math
import queue
import random
import sqlite3
import threading
import time
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from datetime import date, datetime
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
from biblio.maps.build import CachedEmbed, ConceptMap, generate_map, topic_in_text
from biblio.maps.export import map_fragments, numbered_fragments, to_canvas, to_markdown
from biblio.maps.history import MapStore, concept_map_from_page, fragment_from_page
from biblio.offline import loopback_port_open
from biblio.practice.dictionary import Entry, build_dictionary, to_anki
from biblio.practice.dictionary import to_markdown as dictionary_markdown
from biblio.practice.feynman import (
    Idea,
    check_claims,
    check_coverage,
    extract_ideas,
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
from biblio.study.gaps import GapStore, extract_topic, same_topic
from biblio.study.plan import (
    PlanItem,
    PlanProgress,
    folder_guides,
    guide_states,
    next_guide,
    plan_folder,
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
        self._gap_store: GapStore | None = None
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
        self,
        question: str,
        *,
        scope: str | None = None,
        map_reduce: str = "off",
        passage: Hit | None = None,
        level: str | None = None,
    ) -> Generator[Event, None, None]:
        """UI events: ("meta", …) once, ("token", …) while generating, ("done", …) at the end."""
        with self.lock:
            self._release_reranker()  # before the LLM loads its model on the GPU (ADR-014)
            if self._counter is None:
                self._counter = self._counter_factory(self.config)
            llm = self._llm_factory(self.config)
            try:
                tutor = Tutor(self.config, self._base(scope), llm, self._counter, map_reduce, level)
                t0 = time.perf_counter()
                prepared = (  # a passage of a note is fragment [1] (ADR-031)
                    tutor.prepare_passage(
                        question, passage, passage.text, self._section_of(passage)
                    )
                    if passage is not None
                    else tutor.prepare(question)
                )
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
                with self.lock:
                    retriever = self._base() if kind == "topic" else None
                    candidates = candidate_sections(
                        self._conn, self.config, kind, value, retriever, review
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
        for n in stored.map["nodes"]:
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

    def feynman_start(self, concept: str) -> Generator[Event, None, None]:
        """Step 1, while the user writes: find the notes and take their key ideas. One event,
        ("ready", {session_id, …}); the ideas stay on the server until the explanation arrives.
        Raises ValueError (before any event) when the notes say too little."""
        self._claim_generation()
        try:
            cfg = self.config.feynman
            with self.lock:
                hits = self._base().search(concept, cfg.fragments * 2)
            fragments = numbered_fragments(hits, cfg.fragments, cfg.max_fragment_chars)
            if len(fragments) < cfg.min_fragments:
                raise ValueError(f"Tus notas no tienen suficiente sobre «{concept}».")
            with self.lock:
                self._release_reranker()  # the GPU is the LLM's now (ADR-014)
            llm = self._llm_factory(self.config)
            try:
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
            llm = self._llm_factory(self.config)
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
            finally:
                llm.close()
            claims_json = [
                {"text": c.text, "verdict": c.verdict, "fragment": c.fragment, "quote": c.quote}
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
        self,
        question: str,
        *,
        scope: str | None,
        strategy: str,
        conversation_id: int | None,
        passage: Hit | None = None,
        level: str | None = None,
    ) -> Generator[Event, None, None]:
        """`ask`, stored in the history when the answer is complete: the "done" event carries the
        conversation id. An error is stored too and sent as an ("error", …) event with the id."""
        map_reduce = "on" if strategy == "map-reduce" else "off"  # opt-in only (ADR-017)
        meta: dict[str, Any] = {}
        parts: list[str] = []
        abstained = False
        try:
            level = level or self.config.tutor.default_level  # ADR-032
            asked = self.ask(
                question, scope=scope, map_reduce=map_reduce, passage=passage, level=level
            )
            for name, data in asked:
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
                        payload={**meta, "done": data, "level": level},
                    )
                    cid, _ = self.conversations.add_turn(conversation_id, turn)
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
                self._record_gap(topic, "map", topic, "sin secciones")  # ADR-029
                raise ValueError(
                    "No encontré secciones sobre ese tema en tus notas. "
                    "Lo apunté como tema pendiente en Guías."
                )
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
            concepts = len(progress.map.nodes) - 1  # the topic is a node too
            absent = not any(topic_in_text(topic, f.text) for f in fragments)
            if absent or concepts < self.config.gaps.map_min_concepts:  # thin (ADR-029, 030)
                detail = (
                    "el tema no aparece en tus notas"
                    if absent
                    else f"{concepts} {'concepto' if concepts == 1 else 'conceptos'}"
                )
                gap = self._record_gap(topic, "map", topic, detail)
                if gap is not None:
                    yield "gap", {**gap, "absent": absent}
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

    def build_dictionary(self, kind: str, value: str) -> Generator[Event, None, None]:
        """Events: ("start", {fragments}), ("entries", …) after each fragment with the entries so
        far, then ("done", {id}). Raises ValueError before any event when nothing is found."""
        self._claim_generation()
        try:
            cfg = self.config.dictionary
            if kind == "guide":
                titles = dict(self.practice_guides())
                if value not in titles:
                    raise ValueError("Esa guía no está en tu biblioteca indexada.")
                fragments, available = self._guide_fragments(value)
                title = titles[value]
                focus = f"Guía: {title}. Incluye las entradas importantes del fragmento."
            else:
                with self.lock:
                    hits = self._base().search(value, cfg.fragments * 2)
                fragments = numbered_fragments(hits, cfg.fragments, cfg.max_fragment_chars)
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
            Entry(e["kind"], e["term"], e["text"], e["detail"], tuple(e["sources"]))
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
        """Every indexed note for the «Biblioteca» page (ADR-033): path, title, read mark."""
        with self.lock:
            rows = self._conn.execute("SELECT path, title FROM documents ORDER BY path").fetchall()
        reads = self.practice.reads()
        return [{"path": p, "title": t, "read": p in reads} for p, t in rows]

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
        tree = note_tree(text, doc_path, resolve)
        return {
            "path": doc_path,
            **tree,
            "open_url": self.open_url(doc_path),
            "read": doc_path in self.practice.reads(),
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
    }
