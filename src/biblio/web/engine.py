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

from biblio.config import Config
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
from biblio.index.embedder import Embedder
from biblio.ingest.tokens import TokenCounter, load_llm_token_counter
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


def _streamed(
    work: Callable[[Callable[[str], None]], T], event: Callable[[str], Event]
) -> Generator[Event, None, T]:
    """Run `work(on_token)` in a thread and yield each token as an event; return its result.

    If the consumer stops (the browser closed the stream), the next token raises inside `work`
    so the LLM call is abandoned instead of running on unseen.
    """
    tokens: queue.Queue[tuple[str, str]] = queue.Queue()
    outcome: dict[str, Any] = {}
    cancelled = threading.Event()

    def on_token(text: str) -> None:
        if cancelled.is_set():
            raise _Cancelled
        tokens.put(("token", text))

    def run() -> None:
        try:
            outcome["value"] = work(on_token)
        except BaseException as exc:  # handed to the consumer below
            outcome["error"] = exc
        finally:
            tokens.put(("end", ""))

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        while True:
            kind, text = tokens.get()
            if kind == "end":
                break
            yield event(text)
    finally:
        cancelled.set()
    worker.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]  # type: ignore[no-any-return]


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
    ) -> None:
        self.config = config
        self.lock = threading.Lock()
        self._conn = conn
        self._embedder = embedder
        self._reranker_factory = reranker_factory
        self._plan_resolver = plan_resolver
        self._llm_factory = llm_factory
        self._counter_factory = counter_factory
        self._reranker: Reranker | None = None
        self._reranker_device: str | None = None
        self._counter: TokenCounter | None = None
        self._generating = threading.Lock()  # one draft or quiz generation at a time
        self._llm_lock = threading.Lock()  # quiz generation and answer grading take turns
        self._practice: PracticeStore | None = None

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
