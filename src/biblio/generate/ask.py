"""`biblio ask`: retrieve → assemble numbered evidence → stream the answer → validate citations.

Two strategies (`rag-prompting` skill):
- direct: up to `max_fragments` fragments in one prompt;
- map-reduce: when the evidence spans many guides or does not fit the budget, each guide's fragments
  are summarized for the question keeping their [n] (map), then the notes are synthesized (reduce).
Fragment numbers are global in both, so the final answer's citations always point to real fragments.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from importlib import resources

from biblio.config import TUTOR_LEVELS, Config
from biblio.generate.citations import Validation, is_abstention, validate
from biblio.generate.context import Context, Fragment, assemble
from biblio.generate.llm import LLMClient, Message
from biblio.generate.thread import Plan, PriorTurn, make_plan, render_history, search_plan
from biblio.ingest.tokens import TokenCounter
from biblio.retrieve.types import Hit, Retriever

_PROMPTS = resources.files("biblio.generate") / "prompts"
DIRECT = "direct"
MAP_REDUCE = "map-reduce"


def load_prompt(name: str) -> str:
    return (_PROMPTS / f"{name}.txt").read_text(encoding="utf-8").strip()


@dataclass(frozen=True)
class MapNote:
    guide: str
    text: str  # bullet points citing global fragment numbers


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    context: Context
    validation: Validation
    retriever: str
    strategy: str = DIRECT
    notes: tuple[MapNote, ...] = ()


@dataclass
class PreparedAsk:
    question: str
    context: Context
    messages: tuple[Message, ...]
    retriever: str
    strategy: str = DIRECT
    groups: tuple[tuple[Fragment, ...], ...] = ()  # map-reduce: fragments per guide
    notes: list[MapNote] = field(default_factory=list)  # filled while streaming
    plan: Plan | None = None  # how the message was read and searched (ADR-038)


class Tutor:
    def __init__(
        self,
        config: Config,
        retriever: Retriever,
        llm: LLMClient,
        counter: TokenCounter,
        map_reduce: str | None = None,  # overrides llm.map_reduce: "auto" | "on" | "off"
        level: str | None = None,  # beginner | intermediate | advanced (ADR-032)
    ) -> None:
        self._config = config
        self._retriever = retriever
        self._llm = llm
        self._counter = counter
        self._mode = map_reduce or config.llm.map_reduce
        self.abstention = load_prompt("abstention")
        self.level = level or config.tutor.default_level
        if self.level not in TUTOR_LEVELS:
            raise ValueError(f"unknown tutor level {self.level!r}")
        level_block = load_prompt(f"tutor_level_{self.level}")
        self._answer_tokens = config.tutor.max_tokens(self.level)
        self._system = load_prompt("tutor_system").format(
            abstention=self.abstention, level=level_block
        )
        self._user = load_prompt("tutor_user")
        self._empty = load_prompt("map_empty")
        self._map_system = load_prompt("map_system").format(empty=self._empty)
        self._map_user = load_prompt("map_user")
        self._reduce_system = load_prompt("reduce_system").format(
            abstention=self.abstention, level=level_block
        )
        self._reduce_user = load_prompt("reduce_user")
        self._thread_user = load_prompt("tutor_thread_user")
        self._relate = load_prompt("tutor_relate")

    # --- preparation ----------------------------------------------------------------------------

    def prepare(self, question: str) -> PreparedAsk:
        if self._config.thread.plan:  # a lone question is a conversation's first message (ADR-038)
            return self.prepare_planned(self.plan(question))
        llm = self._config.llm
        final_k = self._config.retrieval.final_k
        depth = final_k if self._mode == "off" else max(final_k, llm.map_reduce_max_fragments)
        hits = self._retriever.search(question, depth)
        direct = self._direct(question, hits[:final_k])
        if not direct.context.fragments or not self._wants_map_reduce(direct.context):
            return direct
        return self._map_reduce_plan(question, hits)

    def plan(self, message: str, frame: str = "", history: Sequence[PriorTurn] = ()) -> Plan:
        """How to read and search a message of a conversation (ADR-038)."""
        return make_plan(self._llm, self._config.thread, message, frame, history)

    def prepare_planned(self, plan: Plan, history: Sequence[PriorTurn] = ()) -> PreparedAsk:
        """The plan's searches, fused and spread across guides; the previous turns, if any, go
        into the prompt as context (never as a source)."""
        llm = self._config.llm
        final_k = self._config.retrieval.final_k
        depth = final_k if self._mode == "off" else max(final_k, llm.map_reduce_max_fragments)
        rrf_k = self._config.retrieval.rrf_k
        hits = search_plan(self._retriever, plan, depth, self._config.thread, rrf_k)
        frame = plan.frame if history else ""
        evidence = self._config.thread.evidence_fragments if plan.searches else llm.max_fragments
        shown = hits[: max(final_k, evidence)]
        prepared = self._direct(plan.question, shown, True, frame, history, plan.relate, evidence)
        if prepared.context.fragments and self._wants_map_reduce(prepared.context):
            prepared = self._map_reduce_plan(plan.question, hits)
        prepared.plan = plan
        prepared.retriever = f"{self._retriever.name}+plan"
        return prepared

    def prepare_passage(
        self, question: str, passage: Hit, query: str, around: Hit | None = None
    ) -> PreparedAsk:
        """A question about a passage of a note (ADR-031): the passage is fragment [1], the section
        it sits in (`around`) [2], then what retrieval finds for `query`; hits overlapping either
        are left out."""
        pinned = [passage] if around is None else [passage, around]

        def overlaps(h: Hit) -> bool:
            return any(
                h.doc_path == p.doc_path
                and h.start_line <= p.end_line
                and p.start_line <= h.end_line
                for p in pinned
            )

        hits = [
            h
            for h in self._retriever.search(query, self._config.retrieval.final_k)
            if not overlaps(h)
        ]
        # pinned first, never merged with each other (the passage is inside its section)
        return self._direct(question, [*pinned, *hits], merge=False)

    def _direct(
        self,
        question: str,
        hits: Sequence[Hit],
        merge: bool = True,
        frame: str = "",
        history: Sequence[PriorTurn] = (),
        relate: bool = False,
        max_fragments: int | None = None,
    ) -> PreparedAsk:
        llm = self._config.llm
        system = f"{self._system}\n\n{self._relate}" if relate else self._system

        def user(evidence: str) -> str:
            if not history:
                return self._user.format(evidence=evidence, question=question)
            return self._thread_user.format(
                frame=frame or "(ninguno)",
                history=render_history(history),
                evidence=evidence,
                question=question,
            )

        # the history counts against the evidence budget: the context window is fixed (ADR-015)
        fixed = self._counter.count(system) + self._counter.count(user(""))
        context = assemble(
            hits,
            self._counter,
            llm.context_budget_tokens,
            max_fragments or llm.max_fragments,
            fixed,
            merge,
        )
        evidence = context.render() or "(ningún fragmento)"
        messages = (Message("system", system), Message("user", user(evidence)))
        return PreparedAsk(question, context, messages, self._retriever.name)

    def _wants_map_reduce(self, context: Context) -> bool:
        if self._mode == "on":
            return True
        if self._mode == "off":
            return False
        guides = {f.doc_path for f in context.fragments}
        return len(guides) >= self._config.llm.map_reduce_min_guides or context.over_budget > 0

    def _map_reduce_plan(self, question: str, hits: Sequence[Hit]) -> PreparedAsk:
        llm = self._config.llm
        # Global numbering over all the evidence; each map call then gets one guide's share,
        # trimmed to the budget fragment by fragment (never cut inside a fragment).
        everything = assemble(hits, self._counter, sys.maxsize, llm.map_reduce_max_fragments, 0)
        by_guide: dict[str, list[Fragment]] = {}
        for fragment in everything.fragments:
            by_guide.setdefault(fragment.doc_path, []).append(fragment)
        groups = []
        for fragments in by_guide.values():
            fixed = self._counter.count(self._map_system) + self._counter.count(
                self._map_user.format(question=question, guide=fragments[0].citation, evidence="")
            )
            kept, used = [], fixed
            for fragment in fragments:
                cost = self._counter.count(fragment.render()) + 2
                if used + cost <= llm.context_budget_tokens:
                    kept.append(fragment)
                    used += cost
            if kept:
                groups.append(tuple(kept))
        return PreparedAsk(
            question, everything, (), self._retriever.name, MAP_REDUCE, tuple(groups)
        )

    # --- generation -----------------------------------------------------------------------------

    def _complete(self, system: str, user: str) -> str:
        messages = (Message("system", system), Message("user", user))
        return "".join(self._llm.stream(messages, self._config.llm.max_answer_tokens)).strip()

    def stream(self, prepared: PreparedAsk) -> Iterator[str]:
        if not prepared.context.fragments:
            # Nothing retrieved: abstain without spending a generation on it.
            yield self.abstention
            return
        if prepared.strategy == DIRECT:
            yield from self._llm.stream(prepared.messages, self._answer_tokens)
            return
        for group in prepared.groups:
            guide = f"{group[0].heading_path[0]} ({group[0].doc_path})"
            evidence = "\n\n".join(f.render() for f in group)
            text = self._complete(
                self._map_system,
                self._map_user.format(question=prepared.question, guide=guide, evidence=evidence),
            )
            if text and not is_abstention(text, self._empty):
                prepared.notes.append(MapNote(guide, text))
        if not prepared.notes:
            yield self.abstention
            return
        notes = "\n\n".join(f"Guía «{n.guide}»:\n{n.text}" for n in prepared.notes)
        messages = (
            Message("system", self._reduce_system),
            Message("user", self._reduce_user.format(notes=notes, question=prepared.question)),
        )
        yield from self._llm.stream(messages, self._answer_tokens)

    def finish(self, prepared: PreparedAsk, text: str) -> Answer:
        validation = validate(text, len(prepared.context.fragments), self.abstention)
        return Answer(
            prepared.question,
            text,
            prepared.context,
            validation,
            prepared.retriever,
            prepared.strategy,
            tuple(prepared.notes),
        )

    def ask(self, question: str, on_token: Callable[[str], None] | None = None) -> Answer:
        prepared = self.prepare(question)
        parts: list[str] = []
        for token in self.stream(prepared):
            parts.append(token)
            if on_token:
                on_token(token)
        return self.finish(prepared, "".join(parts).strip())
