"""The «Razonar» mode (ADR-056): answers in marked layers, the model's own knowledge gated by topic.

1. The LLM names the question's field a few ways, in Spanish and English («optimización de
   carteras», «teoría de portafolios», «portfolio optimization»…).
2. Topic gate: each name's definitional question is searched and its candidates reranked on CPU
   (ADR-051 stage 1); the field is explained in the library when `gate_min` sections score
   `gate_score` or more for one of the names. Several names, because the notes' own words decide
   the score («optimización de carteras» 0.06, «teoría de portafolios» 0.82). Measured by hand on
   21 covered fields and 22 absent ones: covered ≥ 0.76 at the best section, absent ≤ 0.15.
3. A closed gate is the fixed abstention: nothing is searched or generated.
4. An open gate: the tutor's evidence (the planned search of ADR-038), then one answer with the
   layers «Según tus notas» (cited), «Razonamiento» (steps that open with a quote of their
   fragment, checked with `practice.feynman.contains`), «Conocimiento general (no está en tus
   notas)» (only for what the fragments lack; never cited) and «Respuesta».
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

from biblio.config import Config
from biblio.generate.ask import Answer, PreparedAsk, Tutor, load_prompt
from biblio.generate.citations import extract_citations, validate
from biblio.generate.context import Context
from biblio.generate.llm import LLMClient, Message
from biblio.maps.select import definitional_query, map_candidates
from biblio.practice.feynman import contains
from biblio.practice.quiz import _json_object
from biblio.retrieve.rerank import passage
from biblio.retrieve.types import Retriever

REASON = "reason"
Score = Callable[[str, Sequence[str]], list[float]]

NOTES, STEPS, GENERAL, ANSWER = "notes", "steps", "general", "answer"
_LAYER_TITLES = {
    NOTES: "según tus notas",
    STEPS: "razonamiento",
    GENERAL: "conocimiento general",
    ANSWER: "respuesta",
}
_PARENS_RE = re.compile(r"\([^)]*\)")
# «quote» [n]: a step's premise, as the prompt asks for it
_QUOTE_RE = re.compile(r"«([^»]+)»\s*\[(\d+)\]")
_MAX_NAME = 60


@dataclass(frozen=True)
class Gate:
    subjects: tuple[str, ...]  # the field's names the LLM proposed
    open: bool  # the library explains one of them: the model's own knowledge is allowed
    subject: str = ""  # the name that opened it
    sections: int = 0  # sections at `gate_score` or more for it (or the best name's count)


@dataclass(frozen=True)
class ReasonChecks:
    gate: Gate
    layers: tuple[str, ...]  # the layers found in the answer, in order
    quotes: int  # premises quoted in «Razonamiento»
    quotes_found: int  # of those, written in the fragment they cite
    general_citations: int  # citation numbers inside «Conocimiento general» (must be 0)


def _layer_heading(line: str) -> tuple[str, str] | None:
    """(layer, text after the title) when the line opens a layer: «**Según tus notas:** …»,
    «### Razonamiento», «Conocimiento general (no está en tus notas): …». A title needs a mark
    (`#`, `**`) or a colon, so a sentence that starts with «Respuesta» is not one."""
    stripped = line.strip()
    title, colon, rest = stripped.lstrip("#* ").partition(":")
    title = _PARENS_RE.sub("", title).strip(" *").casefold()
    if not (colon or stripped.startswith(("#", "**"))):
        return None
    layer = next((k for k, t in _LAYER_TITLES.items() if title == t), None)
    return (layer, rest.strip(" *")) if layer else None


def split_layers(text: str) -> dict[str, str]:
    """The answer's text under each layer's title; text before the first title is left out."""
    layers: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if heading := _layer_heading(line):
            current, rest = heading
            layers.setdefault(current, []).extend([rest] if rest else [])
        elif current is not None:
            layers[current].append(line)
    return {k: "\n".join(v).strip() for k, v in layers.items()}


def check_answer(text: str, context: Context, gate: Gate) -> ReasonChecks:
    layers = split_layers(text)
    quotes = _QUOTE_RE.findall(layers.get(STEPS, ""))
    fragments = {f.number: f for f in context.fragments}
    found = sum(
        1 for quote, n in quotes if int(n) in fragments and contains(fragments[int(n)].text, quote)
    )
    return ReasonChecks(
        gate=gate,
        layers=tuple(k for k in _LAYER_TITLES if layers.get(k)),
        quotes=len(quotes),
        quotes_found=found,
        general_citations=len(extract_citations(layers.get(GENERAL, ""))),
    )


def parse_subjects(reply: str, limit: int) -> tuple[str, ...]:
    data = _json_object(reply)
    raw = data.get("campos") if data else None
    if not isinstance(raw, list):
        return ()
    names: list[str] = []
    for item in raw:
        if isinstance(item, str):
            name = " ".join(item.split()).strip(" «»\"'.")
            if 0 < len(name) <= _MAX_NAME and name.casefold() not in map(str.casefold, names):
                names.append(name)
    return tuple(names[:limit])


def field_names(llm: LLMClient, config: Config, question: str) -> tuple[str, ...]:
    """The question's field, named a few ways in Spanish and English (step 1)."""
    cfg = config.reason
    user = load_prompt("reason_subject_user").format(question=question, n=cfg.subjects)
    messages = (Message("system", load_prompt("reason_subject_system")), Message("user", user))
    return parse_subjects("".join(llm.stream(messages, cfg.subject_max_tokens)), cfg.subjects)


def topic_gate(
    llm: LLMClient, config: Config, retriever: Retriever, score: Score | None, question: str
) -> Gate:
    """Whether the library explains the question's field (step 2): the first name with
    `gate_min` sections scoring `gate_score` or more opens it. Without a reranker it stays
    closed. Also used by «Explícamelo tú» (ADR-058)."""
    cfg = config.reason
    names = field_names(llm, config, question)
    if score is None or not names:
        return Gate(names, False)
    best = 0
    for name in names:  # stops at the first name the library explains
        hits = map_candidates(retriever.search, name, config.maps, config.retrieval.rrf_k)[
            : cfg.gate_candidates
        ]
        query = definitional_query(name, config.maps)
        scores = score(query, [passage(h) for h in hits]) if hits else []
        count = sum(s >= cfg.gate_score for s in scores)
        if count >= cfg.gate_min:
            return Gate(names, True, name, count)
        best = max(best, count)
    return Gate(names, False, "", best)


class Reasoner:
    """Answers like `Tutor.ask`, with the layers of ADR-056. `score` is the reranker's (on CPU:
    the LLM holds the GPU); without it the gate stays closed and every answer abstains."""

    def __init__(
        self,
        config: Config,
        tutor: Tutor,
        llm: LLMClient,
        retriever: Retriever,
        score: Score | None,
    ) -> None:
        self._config = config
        self._tutor = tutor
        self._llm = llm
        self._retriever = retriever
        self._score = score
        self.abstention = tutor.abstention
        self._system = load_prompt("reason_system")
        self._user = load_prompt("reason_user")

    def gate(self, question: str) -> Gate:
        return topic_gate(self._llm, self._config, self._retriever, self._score, question)

    def stream(self, prepared: PreparedAsk, gate: Gate) -> Iterator[str]:
        """The answer's tokens: the fixed abstention when the gate is closed or nothing was found.
        Asked to abstain in its own words, the model explained in its layers that nothing was
        found (3 of 5 in `reason-v1`) instead of saying the fixed sentence."""
        if not gate.open or not prepared.context.fragments:
            yield self.abstention
            return
        evidence = prepared.context.render()
        user = self._user.format(evidence=evidence, question=prepared.question)
        messages = (Message("system", self._system), Message("user", user))
        yield from self._llm.stream(messages, self._config.reason.max_answer_tokens)

    def finish(self, prepared: PreparedAsk, text: str, gate: Gate) -> Answer:
        context = prepared.context if gate.open else Context((), 0, 0)
        return Answer(
            prepared.question,
            text,
            context,
            validate(text, len(context.fragments), self.abstention),
            prepared.retriever,
            REASON,
            reason=check_answer(text, context, gate),
        )

    def ask(self, question: str, on_token: Callable[[str], None] | None = None) -> Answer:
        gate = self.gate(question)
        # a closed gate: the library does not explain the field, nothing is searched (the
        # user's rule is to abstain)
        prepared = (
            self._tutor.prepare(question)
            if gate.open
            else PreparedAsk(question, Context((), 0, 0), (), self._retriever.name)
        )
        parts: list[str] = []
        for token in self.stream(prepared, gate):
            parts.append(token)
            if on_token:
                on_token(token)
        return self.finish(prepared, "".join(parts).strip(), gate)
