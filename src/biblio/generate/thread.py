"""Conversations that keep their topic, and questions that reach every area they join (ADR-038).

Before searching, one short LLM call plans the message from the conversation's topic and its last
turns: whether it follows, corrects or leaves the topic, the message as a standalone question, and
a few searches (one per area the question joins, plus one joining them). The searches go through
the usual first stage; the question's own best hits keep their places and the plan's searches fill
two more, so that a question joining areas reaches the other one (7 of 8 fragments from one
chemistry guide once hid every note about data science applied to chemistry).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from importlib import resources

from biblio.config import ThreadConfig
from biblio.generate.llm import LLMClient, Message
from biblio.retrieve.fusion import rrf
from biblio.retrieve.types import Hit, Retriever

RELATIONS = ("inicio", "sigue", "corrige", "nuevo")
_MAX_QUERY = 120
_CITATION = re.compile(r"\s*\[\d+\](\[\d+\])*")
_PROMPTS = resources.files("biblio.generate") / "prompts"


def _json_object(text: str) -> dict[str, object] | None:
    """The JSON object in a reply (the models wrap it in prose or fences now and then)."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def load_prompt(name: str) -> str:  # as `ask.load_prompt`, which imports this module
    return (_PROMPTS / f"{name}.txt").read_text(encoding="utf-8").strip()


@dataclass(frozen=True)
class PriorTurn:
    question: str  # as the user wrote it
    gist: str  # the first sentences of the answer, without citations


@dataclass(frozen=True)
class Plan:
    relation: str  # inicio | sigue | corrige | nuevo
    question: str  # standalone
    searches: tuple[str, ...]  # besides the standalone question
    frame: str  # the conversation's topic after this message
    relate: bool  # the question relates two or more areas
    planned: bool = True  # False: the reply was unusable, the message is searched as written


def gist(answer: str, limit: int) -> str:
    """The first sentences of an answer, without its citations, up to `limit` characters."""
    text = " ".join(_CITATION.sub("", answer).split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return cut[: end + 1] if end > limit // 3 else cut.rstrip() + "…"


def render_history(turns: Sequence[PriorTurn]) -> str:
    if not turns:
        return "(ninguno)"
    lines = []
    for t in turns:
        lines.append(f"Usuario: {t.question}")
        if t.gist:
            lines.append(f"Tutor (resumen): {t.gist}")
    return "\n".join(lines)


def _clean(value: object, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split()).strip(" «»\"'")
    return text[:limit] if text else None


def parse_plan(reply: str, message: str, frame: str, cfg: ThreadConfig, first: bool) -> Plan | None:
    """The model's plan, validated; None if unusable."""
    data = _json_object(reply)
    if data is None:
        return None
    question = _clean(data.get("pregunta"), 400)
    raw = data.get("busquedas")
    if question is None or not isinstance(raw, list):
        return None
    searches: list[str] = []
    for item in raw:
        query = _clean(item, _MAX_QUERY)
        if query and query.casefold() not in {s.casefold() for s in [question, *searches]}:
            searches.append(query)
        if len(searches) == cfg.max_searches:
            break
    relation = str(data.get("relacion", "")).strip().lower()
    if first:
        relation = "inicio"
        # a first message needs no context: rewriting it once cut a question in half and once
        # added a topic it did not mention (measured on eval/questions.jsonl, ADR-038)
        question = message
    elif relation not in RELATIONS[1:]:
        relation = "sigue"
    new_frame = _clean(data.get("tema"), cfg.frame_max_chars) or frame or question
    if relation == "nuevo" and frame:
        new_frame = frame  # answered here only if the user insists; the topic stays
    return Plan(relation, question, tuple(searches), new_frame, data.get("relacionar") is True)


def as_written(message: str, frame: str) -> Plan:
    """The fallback: the message searched as written, the topic unchanged."""
    return Plan("sigue", message, (), frame or message, False, planned=False)


def make_plan(
    llm: LLMClient,
    cfg: ThreadConfig,
    message: str,
    frame: str = "",
    history: Sequence[PriorTurn] = (),
) -> Plan:
    """One LLM call, asked once more if the reply is unusable, then the message as written."""
    first = not history
    user = load_prompt("thread_plan_user").format(
        frame=frame or "(todavía ninguno: es el primer mensaje)",
        history=render_history(history),
        message=message,
    )
    messages = (Message("system", load_prompt("thread_plan_system")), Message("user", user))
    for _ in range(2):
        reply = "".join(llm.stream(messages, cfg.plan_max_tokens))
        plan = parse_plan(reply, message, frame, cfg, first)
        if plan is not None:
            return plan
    return as_written(message, frame)


def reserve_and_fill(own: Sequence[Hit], fused: Sequence[Hit], head: int, fill: int) -> list[Hit]:
    """The question's own best `head` hits, then the `fill` best hits of the fused ranking that
    the question missed, then everything else in order. The tutor reads `head + fill` fragments
    when there is a plan (`thread.evidence_fragments`), so nothing it read before is lost.

    Measured on eval/questions.jsonl (ADR-038): fusing the plan's searches with equal weight, or
    capping fragments per guide, dropped synthesis recall@8 from 0.54 to 0.43-0.48, because a lone
    question is its own best search; slotting two fills inside the six still cost one item."""
    taken = list(own[:head])
    ids = {h.chunk_id for h in taken}
    extra = [h for h in fused if h.chunk_id not in ids][:fill]
    taken += extra
    ids |= {h.chunk_id for h in extra}
    seen: set[int] = set()
    rest = []
    for hit in [*own, *fused]:
        if hit.chunk_id not in ids and hit.chunk_id not in seen:
            seen.add(hit.chunk_id)
            rest.append(hit)
    return taken + rest


def search_plan(
    retriever: Retriever,
    plan: Plan,
    k: int,
    cfg: ThreadConfig,
    rrf_k: int,
) -> list[Hit]:
    """The standalone question first; the plan's searches only fill places it leaves (ADR-038)."""
    depth = max(k, cfg.search_depth)
    own = retriever.search(plan.question, depth)
    if not plan.searches:
        return own[:k]
    rankings = [own, *(retriever.search(q, depth) for q in plan.searches)]
    ranked = reserve_and_fill(own, rrf(rankings, rrf_k), cfg.question_head, cfg.fill)
    return [replace(h, rank=i) for i, h in enumerate(ranked[:k], start=1)]


class PlannedRetriever:
    """A retriever that plans each query first, as a first message of a conversation (ADR-038):
    what `biblio eval --plan` measures, and what the CLI's `ask` uses."""

    def __init__(
        self,
        base: Retriever,
        llm: LLMClient,
        cfg: ThreadConfig,
        rrf_k: int,
    ) -> None:
        self._base = base
        self._llm = llm
        self._cfg = cfg
        self._rrf_k = rrf_k
        self.name = f"{base.name}+plan"
        self.last_plan: Plan | None = None

    def search(self, query: str, k: int) -> list[Hit]:
        self.last_plan = make_plan(self._llm, self._cfg, query)
        return search_plan(self._base, self.last_plan, k, self._cfg, self._rrf_k)

    def close(self) -> None:
        self._base.close()
        self._llm.close()
