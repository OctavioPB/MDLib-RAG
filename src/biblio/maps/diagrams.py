"""Flowcharts, sequence diagrams and cause-effect diagrams (ADR-036).

As for concept maps (ADR-022), the model reads one fragment at a time and code keeps only what the
fragment says: every step, message and cause carries a quote that must be written in it (accents,
case and punctuation ignored). Each note's process or interaction is a version of its own
(ADR-037); causes are merged across notes, and one short LLM call that never sees the notes groups
them into categories. The drawing is laid out in `biblio.maps.drawing`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from typing import TypeVar

from biblio.config import DiagramsConfig
from biblio.generate.ask import load_prompt
from biblio.generate.context import Fragment
from biblio.generate.llm import LLMClient
from biblio.maps.build import fold
from biblio.practice.feynman import contains
from biblio.practice.quiz import _complete, _json_object

KINDS = ("concepts", "novak", "flow", "sequence", "fishbone")
DIAGRAM_KINDS = ("flow", "sequence", "fishbone")
GROUPINGS = ("model", "6m", "custom")
_SPANISH_KINDS = {
    "conceptos": "concepts",
    "mapa conceptual": "novak",
    "mapa_conceptual": "novak",
    "flujo": "flow",
    "secuencia": "sequence",
    "causa efecto": "fishbone",
    "causa_efecto": "fishbone",
}
_MAX_LABEL = 80
_MAX_LABEL_WORDS = 12
_MAX_QUOTE = 400
_MIN_QUOTE_WORDS = 4  # «Preciso pero lento» once became a step: a quote must say an action
_MAX_CATEGORY = 40
OTHERS = "Otras"
END = "end"

T = TypeVar("T")


def _clean(value: object, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split()).strip(" «»\"'")
    return text if 0 < len(text) <= limit else None


def _label(value: object) -> str | None:
    text = _clean(value, _MAX_LABEL)
    if text is None or len(text.split()) > _MAX_LABEL_WORDS:
        return None
    return text.rstrip(".;:,")


def _quote(value: object, fragment: Fragment) -> str | None:
    """The quote, if it is written in the fragment (ADR-036, as ADR-027)."""
    text = _clean(value, _MAX_QUOTE)
    if text is None or len(text.split()) < _MIN_QUOTE_WORDS:
        return None
    return text if contains(fragment.text, text) else None


def _position(fragment: Fragment, quote: str) -> int:
    """Where the quote starts in the fragment, to tell whether two steps are written in order."""
    return f" {fold(fragment.text)} ".find(f" {fold(quote)} ")


def _number(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _words(text: str) -> set[str]:
    return {w for w in fold(text).split() if len(w) > 3}


def same_label(a: str, b: str) -> bool:
    """The same step or cause written twice: same words, or most content words in common."""
    if fold(a) == fold(b):
        return True
    wa, wb = _words(a), _words(b)
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= 0.6


def named_in(text: str, name: str) -> bool:
    key = fold(name)
    return bool(key) and f" {key} " in f" {fold(text)} "


def _ask(
    llm: LLMClient, system: str, user: str, max_tokens: int, parse: Callable[[str], T | None]
) -> T | None:
    """One retry, then None (the ADR-020 pattern)."""
    for _ in range(2):
        result = parse(_complete(llm, system, user, max_tokens))
        if result is not None:
            return result
    return None


def kind_from_spanish(value: str) -> str | None:
    key = " ".join(fold(value).replace("_", " ").split())
    return _SPANISH_KINDS.get(key) or (value if value in KINDS else None)


# --- flowcharts ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    id: str
    label: str
    decision: bool
    quote: str
    sources: tuple[int, ...]  # Fragment.number: the first is where the quote is
    position: int  # where the quote starts in its fragment (folded text)
    yes: str | None = None  # a decision's «sí»: a step id, END, or None for the next step
    no: str | None = None


@dataclass(frozen=True)
class Flow:
    """One fragment's process (ADR-037): processes of different notes are not mixed."""

    source: int  # the fragment it was read from
    steps: tuple[Step, ...]


@dataclass(frozen=True)
class FlowEdge:
    source: str  # a step id, "start" or END
    target: str
    label: str  # "sí", "no" or ""
    inferred: bool  # the order is not written in the notes: drawn dashed


def _branch(value: object, numbers: dict[int, str], own: str) -> str | None:
    if isinstance(value, str) and fold(value) == "fin":
        return END
    n = _number(value)
    target = numbers.get(n) if n is not None else None
    return target if target is not None and target != own else None


def parse_steps(reply: str, fragment: Fragment, limit: int) -> list[Step] | None:
    """The process the model read in one fragment, each step checked against it; [] when the
    fragment describes none; None if the reply is unusable."""
    data = _json_object(reply)
    raw = data.get("pasos") if data is not None else None
    if not isinstance(raw, list):
        return None
    kept: list[tuple[dict[str, object], str, str, str]] = []
    numbers: dict[int, str] = {}
    for i, item in enumerate(raw[: limit * 2], start=1):
        if not isinstance(item, dict):
            continue
        label, quote = _label(item.get("paso")), _quote(item.get("cita"), fragment)
        if label is None or quote is None:
            continue
        step_id = f"f{fragment.number}s{len(kept) + 1}"
        numbers.setdefault(_number(item.get("n")) or i, step_id)
        kept.append((item, label, quote, step_id))
        if len(kept) == limit:
            break
    steps = []
    for j, (item, label, quote, step_id) in enumerate(kept):
        following = kept[j + 1][3] if j + 1 < len(kept) else END
        decision = fold(str(item.get("tipo", ""))) == "decision"
        yes = _branch(item.get("si"), numbers, step_id) if decision else None
        no = _branch(item.get("no"), numbers, step_id) if decision else None
        if decision and (yes or following) == (no or following):
            # both answers lead to the same place (once both «fin»): a question, not a fork
            decision, yes, no = False, None, None
        if decision:
            label = f"¿{label.strip('¿?')}?"
        steps.append(
            Step(
                step_id,
                label,
                decision,
                quote,
                (fragment.number,),
                _position(fragment, quote),
                yes,
                no,
            )
        )
    return steps


def extract_steps(
    llm: LLMClient, cfg: DiagramsConfig, topic: str, fragment: Fragment
) -> list[Step] | None:
    user = load_prompt("flow_user").format(
        topic=topic,
        citation=fragment.citation,
        section=fragment.text[: cfg.max_fragment_chars],
        max_steps=cfg.steps_per_fragment,
    )
    return _ask(
        llm,
        load_prompt("flow_system"),
        user,
        cfg.extract_max_tokens,
        lambda reply: parse_steps(reply, fragment, cfg.steps_per_fragment),
    )


def flow_versions(found: Sequence[Sequence[Step]], min_steps: int = 1) -> tuple[Flow, ...]:
    """Each fragment's process is a version, the one with most steps first (ADR-037); fewer than
    `min_steps` steps are not a process. A step written in another version too adds that
    fragment to its sources."""
    flows = [Flow(steps[0].sources[0], tuple(steps)) for steps in found if len(steps) >= min_steps]
    out = []
    for flow in flows:
        steps = []
        for step in flow.steps:
            also = tuple(
                other.source
                for other in flows
                if other is not flow and any(same_label(s.label, step.label) for s in other.steps)
            )
            steps.append(replace(step, sources=(step.sources[0], *also)))
        out.append(replace(flow, steps=tuple(steps)))
    return tuple(sorted(out, key=lambda f: -len(f.steps)))


def _next_inferred(a: Step, b: Step) -> bool:
    """An arrow is cited only when the second step's quote comes later in the text."""
    return a.sources[0] != b.sources[0] or b.position <= a.position


def flow_edges(flow: Flow) -> list[FlowEdge]:
    steps = flow.steps
    if not steps:
        return []
    ids = {s.id for s in steps}
    edges = [FlowEdge("start", steps[0].id, "", False)]
    for i, step in enumerate(steps):
        nxt = steps[i + 1] if i + 1 < len(steps) else None
        next_id = nxt.id if nxt is not None else END
        to_next = FlowEdge(step.id, next_id, "", nxt is not None and _next_inferred(step, nxt))
        if not step.decision:
            edges.append(to_next)
            continue
        branches = []
        for target, label in ((step.yes, "sí"), (step.no, "no")):
            goal = target if target in ids or target == END else None
            if goal is None or goal == next_id:
                branches.append(replace(to_next, label=label))
            else:
                branches.append(FlowEdge(step.id, goal, label, False))  # read in the fragment
        edges.append(branches[0])
        if branches[1].target != branches[0].target:
            edges.append(branches[1])
    return edges


def move_step(flow: Flow, step_id: str, delta: int) -> Flow | None:
    """The step moved up (-1) or down (+1); None if it cannot move."""
    ids = [s.id for s in flow.steps]
    if step_id not in ids or delta not in (-1, 1):
        return None
    i = ids.index(step_id)
    j = i + delta
    if not 0 <= j < len(ids):
        return None
    steps = list(flow.steps)
    steps[i], steps[j] = steps[j], steps[i]
    return replace(flow, steps=tuple(steps))


# --- sequence diagrams --------------------------------------------------------------------------


@dataclass(frozen=True)
class Message:
    id: str
    sender: str
    receiver: str
    label: str
    reply: bool
    condition: str  # "" or the case in which it happens: consecutive ones form a frame
    quote: str


@dataclass(frozen=True)
class SequenceVersion:
    source: int  # the fragment it was read from
    participants: tuple[str, ...]
    messages: tuple[Message, ...]


def parse_sequence(
    reply: str, fragment: Fragment, max_messages: int, max_participants: int
) -> SequenceVersion | None:
    """Participants named in the fragment and messages between them with a quote from it; an
    empty version when there is no interaction; None if the reply is unusable."""
    data = _json_object(reply)
    if data is None:
        return None
    raw_people, raw_messages = data.get("participantes"), data.get("mensajes")
    if not isinstance(raw_people, list) or not isinstance(raw_messages, list):
        return None
    people: dict[str, str] = {}  # folded → as written
    for raw in raw_people:
        name = _clean(raw, 60)
        if name and named_in(fragment.text, name) and fold(name) not in people:
            people[fold(name)] = name
        if len(people) == max_participants:
            break
    messages: list[Message] = []
    for item in raw_messages[: max_messages * 2]:
        if not isinstance(item, dict):
            continue
        sender = people.get(fold(str(item.get("de", ""))))
        receiver = people.get(fold(str(item.get("a", ""))))
        label, quote = _label(item.get("mensaje")), _quote(item.get("cita"), fragment)
        if sender is None or receiver is None or label is None or quote is None:
            continue
        condition = _label(item.get("condicion")) or ""
        reply_flag = item.get("respuesta") is True
        messages.append(
            Message(
                f"f{fragment.number}m{len(messages) + 1}",
                sender,
                receiver,
                label,
                reply_flag,
                condition,
                quote,
            )
        )
        if len(messages) == max_messages:
            break
    used = {m.sender for m in messages} | {m.receiver for m in messages}
    participants = tuple(p for p in people.values() if p in used)
    return SequenceVersion(fragment.number, participants, tuple(messages))


def extract_sequence(
    llm: LLMClient, cfg: DiagramsConfig, topic: str, fragment: Fragment
) -> SequenceVersion | None:
    user = load_prompt("sequence_user").format(
        topic=topic,
        citation=fragment.citation,
        section=fragment.text[: cfg.max_fragment_chars],
        max_messages=cfg.messages_per_fragment,
    )
    return _ask(
        llm,
        load_prompt("sequence_system"),
        user,
        cfg.extract_max_tokens,
        lambda reply: parse_sequence(
            reply, fragment, cfg.messages_per_fragment, cfg.max_participants
        ),
    )


def order_versions(versions: Sequence[SequenceVersion]) -> tuple[SequenceVersion, ...]:
    """Fragments with an interaction, the one with most messages first (retrieval order on ties)."""
    found = [v for v in versions if v.messages]
    return tuple(sorted(found, key=lambda v: -len(v.messages)))


def frames(messages: Sequence[Message]) -> list[tuple[int, int, str]]:
    """Runs of consecutive messages under one condition: (first index, last index, condition)."""
    runs: list[tuple[int, int, str]] = []
    for i, m in enumerate(messages):
        if not m.condition:
            continue
        if runs and runs[-1][1] == i - 1 and fold(runs[-1][2]) == fold(m.condition):
            runs[-1] = (runs[-1][0], i, runs[-1][2])
        else:
            runs.append((i, i, m.condition))
    return runs


# --- cause-effect diagrams ----------------------------------------------------------------------


@dataclass(frozen=True)
class Cause:
    id: str
    label: str
    quote: str
    sources: tuple[int, ...]  # the first is where the quote is


@dataclass(frozen=True)
class Fishbone:
    effect: str
    causes: tuple[Cause, ...]
    categories: tuple[tuple[str, tuple[str, ...]], ...]  # (name, cause ids), drawn in this order
    grouping: str  # model | 6m | custom | none (not grouped yet, or the model failed twice)
    omitted: int  # causes over the cap


_EFFECT_PREFIX = re.compile(
    r"^(por\s*qu[eé]\s|causas?\s+(de\s+(la|las|los|el)\s+|del\s+|de\s+)?)", re.IGNORECASE
)


def effect_label(topic: str) -> str:
    """«por qué se degrada un modelo» → «Se degrada un modelo»: the head of the fishbone."""
    text = " ".join(topic.split()).strip("¿?¡! ")
    stripped = _EFFECT_PREFIX.sub("", text).strip()
    text = stripped or text
    return text[:1].upper() + text[1:]


def parse_causes(reply: str, fragment: Fragment, limit: int) -> list[Cause] | None:
    data = _json_object(reply)
    raw = data.get("causas") if data is not None else None
    if not isinstance(raw, list):
        return None
    causes: list[Cause] = []
    for item in raw[: limit * 2]:
        if not isinstance(item, dict):
            continue
        label, quote = _label(item.get("causa")), _quote(item.get("cita"), fragment)
        if label is None or quote is None or any(same_label(c.label, label) for c in causes):
            continue
        label = label[:1].upper() + label[1:]
        causes.append(
            Cause(f"f{fragment.number}c{len(causes) + 1}", label, quote, (fragment.number,))
        )
        if len(causes) == limit:
            break
    return causes


def extract_causes(
    llm: LLMClient, cfg: DiagramsConfig, effect: str, fragment: Fragment
) -> list[Cause] | None:
    user = load_prompt("cause_user").format(
        effect=effect,
        citation=fragment.citation,
        section=fragment.text[: cfg.max_fragment_chars],
        max_causes=cfg.causes_per_fragment,
    )
    return _ask(
        llm,
        load_prompt("cause_system"),
        user,
        cfg.extract_max_tokens,
        lambda reply: parse_causes(reply, fragment, cfg.causes_per_fragment),
    )


def merge_causes(
    causes: Sequence[Cause], new: Sequence[Cause], cap: int
) -> tuple[list[Cause], int]:
    """The same cause from several fragments is one cause with all its sources. Returns the
    causes and how many new ones did not fit under the cap."""
    out = list(causes)
    over = 0
    for cause in new:
        match = next((i for i, c in enumerate(out) if same_label(c.label, cause.label)), None)
        if match is not None:
            out[match] = replace(out[match], sources=(*out[match].sources, *cause.sources))
        elif len(out) < cap:
            out.append(cause)
        else:
            over += 1
    return out, over


def ungrouped(effect: str, causes: Sequence[Cause], omitted: int) -> Fishbone:
    """All causes on one bone: while reading, or when grouping failed."""
    group = (("Causas", tuple(c.id for c in causes)),) if causes else ()
    return Fishbone(effect, tuple(causes), group, "none", omitted)


def parse_groups(
    reply: str, count: int, allowed: Sequence[str] | None, max_categories: int
) -> list[tuple[str, list[int]]] | None:
    """Category → cause indices (from 0). With `allowed`, only those names (as written there).
    Every cause ends in one category; the ones the model leaves out go to «Otras»."""
    data = _json_object(reply)
    raw = data.get("categorias") if data is not None else None
    if not isinstance(raw, list):
        return None
    names = {fold(a): a for a in allowed} if allowed is not None else None
    groups: list[tuple[str, list[int]]] = []
    taken: set[int] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = _clean(item.get("nombre"), _MAX_CATEGORY)
        members = item.get("causas")
        if name is None or not isinstance(members, list):
            continue
        if names is not None:
            if fold(name) not in names:
                continue
            name = names[fold(name)]
        elif len(groups) == max_categories or fold(name) == fold(OTHERS):
            continue  # its causes end in «Otras»
        name = name[:1].upper() + name[1:]
        picked = []
        for value in members:
            n = _number(value)
            if n is not None and 1 <= n <= count and n - 1 not in taken:
                taken.add(n - 1)
                picked.append(n - 1)
        existing = next((g for g in groups if fold(g[0]) == fold(name)), None)
        if existing is not None:
            existing[1].extend(picked)
        elif picked:
            groups.append((name, picked))
    if not groups:
        return None
    if allowed is not None:  # in the order of the fixed list
        order = [fold(a) for a in allowed]
        groups.sort(key=lambda g: order.index(fold(g[0])))
    rest = [i for i in range(count) if i not in taken]
    if rest:
        groups.append((OTHERS, rest))
    return groups


def group_causes(
    llm: LLMClient,
    cfg: DiagramsConfig,
    fishbone: Fishbone,
    grouping: str,
    custom: Sequence[str] = (),
) -> Fishbone:
    """The causes in categories: proposed by the model, the 6M or the user's list (ADR-036).
    Reads only the cause names, never the notes."""
    causes = fishbone.causes
    if not causes:
        return fishbone
    allowed: Sequence[str] | None
    if grouping == "6m":
        allowed = cfg.six_m
    elif grouping == "custom":
        allowed = list(dict.fromkeys(c for c in custom if c))
    else:
        allowed = None
    if allowed is None:
        instruction = load_prompt("cause_group_auto").format(
            min=cfg.categories_min, max=cfg.categories_max
        )
    else:
        instruction = load_prompt("cause_group_fixed").format(
            names=", ".join(f"«{a}»" for a in allowed)
        )
    user = load_prompt("cause_group_user").format(
        effect=fishbone.effect,
        causes="\n".join(f"{i}. {c.label}" for i, c in enumerate(causes, start=1)),
        instruction=instruction.strip(),
    )
    groups = _ask(
        llm,
        load_prompt("cause_group_system"),
        user,
        cfg.group_max_tokens,
        lambda reply: parse_groups(reply, len(causes), allowed, cfg.categories_max),
    )
    if groups is None:
        return ungrouped(fishbone.effect, causes, fishbone.omitted)
    categories = tuple((name, tuple(causes[i].id for i in members)) for name, members in groups)
    return replace(fishbone, categories=categories, grouping=grouping)


def move_cause(fishbone: Fishbone, cause_id: str, category: str) -> Fishbone | None:
    """The cause moved by hand to another category; None if either does not exist."""
    names = [name for name, _ in fishbone.categories]
    if category not in names or cause_id not in {c.id for c in fishbone.causes}:
        return None
    categories = []
    for name, ids in fishbone.categories:
        kept = tuple(i for i in ids if i != cause_id)
        if name == category:
            kept = (*kept, cause_id)
        categories.append((name, kept))
    return replace(fishbone, categories=tuple(categories))


# --- the type suggested for a topic ---------------------------------------------------------------


@dataclass(frozen=True)
class Suggestion:
    kind: str
    reason: str


def parse_suggestion(reply: str) -> Suggestion | None:
    data = _json_object(reply)
    if data is None:
        return None
    kind = kind_from_spanish(str(data.get("tipo", "")))
    if kind is None:
        return None
    return Suggestion(kind, _clean(data.get("motivo"), 240) or "")


def suggest_kind(
    llm: LLMClient, cfg: DiagramsConfig, topic: str, headings: Sequence[str]
) -> Suggestion | None:
    user = load_prompt("diagram_suggest_user").format(
        topic=topic, headings="\n".join(f"- {h}" for h in headings) or "- (ninguna)"
    )
    return _ask(
        llm, load_prompt("diagram_suggest_system"), user, cfg.suggest_max_tokens, parse_suggestion
    )


# --- the whole run ------------------------------------------------------------------------------


Diagram = tuple[Flow, ...] | tuple[SequenceVersion, ...] | Fishbone


@dataclass(frozen=True)
class DiagramProgress:
    processed: int  # fragments read so far
    skipped: int  # fragments whose reply was unusable twice
    stage: str  # reading | grouping | done
    diagram: Diagram


def generate_diagram(
    llm: LLMClient,
    cfg: DiagramsConfig,
    kind: str,
    topic: str,
    fragments: Sequence[Fragment],
    llm_turn: Callable[[], AbstractContextManager[object]] = nullcontext,
) -> Iterator[DiagramProgress]:
    """Read the fragments one by one and yield the diagram as it grows, then combine (ADR-036)."""
    skipped = 0
    found_steps: list[list[Step]] = []
    versions: list[SequenceVersion] = []
    causes: list[Cause] = []
    over = 0
    effect = effect_label(topic)
    for i, fragment in enumerate(fragments, start=1):
        with llm_turn():
            if kind == "flow":
                steps = extract_steps(llm, cfg, topic, fragment)
                if steps is None:
                    skipped += 1
                elif steps:
                    found_steps.append(steps)
            elif kind == "sequence":
                version = extract_sequence(llm, cfg, topic, fragment)
                if version is None:
                    skipped += 1
                else:
                    versions.append(version)
            else:
                new = extract_causes(llm, cfg, effect, fragment)
                if new is None:
                    skipped += 1
                else:
                    causes, extra = merge_causes(causes, new, cfg.max_causes)
                    over += extra
        partial: Diagram
        if kind == "flow":
            partial = flow_versions(found_steps, cfg.min_steps)
        elif kind == "sequence":
            partial = order_versions(versions)
        else:
            partial = ungrouped(effect, causes, over)
        yield DiagramProgress(i, skipped, "reading", partial)
    final: Diagram
    if kind == "flow":
        final = flow_versions(found_steps, cfg.min_steps)
    elif kind == "sequence":
        final = order_versions(versions)
    else:
        fishbone = ungrouped(effect, causes, over)
        if causes:
            yield DiagramProgress(len(fragments), skipped, "grouping", fishbone)
            with llm_turn():
                fishbone = group_causes(llm, cfg, fishbone, "model")
        final = fishbone
    yield DiagramProgress(len(fragments), skipped, "done", final)
