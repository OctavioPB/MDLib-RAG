"""Layouts, page JSON and exports of the diagram types of Mapas (ADR-036).

The server lays every diagram out and the page only draws what it receives: boxes (with their
label already wrapped), polylines and frames, all in one coordinate space. The same drawing gives
the Obsidian Canvas file; Mermaid text is written from the checked structure, not the drawing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date
from typing import Any

from biblio.generate.context import Fragment
from biblio.maps.build import ConceptMap
from biblio.maps.diagrams import (
    END,
    Cause,
    Fishbone,
    Flow,
    Message,
    SequenceVersion,
    Step,
    flow_edges,
    frames,
)

Point = tuple[int, int]


@dataclass(frozen=True)
class Box:
    id: str
    role: str  # start | end | step | decision | participant | effect | category | cause
    label: str
    lines: tuple[str, ...]  # the label wrapped to the box
    x: int  # top-left corner
    y: int
    w: int
    h: int
    sources: tuple[int, ...] = ()


@dataclass(frozen=True)
class Line:
    id: str  # a message id (selectable) or ""
    role: str  # flow | message | reply | lifeline | bone | spine | twig
    points: tuple[Point, ...]
    arrow: bool
    dashed: bool = False
    label: str = ""
    label_at: Point = (0, 0)
    anchor: str = "middle"  # text-anchor of the label
    source: str = ""  # box ids, for the Canvas edges
    target: str = ""


@dataclass(frozen=True)
class Frame:
    label: str
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class Drawing:
    width: int
    height: int
    boxes: tuple[Box, ...]
    lines: tuple[Line, ...]
    frames: tuple[Frame, ...] = ()


def wrap(text: str, chars: int, max_lines: int = 3) -> tuple[str, ...]:
    """Greedy word wrap; the last line ends with «…» if the text does not fit."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) <= chars or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        rest = " ".join(lines[max_lines - 1 :])
        lines = [*lines[: max_lines - 1], rest[: chars - 1].rstrip() + "…"]
    return tuple(lines) or ("",)


def _shift(boxes: list[Box], lines: list[Line], found: list[Frame], margin: int = 24) -> Drawing:
    """Move everything so the drawing starts at (margin, margin)."""
    xs = [b.x for b in boxes] + [p[0] for ln in lines for p in ln.points] + [f.x for f in found]
    ys = [b.y for b in boxes] + [p[1] for ln in lines for p in ln.points] + [f.y for f in found]
    xe = [b.x + b.w for b in boxes] + [f.x + f.w for f in found] + xs
    ye = [b.y + b.h for b in boxes] + [f.y + f.h for f in found] + ys
    if not xs:
        return Drawing(0, 0, (), (), ())
    dx, dy = margin - min(xs), margin - min(ys)

    def move(p: Point) -> Point:
        return (p[0] + dx, p[1] + dy)

    return Drawing(
        max(xe) - min(xs) + 2 * margin,
        max(ye) - min(ys) + 2 * margin,
        tuple(replace(b, x=b.x + dx, y=b.y + dy) for b in boxes),
        tuple(
            replace(ln, points=tuple(move(p) for p in ln.points), label_at=move(ln.label_at))
            for ln in lines
        ),
        tuple(replace(f, x=f.x + dx, y=f.y + dy) for f in found),
    )


# --- flowchart: one column, loops on the left, skips on the right --------------------------------

_STEP_W, _DECISION_W, _PILL_W, _GAP, _LANE = 280, 250, 140, 56, 30


def layout_flow(flow: Flow) -> Drawing:
    if not flow.steps:
        return Drawing(0, 0, (), (), ())
    boxes: list[Box] = [Box("start", "start", "Inicio", ("Inicio",), -_PILL_W // 2, 0, _PILL_W, 44)]
    y = 44 + _GAP
    for step in flow.steps:
        if step.decision:
            lines = wrap(step.label, 20)
            h = max(100, 22 * len(lines) + 64)
            box = Box(step.id, "decision", step.label, lines, -_DECISION_W // 2, y, _DECISION_W, h)
        else:
            lines = wrap(step.label, 30)
            h = max(52, 22 * len(lines) + 22)
            box = Box(step.id, "step", step.label, lines, -_STEP_W // 2, y, _STEP_W, h)
        boxes.append(replace(box, sources=step.sources))
        y += h + _GAP
    boxes.append(Box(END, "end", "Fin", ("Fin",), -_PILL_W // 2, y, _PILL_W, 44))
    order = {b.id: i for i, b in enumerate(boxes)}
    by_id = {b.id: b for b in boxes}
    left = -_STEP_W // 2 - 40
    right = _STEP_W // 2 + 40
    lanes = {"left": 0, "right": 0}
    lines_out: list[Line] = []
    for edge in flow_edges(flow):
        a, b = by_id[edge.source], by_id[edge.target]
        ia, ib = order[a.id], order[b.id]
        ay, by_ = a.y + a.h // 2, b.y + b.h // 2
        if ib == ia + 1:
            points: tuple[Point, ...] = ((0, a.y + a.h), (0, b.y))
            label_at: Point = (16, a.y + a.h + 18)
            anchor = "start"
        elif ib <= ia:  # back to an earlier step: a loop on the left
            lane = left - _LANE * lanes["left"]
            lanes["left"] += 1
            points = ((a.x, ay), (lane, ay), (lane, by_), (b.x, by_))
            label_at, anchor = (a.x - 10, ay - 8), "end"
        else:  # skipping ahead: on the right
            lane = right + _LANE * lanes["right"]
            lanes["right"] += 1
            points = ((a.x + a.w, ay), (lane, ay), (lane, by_), (b.x + b.w, by_))
            label_at, anchor = (a.x + a.w + 10, ay - 8), "start"
        lines_out.append(
            Line(
                "",
                "flow",
                points,
                True,
                edge.inferred,
                edge.label,
                label_at,
                anchor,
                a.id,
                b.id,
            )
        )
    return _shift(boxes, lines_out, [])


# --- sequence diagram: participants in columns, messages top to bottom ---------------------------

_P_W, _P_H, _ROW = 170, 50, 58


def layout_sequence(version: SequenceVersion) -> Drawing:
    if not version.messages:
        return Drawing(0, 0, (), (), ())
    longest = max(len(m.label) for m in version.messages)
    spacing = max(210, min(380, longest * 7 + 60))
    column = {p: i * spacing for i, p in enumerate(version.participants)}
    boxes = [
        Box(f"p{i}", "participant", p, wrap(p, 20, 2), column[p] - _P_W // 2, 0, _P_W, _P_H)
        for i, p in enumerate(version.participants)
    ]
    runs = frames(version.messages)
    starts = {r[0]: r for r in runs}
    ends = {r[1] for r in runs}
    y = _P_H + 46
    rows: list[int] = []
    for i, _m in enumerate(version.messages):
        if i in starts:
            y += 30  # room for the frame's label
        rows.append(y)
        y += _ROW
        if i in ends:
            y += 14
    bottom = y
    lines: list[Line] = [
        Line("", "lifeline", ((column[p], _P_H), (column[p], bottom)), False, True)
        for p in version.participants
    ]
    for m, row in zip(version.messages, rows, strict=True):
        xs, xr = column[m.sender], column[m.receiver]
        role = "reply" if m.reply else "message"
        if xs == xr:
            points: tuple[Point, ...] = (
                (xs, row),
                (xs + 46, row),
                (xs + 46, row + 24),
                (xs, row + 24),
            )
            lines.append(
                Line(m.id, role, points, True, m.reply, m.label, (xs + 56, row + 16), "start")
            )
        else:
            lines.append(
                Line(
                    m.id,
                    role,
                    ((xs, row), (xr, row)),
                    True,
                    m.reply,
                    m.label,
                    ((xs + xr) // 2, row - 9),
                )
            )
    box_of = {p: f"p{i}" for i, p in enumerate(version.participants)}
    by_id = {m.id: m for m in version.messages}
    lines = [
        replace(ln, source=box_of[by_id[ln.id].sender], target=box_of[by_id[ln.id].receiver])
        if ln.id in by_id
        else ln
        for ln in lines
    ]
    found = []
    for first, last, condition in runs:
        involved = [column[m.sender] for m in version.messages[first : last + 1]]
        involved += [column[m.receiver] for m in version.messages[first : last + 1]]
        self_loop = any(m.sender == m.receiver for m in version.messages[first : last + 1])
        x0 = min(involved) - 70
        x1 = max(involved) + (170 if self_loop else 70)
        y0 = rows[first] - 52
        y1 = rows[last] + (40 if self_loop else 18)
        found.append(Frame(condition, x0, y0, x1 - x0, y1 - y0))
    return _shift(boxes, lines, found)


# --- cause-effect: categories above and below a spine that points at the effect ------------------

_COL, _CAT_W, _CAT_H, _CAUSE_W, _CAUSE_GAP = 380, 190, 46, 236, 14


def _cause_height(lines: Sequence[str]) -> int:
    return max(40, 20 * len(lines) + 16)


def layout_fishbone(fishbone: Fishbone) -> Drawing:
    causes = {c.id: c for c in fishbone.causes}
    categories = [
        (name, [causes[i] for i in ids if i in causes]) for name, ids in fishbone.categories
    ]
    categories = [(name, members) for name, members in categories if members]
    effect_lines = wrap(fishbone.effect, 22, 4)
    effect_h = max(84, 22 * len(effect_lines) + 30)
    if not categories:
        return Drawing(0, 0, (), (), ())
    wrapped = [[(c, wrap(c.label, 26)) for c in members] for _name, members in categories]
    stacks = [sum(_cause_height(ls) + _CAUSE_GAP for _c, ls in group) for group in wrapped]
    top_stack = max((stacks[i] for i in range(0, len(categories), 2)), default=0)
    bottom_stack = max((stacks[i] for i in range(1, len(categories), 2)), default=0)
    spine = _CAT_H + 24 + top_stack + 30
    columns = math.ceil(len(categories) / 2)
    boxes: list[Box] = []
    lines: list[Line] = []
    for i, ((name, _members), group) in enumerate(zip(categories, wrapped, strict=True)):
        bone_x = (i // 2) * _COL + _CAT_W // 2
        top = i % 2 == 0
        cat_id = f"k{i}"
        cat_y = 0 if top else spine + 30 + bottom_stack + 24
        sources = tuple(sorted({s for c, _ls in group for s in c.sources}))
        boxes.append(
            Box(
                cat_id,
                "category",
                name,
                wrap(name, 22, 2),
                bone_x - _CAT_W // 2,
                cat_y,
                _CAT_W,
                _CAT_H,
                sources,
            )
        )
        bone = ((bone_x, _CAT_H), (bone_x, spine)) if top else ((bone_x, spine), (bone_x, cat_y))
        lines.append(Line("", "bone", bone, False, source=cat_id, target="effect"))
        y = _CAT_H + 24 if top else spine + 30
        for cause, cause_lines in group:
            h = _cause_height(cause_lines)
            boxes.append(
                Box(
                    cause.id,
                    "cause",
                    cause.label,
                    cause_lines,
                    bone_x + 28,
                    y,
                    _CAUSE_W,
                    h,
                    cause.sources,
                )
            )
            mid = y + h // 2
            lines.append(
                Line(
                    "",
                    "twig",
                    ((bone_x, mid), (bone_x + 28, mid)),
                    False,
                    source=cause.id,
                    target=cat_id,
                )
            )
            y += h + _CAUSE_GAP
    effect_x = columns * _COL + 20
    boxes.append(
        Box(
            "effect",
            "effect",
            fishbone.effect,
            effect_lines,
            effect_x,
            spine - effect_h // 2,
            220,
            effect_h,
        )
    )
    lines.append(Line("", "spine", ((-30, spine), (effect_x, spine)), True))
    return _shift(boxes, lines, [])


# --- the Novak concept map: the same map, top-down in levels -------------------------------------


_ROW_GAP = 64  # between children of one branch
_BRANCH_GAP = 28  # between branches


def novak_layout(cmap: ConceptMap) -> ConceptMap:
    """A tree read from left to right: the topic, its branches in a column, each branch's
    children beside it. Top-down rows were tried first: with 9 branches they were ~5,000 px wide,
    and wrapping them into bands made the topic's lines cross the bands above."""
    level1 = [n for n in cmap.nodes if n.level == 1]
    kids = {n.id: [c for c in cmap.nodes if c.parent == n.id] for n in level1}
    placed: dict[str, tuple[int, int]] = {}
    y = 0
    for node in level1:
        children = kids[node.id]
        block = max(1, len(children)) * _ROW_GAP
        placed[node.id] = (400, y + block // 2)
        for j, child in enumerate(children):
            placed[child.id] = (850, y + j * _ROW_GAP + _ROW_GAP // 2)
        y += block + _BRANCH_GAP
    placed["root"] = (0, max(0, y - _BRANCH_GAP) // 2)
    nodes = tuple(
        replace(n, x=placed.get(n.id, (n.x, n.y))[0], y=placed.get(n.id, (n.x, n.y))[1])
        for n in cmap.nodes
    )
    return replace(cmap, nodes=nodes)


# --- Mermaid (pasted into Obsidian, which draws it) ----------------------------------------------


def _quoted(text: str) -> str:
    return '"' + text.replace('"', "#quot;") + '"'


def _plain(text: str) -> str:
    """For mindmap and sequence text, where brackets, quotes and semicolons have a meaning."""
    for ch in '()[]{}"`;#':
        text = text.replace(ch, " ")
    return " ".join(text.split()) or "·"


def flow_mermaid(flow: Flow) -> str:
    lines = ["flowchart TD", '  inicio(["Inicio"])']
    for s in flow.steps:
        lines.append(
            f"  {s.id}{{{_quoted(s.label)}}}" if s.decision else f"  {s.id}[{_quoted(s.label)}]"
        )
    lines.append('  fin(["Fin"])')
    names = {"start": "inicio", END: "fin"}
    for e in flow_edges(flow):
        arrow = "-.->" if e.inferred else "-->"
        label = f"|{_quoted(e.label)}|" if e.label else ""
        lines.append(
            f"  {names.get(e.source, e.source)} {arrow}{label} {names.get(e.target, e.target)}"
        )
    return "\n".join(lines) + "\n"


def sequence_mermaid(version: SequenceVersion) -> str:
    ids = {p: f"p{i}" for i, p in enumerate(version.participants)}
    lines = ["sequenceDiagram"]
    lines += [f"  participant {ids[p]} as {_plain(p)}" for p in version.participants]
    chains: list[list[tuple[int, int, str]]] = []  # frames that follow each other: alt … else
    for run in frames(version.messages):
        if chains and chains[-1][-1][1] + 1 == run[0]:
            chains[-1].append(run)
        else:
            chains.append([run])
    opening = {ch[0][0]: ("alt" if len(ch) > 1 else "opt", ch[0][2]) for ch in chains}
    otherwise = {run[0]: run[2] for ch in chains for run in ch[1:]}
    closing = {ch[-1][1] for ch in chains}
    for i, m in enumerate(version.messages):
        if i in opening:
            keyword, condition = opening[i]
            lines.append(f"  {keyword} {_plain(condition)}")
        elif i in otherwise:
            lines.append(f"  else {_plain(otherwise[i])}")
        arrow = "-->>" if m.reply else "->>"
        lines.append(f"  {ids[m.sender]}{arrow}{ids[m.receiver]}: {_plain(m.label)}")
        if i in closing:
            lines.append("  end")
    return "\n".join(lines) + "\n"


def fishbone_mermaid(fishbone: Fishbone) -> str:
    causes = {c.id: c for c in fishbone.causes}
    lines = ["mindmap", f"  root(({_plain(fishbone.effect)}))"]
    for name, ids in fishbone.categories:
        members = [causes[i] for i in ids if i in causes]
        if not members:
            continue
        lines.append(f"    {_plain(name)}")
        lines += [f"      {_plain(c.label)}" for c in members]
    return "\n".join(lines) + "\n"


def novak_mermaid(cmap: ConceptMap) -> str:
    lines = ["flowchart TD"]
    lines += [f"  {n.id}[{_quoted(n.shown)}]" for n in cmap.nodes]
    for e in cmap.edges:
        arrow = "-.->" if e.kind == "cross" else "-->"
        label = f"|{_quoted(e.label)}|" if e.label else ""
        lines.append(f"  {e.source} {arrow}{label} {e.target}")
    return "\n".join(lines) + "\n"


# --- page JSON, and back ------------------------------------------------------------------------


def drawing_json(drawing: Drawing) -> dict[str, Any]:
    return {
        "width": drawing.width,
        "height": drawing.height,
        "boxes": [
            {**asdict(b), "lines": list(b.lines), "sources": list(b.sources)} for b in drawing.boxes
        ],
        "lines": [
            {**asdict(ln), "points": [list(p) for p in ln.points], "label_at": list(ln.label_at)}
            for ln in drawing.lines
        ],
        "frames": [asdict(f) for f in drawing.frames],
    }


def flow_json(versions: Sequence[Flow], current: int) -> dict[str, Any]:
    return {
        "current": current,
        "versions": [
            {
                "source": f.source,
                "steps": [{**asdict(s), "sources": list(s.sources)} for s in f.steps],
            }
            for f in versions
        ],
    }


def flow_from_json(data: dict[str, Any]) -> tuple[tuple[Flow, ...], int]:
    versions = tuple(
        Flow(
            int(v["source"]),
            tuple(
                Step(
                    s["id"],
                    s["label"],
                    bool(s["decision"]),
                    s["quote"],
                    tuple(s["sources"]),
                    int(s["position"]),
                    s.get("yes"),
                    s.get("no"),
                )
                for s in v["steps"]
            ),
        )
        for v in data["versions"]
    )
    return versions, int(data.get("current", 0))


def sequence_json(versions: Sequence[SequenceVersion], current: int) -> dict[str, Any]:
    return {
        "current": current,
        "versions": [
            {
                "source": v.source,
                "participants": list(v.participants),
                "messages": [asdict(m) for m in v.messages],
            }
            for v in versions
        ],
    }


def sequence_from_json(data: dict[str, Any]) -> tuple[tuple[SequenceVersion, ...], int]:
    versions = tuple(
        SequenceVersion(
            int(v["source"]),
            tuple(v["participants"]),
            tuple(Message(**m) for m in v["messages"]),
        )
        for v in data["versions"]
    )
    return versions, int(data.get("current", 0))


def fishbone_json(fishbone: Fishbone) -> dict[str, Any]:
    return {
        "effect": fishbone.effect,
        "causes": [{**asdict(c), "sources": list(c.sources)} for c in fishbone.causes],
        "categories": [{"name": name, "causes": list(ids)} for name, ids in fishbone.categories],
        "grouping": fishbone.grouping,
        "omitted": fishbone.omitted,
    }


def fishbone_from_json(data: dict[str, Any]) -> Fishbone:
    return Fishbone(
        data["effect"],
        tuple(Cause(c["id"], c["label"], c["quote"], tuple(c["sources"])) for c in data["causes"]),
        tuple((g["name"], tuple(g["causes"])) for g in data["categories"]),
        data["grouping"],
        int(data.get("omitted", 0)),
    )


def diagram_page(kind: str, topic: str, diagram: object, current: int = 0) -> dict[str, Any]:
    """What the page receives and `data/maps.db` keeps for a flow, sequence or fishbone."""
    if kind == "flow":
        assert isinstance(diagram, tuple)
        flows: tuple[Flow, ...] = diagram
        current = current if 0 <= current < len(flows) else 0
        flow = flows[current] if flows else Flow(0, ())
        data, drawing, mermaid = flow_json(flows, current), layout_flow(flow), flow_mermaid(flow)
        count, omitted = len(flow.steps), 0
        names = [s.label for s in flow.steps]
    elif kind == "sequence":
        assert isinstance(diagram, tuple)
        versions: tuple[SequenceVersion, ...] = diagram
        current = current if 0 <= current < len(versions) else 0
        shown = versions[current] if versions else SequenceVersion(0, (), ())
        data = sequence_json(versions, current)
        drawing, mermaid = layout_sequence(shown), sequence_mermaid(shown)
        count, omitted = len(shown.messages), 0
        names = list(shown.participants) + [m.label for m in shown.messages]
    else:
        assert isinstance(diagram, Fishbone)
        data, drawing, mermaid = (
            fishbone_json(diagram),
            layout_fishbone(diagram),
            fishbone_mermaid(diagram),
        )
        count, omitted = len(diagram.causes), diagram.omitted
        names = [name for name, _ in diagram.categories] + [c.label for c in diagram.causes]
    return {
        "kind": kind,
        "topic": topic,
        "data": data,
        "drawing": drawing_json(drawing),
        "mermaid": mermaid,
        "count": count,
        "omitted": omitted,
        "names": names,
    }


def diagram_from_page(page: dict[str, Any]) -> tuple[str, object, int]:
    """(kind, structure, current version) of a stored flow, sequence or fishbone."""
    kind = page["kind"]
    if kind == "flow":
        flows, current = flow_from_json(page["data"])
        return kind, flows, current
    if kind == "sequence":
        versions, current = sequence_from_json(page["data"])
        return kind, versions, current
    return kind, fishbone_from_json(page["data"]), 0


# --- saving to drafts/maps: Markdown with the Mermaid block, and Obsidian Canvas ------------------

_TITLES = {
    "flow": "Diagrama de flujo",
    "sequence": "Diagrama de secuencia",
    "fishbone": "Causa-efecto",
}
_CANVAS_COLOURS = {
    "start": "4",
    "end": "4",
    "effect": "4",
    "category": "5",
    "decision": "3",
    "participant": "5",
}


def _cites(sources: Sequence[int]) -> str:
    return "".join(f"[{s}]" for s in sources)


def _sources(fragments: Sequence[Fragment]) -> list[str]:
    lines = ["", "## Fuentes", ""]
    lines += [f"{f.number}. {f.citation} (líneas {f.start_line}-{f.end_line})" for f in fragments]
    return lines


def diagram_markdown(page: dict[str, Any], fragments: Sequence[Fragment], today: date) -> str:
    kind, diagram, current = diagram_from_page(page)
    lines = [
        f"# {_TITLES[kind]}: {page['topic']}",
        "",
        f"> **Generado por Biblioteca el {today.isoformat()}** a partir de tus notas.",
        "> Cada pieza cita con [n] el fragmento de donde sale; revísala contra la fuente.",
        "",
    ]
    if kind == "flow":
        assert isinstance(diagram, tuple)
        flow = diagram[current] if diagram else Flow(0, ())
        number = {s.id: i for i, s in enumerate(flow.steps, start=1)}
        for i, s in enumerate(flow.steps, start=1):
            branches = ""
            if s.decision:
                where = {e.label: e.target for e in flow_edges(flow) if e.source == s.id}
                parts = [
                    f"{label}: {'fin' if t == END else f'paso {number[t]}'}"
                    for label, t in where.items()
                ]
                branches = f" ({', '.join(parts)})"
            lines.append(f"{i}. **{s.label}**{branches} {_cites(s.sources)} — «{s.quote}»")
        lines += ["", "Las flechas punteadas marcan un orden que tus notas no escriben."]
    elif kind == "sequence":
        assert isinstance(diagram, tuple)
        shown = diagram[current] if diagram else SequenceVersion(0, (), ())
        lines.append(f"Participantes: {', '.join(shown.participants)}")
        lines.append("")
        for i, m in enumerate(shown.messages, start=1):
            when = f" _(si {m.condition})_" if m.condition else ""
            kind_note = " (respuesta)" if m.reply else ""
            lines.append(
                f"{i}. {m.sender} → {m.receiver}: **{m.label}**{kind_note}{when} "
                f"[{shown.source}] — «{m.quote}»"
            )
    else:
        assert isinstance(diagram, Fishbone)
        causes = {c.id: c for c in diagram.causes}
        lines.append(f"Efecto: **{diagram.effect}**")
        for name, ids in diagram.categories:
            members = [causes[i] for i in ids if i in causes]
            if members:
                lines += ["", f"## {name}", ""]
                lines += [f"- {c.label} {_cites(c.sources)} — «{c.quote}»" for c in members]
    lines += ["", "```mermaid", page["mermaid"].rstrip("\n"), "```"]
    lines += _sources(fragments)
    return "\n".join(lines) + "\n"


def novak_markdown_block(cmap: ConceptMap) -> list[str]:
    return ["", "```mermaid", novak_mermaid(cmap).rstrip("\n"), "```"]


def _wikilink(fragment: Fragment) -> str:
    path = fragment.doc_path.removesuffix(".md")
    section = fragment.heading_path[-1] if len(fragment.heading_path) > 1 else ""
    target = f"{path}#{section}" if section else path
    return f"[[{target}|{fragment.number}]]"


def diagram_canvas(page: dict[str, Any], fragments: Sequence[Fragment]) -> dict[str, Any]:
    """The drawing as JSON Canvas: one text node per box, one edge per line between two boxes."""
    by_number = {f.number: f for f in fragments}
    drawing = page["drawing"]
    nodes = []
    for b in drawing["boxes"]:
        links = " ".join(_wikilink(by_number[s]) for s in b["sources"] if s in by_number)
        text = f"**{b['label']}**" + (f"\n\nFuentes: {links}" if links else "")
        node: dict[str, Any] = {
            "id": b["id"],
            "type": "text",
            "text": text,
            "x": b["x"] * 2,  # Canvas text is larger than the page's: more room between boxes
            "y": b["y"] * 2,
            "width": max(b["w"] * 2 // 1, 160),
            "height": b["h"] * 2 + (40 if links else 0),
        }
        if b["role"] in _CANVAS_COLOURS:
            node["color"] = _CANVAS_COLOURS[b["role"]]
        nodes.append(node)
    ids = {b["id"] for b in drawing["boxes"]}
    edges = []
    number = 0
    for ln in drawing["lines"]:
        if ln["source"] in ids and ln["target"] in ids:
            number += 1
            edge: dict[str, Any] = {
                "id": f"e{number}",
                "fromNode": ln["source"],
                "toNode": ln["target"],
            }
            if ln["label"]:
                edge["label"] = (
                    f"{number}. {ln['label']}" if page["kind"] == "sequence" else ln["label"]
                )
            if ln["dashed"]:
                edge["color"] = "3"
            if not ln["arrow"]:
                edge["toEnd"] = "none"
            edges.append(edge)
    return {"nodes": nodes, "edges": edges}
