"""Choosing the fragments of a map, and saving a map: Markdown outline and Obsidian Canvas.

The Canvas file follows the open JSON Canvas format; each node links to its sources with wikilinks,
which Obsidian resolves when the file is inside the vault. Maps are saved to `drafts/maps/`, outside
the library, so they are never indexed (ADR-022).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

from biblio.config import MapsConfig
from biblio.generate.context import Fragment
from biblio.maps.build import ConceptMap, MapNode, map_outline
from biblio.practice.quiz import is_content_heading
from biblio.retrieve.types import Hit

# JSON Canvas preset colours: 1 red, 2 orange, 3 yellow, 4 green, 5 cyan, 6 purple
_CANVAS_COLOURS = {0: "4", 1: "5"}
_NODE_HEIGHT = 64


def map_fragments(hits: Sequence[Hit], cfg: MapsConfig) -> list[Fragment]:
    """The retrieved sections a map reads, numbered from 1; navigation sections are skipped."""
    return numbered_fragments(hits, cfg.fragments, cfg.max_fragment_chars)


def numbered_fragments(hits: Sequence[Hit], count: int, max_chars: int) -> list[Fragment]:
    """Up to `count` distinct content sections, numbered from 1 (maps, ADR-022; ADR-027)."""
    fragments: list[Fragment] = []
    seen: set[tuple[str, int]] = set()
    for hit in hits:
        key = (hit.doc_path, hit.start_line)
        if key in seen or len(hit.heading_path) < 2 or not is_content_heading(hit.heading_path[-1]):
            continue
        seen.add(key)
        fragments.append(
            Fragment(
                number=len(fragments) + 1,
                doc_path=hit.doc_path,
                heading_path=hit.heading_path,
                text=hit.text[:max_chars],
                start_line=hit.start_line,
                end_line=hit.end_line,
            )
        )
        if len(fragments) == count:
            break
    return fragments


def to_markdown(cmap: ConceptMap, fragments: Sequence[Fragment], today: date) -> str:
    lines = [
        f"# Mapa: {cmap.topic}",
        "",
        f"> **Mapa conceptual generado por Biblioteca el {today.isoformat()}**, "
        "a partir de tus notas.",
        "> Cada concepto cita con [n] el fragmento donde aparece; revísalo contra la fuente.",
        "",
        *map_outline(cmap),
    ]
    cross = [e for e in cmap.edges if e.kind == "cross"]
    if cross:
        names = {n.id: n.label for n in cmap.nodes}
        lines += ["", "## Otras relaciones", ""]
        lines += [f"- {names[e.source]} — _{e.label}_ → {names[e.target]}" for e in cross]
    lines += ["", "## Fuentes", ""]
    for f in fragments:
        lines.append(f"{f.number}. {f.citation} (líneas {f.start_line}-{f.end_line})")
    return "\n".join(lines) + "\n"


def _wikilink(fragment: Fragment) -> str:
    """`[[path/without/extension#Section|n]]`: Obsidian opens the note at that heading."""
    path = fragment.doc_path.removesuffix(".md")
    section = fragment.heading_path[-1] if len(fragment.heading_path) > 1 else ""
    target = f"{path}#{section}" if section else path
    return f"[[{target}|{fragment.number}]]"


def _width(node: MapNode) -> int:
    return max(140, min(340, 9 * len(node.label) + 48))


def to_canvas(cmap: ConceptMap, fragments: Sequence[Fragment]) -> dict[str, Any]:
    by_number = {f.number: f for f in fragments}
    nodes = []
    for n in cmap.nodes:
        width = _width(n)
        links = " ".join(_wikilink(by_number[s]) for s in n.sources if s in by_number)
        text = f"**{n.label}**" + (f"\n\nFuentes: {links}" if links else "")
        node: dict[str, Any] = {
            "id": n.id,
            "type": "text",
            "text": text,
            "x": n.x - width // 2,
            "y": n.y - _NODE_HEIGHT // 2,
            "width": width,
            "height": _NODE_HEIGHT + (40 if links else 0),
        }
        if n.level in _CANVAS_COLOURS:
            node["color"] = _CANVAS_COLOURS[n.level]
        nodes.append(node)
    edges = []
    for i, e in enumerate(cmap.edges):
        edge: dict[str, Any] = {"id": f"e{i}", "fromNode": e.source, "toNode": e.target}
        if e.label:
            edge["label"] = e.label
        if e.kind == "cross":
            edge["color"] = "3"
        edges.append(edge)
    return {"nodes": nodes, "edges": edges}
