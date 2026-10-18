"""`biblio draft`: outline from evidence and sibling guides → user edits it → section by section.

The result is a skeleton to edit, not a finished guide (ADR-008): every claim is cited, sections
without evidence say so, and the draft ends with the sources it actually used.
"""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date

from biblio.config import Config
from biblio.generate.ask import load_prompt
from biblio.generate.citations import extract_citations, is_abstention
from biblio.generate.context import Context, Fragment, assemble
from biblio.generate.llm import LLMClient, Message
from biblio.ingest.tokens import TokenCounter
from biblio.retrieve.types import Hit, Retriever

_CITATION_GROUP_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_SLUG_RE = re.compile(r"[^a-z0-9]+")
OUTLINE_SECTION_PREFIX = "## "


@dataclass(frozen=True)
class OutlineSection:
    title: str
    notes: str = ""


@dataclass(frozen=True)
class Outline:
    topic: str
    sections: tuple[OutlineSection, ...]
    siblings: tuple[str, ...] = ()  # paths of the guides whose structure it imitates


@dataclass
class SourceRegistry:
    """Global [n] numbering of the fragments cited anywhere in the draft."""

    fragments: list[Fragment] = field(default_factory=list)
    _index: dict[tuple[str, int, int], int] = field(default_factory=dict)

    def number(self, fragment: Fragment) -> int:
        key = (fragment.doc_path, fragment.start_line, fragment.end_line)
        if key not in self._index:
            self.fragments.append(fragment)
            self._index[key] = len(self.fragments)
        return self._index[key]


@dataclass(frozen=True)
class SectionResult:
    title: str
    body: str  # citations already renumbered to the global sources
    missing_evidence: bool
    uncited: bool
    invalid_citations: tuple[int, ...]


def slugify(text: str) -> str:
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return _SLUG_RE.sub("-", plain).strip("-")[:60] or "borrador"


# --- outline ------------------------------------------------------------------------------------


def parse_outline(text: str, topic: str) -> Outline:
    """`## Title` lines are sections, in order; the lines under one are its notes."""
    sections: list[OutlineSection] = []
    title: str | None = None
    notes: list[str] = []
    in_comment = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("<!--"):
            in_comment = "-->" not in line
            continue
        if in_comment:
            in_comment = "-->" not in line
            continue
        if line.startswith(OUTLINE_SECTION_PREFIX):
            if title:
                sections.append(OutlineSection(title, " ".join(notes).strip()))
            title, notes = line[len(OUTLINE_SECTION_PREFIX) :].strip(), []
        elif title and line and not line.startswith("#"):
            notes.append(line)
    if title:
        sections.append(OutlineSection(title, " ".join(notes).strip()))
    return Outline(topic, tuple(sections))


_NAVIGATION_TITLES = {
    "indice",
    "tabla de contenidos",
    "tabla de contenido",
    "contenido",
    "contenidos",
}


def drop_non_content_sections(outline: Outline) -> Outline:
    """Remove what small models add despite the prompt: the guide's own title as a section and
    navigation sections (a table of contents has nothing to ground)."""

    def plain(text: str) -> str:
        decomposed = unicodedata.normalize("NFKD", text)
        ascii_text = decomposed.encode("ascii", "ignore").decode().casefold()
        return re.sub(r"^[\d.\s]+", "", re.sub(r"[^\w\s.]", " ", ascii_text)).strip()

    topic = plain(outline.topic)
    kept = tuple(
        s
        for s in outline.sections
        if plain(s.title) not in _NAVIGATION_TITLES
        and not (topic and topic in plain(s.title) and plain(s.title).startswith(("guia", "guide")))
    )
    return Outline(outline.topic, kept)


def render_outline_file(outline: Outline, draft_command: str) -> str:
    lines = [
        f"# Esquema: {outline.topic}",
        "",
        '<!-- Edítalo libremente: cada línea que empieza por "## " es una sección del borrador, en',
        "     este orden; las líneas debajo son notas para esa sección. Después ejecuta:",
        f"     {draft_command} -->",
        "",
    ]
    for section in outline.sections:
        lines.append(f"{OUTLINE_SECTION_PREFIX}{section.title}")
        if section.notes:
            lines.append(section.notes)
        lines.append("")
    return "\n".join(lines)


def sibling_outlines(conn: sqlite3.Connection, doc_paths: Sequence[str], max_lines: int) -> str:
    """The section structure (H2/H3 level) of a few related guides, to imitate their conventions."""
    blocks = []
    for path in doc_paths:
        rows = conn.execute(
            "SELECT c.heading_path FROM chunks c JOIN documents d ON d.id = c.document_id "
            "WHERE d.path = ? ORDER BY c.ordinal",
            (path,),
        ).fetchall()
        seen: list[str] = []
        for (heading_json,) in rows:
            heading = json.loads(heading_json)
            for depth, title in enumerate(heading[1:3], start=2):
                line = f"{'#' * depth} {title}"
                if line not in seen:
                    seen.append(line)
        if seen:
            title = json.loads(rows[0][0])[0]
            blocks.append(f"Guía «{title}» ({path}):\n" + "\n".join(seen[:max_lines]))
    return "\n\n".join(blocks) or "(ninguna)"


# --- drafting -----------------------------------------------------------------------------------


class Drafter:
    def __init__(
        self,
        config: Config,
        retriever: Retriever,
        llm: LLMClient,
        counter: TokenCounter,
        conn: sqlite3.Connection,
    ) -> None:
        self._config = config
        self._retriever = retriever
        self._llm = llm
        self._counter = counter
        self._conn = conn
        self.missing = load_prompt("draft_missing")

    def _generate(self, system: str, user: str, on_token: Callable[[str], None] | None) -> str:
        parts = []
        for token in self._llm.stream(
            (Message("system", system), Message("user", user)), self._config.llm.max_answer_tokens
        ):
            parts.append(token)
            if on_token:
                on_token(token)
        return "".join(parts).strip()

    def _context(self, query: str, system: str, user_template: str, **fields: str) -> Context:
        hits = self._retriever.search(query, self._config.retrieval.final_k)
        fixed = self._counter.count(system) + self._counter.count(
            user_template.format(evidence="", **fields)
        )
        cfg = self._config.llm
        return assemble(hits, self._counter, cfg.context_budget_tokens, cfg.max_fragments, fixed)

    def propose_outline(self, topic: str, on_token: Callable[[str], None] | None = None) -> Outline:
        d = self._config.draft
        hits: list[Hit] = self._retriever.search(topic, self._config.retrieval.final_k)
        siblings = list(dict.fromkeys(h.doc_path for h in hits))[: d.sibling_guides]
        system = load_prompt("draft_outline_system").format(
            min_sections=d.min_sections, max_sections=d.max_sections
        )
        user_template = load_prompt("draft_outline_user")
        sibling_text = sibling_outlines(self._conn, siblings, d.sibling_outline_lines)
        context = self._context(topic, system, user_template, topic=topic, siblings=sibling_text)
        user = user_template.format(
            topic=topic, siblings=sibling_text, evidence=context.render() or "(ninguno)"
        )
        outline = drop_non_content_sections(
            parse_outline(self._generate(system, user, on_token), topic)
        )
        return Outline(topic, outline.sections[: d.max_sections], tuple(siblings))

    def write_section(
        self,
        outline: Outline,
        section: OutlineSection,
        registry: SourceRegistry,
        on_token: Callable[[str], None] | None = None,
    ) -> SectionResult:
        system = load_prompt("draft_section_system").format(missing=self.missing)
        user_template = load_prompt("draft_section_user")
        plan = "\n".join(f"- {s.title}" for s in outline.sections)
        notes = f"Notas: {section.notes}" if section.notes else ""
        fields = {"topic": outline.topic, "outline": plan, "section": section.title, "notes": notes}
        context = self._context(f"{outline.topic} {section.title}", system, user_template, **fields)
        if not context.fragments:
            return SectionResult(section.title, f"> {self.missing}", True, False, ())
        text = self._generate(
            system, user_template.format(evidence=context.render(), **fields), on_token
        )
        if is_abstention(text, self.missing):
            return SectionResult(section.title, f"> {self.missing}", True, False, ())
        body, invalid = renumber(
            strip_repeated_title(text, section.title), context.fragments, registry
        )
        uncited = not extract_citations(body)
        return SectionResult(section.title, body, False, uncited, invalid)


def strip_repeated_title(text: str, title: str) -> str:
    """Drop a leading heading that repeats the section title (the draft writes `## title`)."""
    lines = text.lstrip().splitlines()
    if lines and lines[0].lstrip().startswith("#"):
        heading = lines[0].lstrip("# ").strip().strip("*").strip()
        if heading.casefold() == title.strip().casefold():
            return "\n".join(lines[1:]).lstrip()
    return text


def renumber(
    text: str, fragments: Sequence[Fragment], registry: SourceRegistry
) -> tuple[str, tuple[int, ...]]:
    """Section-local [n] → global source numbers; numbers with no fragment become [?]."""
    local = {f.number: f for f in fragments}
    invalid: list[int] = []

    def replace_group(match: re.Match[str]) -> str:
        out = []
        for part in re.split(r"[,;]", match.group(1)):
            n = int(part.strip())
            if n in local:
                out.append(f"[{registry.number(local[n])}]")
            else:
                invalid.append(n)
                out.append("[?]")
        return "".join(out)

    return _CITATION_GROUP_RE.sub(replace_group, text), tuple(invalid)


def render_draft(
    topic: str, sections: Sequence[SectionResult], registry: SourceRegistry, today: date
) -> str:
    lines = [
        f"# {topic}",
        "",
        f"> **Borrador generado por `biblio draft` el {today.isoformat()}.** Es un esqueleto",
        "> para editar, no una guía terminada (ADR-008). Cada afirmación cita su fuente con [n];",
        "> revísala contra la sección original antes de darla por buena.",
        "",
    ]
    for s in sections:
        lines += [f"## {s.title}", "", s.body, ""]
        if s.uncited:
            lines += [
                "> ⚠ Esta sección no cita ninguna fuente: no la des por buena sin revisarla.",
                "",
            ]
        if s.invalid_citations:
            lines += ["> ⚠ El modelo citó fragmentos inexistentes, marcados como [?].", ""]
    lines += ["## Fuentes", ""]
    for n, fragment in enumerate(registry.fragments, start=1):
        lines.append(f"{n}. {fragment.citation} (líneas {fragment.start_line}-{fragment.end_line})")
    if not registry.fragments:
        lines.append("(ninguna)")
    return "\n".join(lines) + "\n"
