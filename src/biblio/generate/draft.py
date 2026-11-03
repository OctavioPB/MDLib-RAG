"""`biblio draft`: outline from evidence and sibling guides → user edits it → section by section.

The result is a skeleton to edit, not a finished guide (ADR-008): every claim is cited, sections
without evidence say so, and the draft ends with the sources it actually used. Its shape is the
user's series (header, epigraph, contents, numbered sections by role, references, footer) and its
evidence comes from the books (ADR-061).
"""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date

from biblio.books import book_area, book_id
from biblio.config import Config
from biblio.generate.ask import load_prompt
from biblio.generate.citations import extract_citations, is_abstention
from biblio.generate.context import Context, Fragment, assemble
from biblio.generate.guide_style import (
    SOURCES_TITLE,
    numbered_title,
    render_epigraph,
    render_footer,
    render_header,
    render_toc,
    role_of,
)
from biblio.generate.llm import LLMClient, Message
from biblio.ingest.tokens import TokenCounter
from biblio.maps.build import in_english
from biblio.retrieve.fusion import rrf
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


@dataclass(frozen=True)
class GuideMeta:
    """What the header and the epigraph of the series need (ADR-061)."""

    author: str = ""
    track: str = "Reference"
    series: str = ""
    level: str = "Intermedio-Avanzado"
    concepts: tuple[str, ...] = ()
    epigraph: str = ""


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


_NUMBERED_RE = re.compile(r"^\d+[.)]\s")


def parse_outline(text: str, topic: str) -> Outline:
    """`## Title` lines are sections, in order; the lines under one are its notes. In an outline
    whose sections are numbered, an unnumbered `## line` after a section is that section's note:
    a small model often writes its one-line notes with the same `## ` mark."""
    found: list[tuple[str, list[str]]] = []
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
            found.append((line[len(OUTLINE_SECTION_PREFIX) :].strip(), []))
        elif found and line and not line.startswith("#"):
            found[-1][1].append(line)
    numbered = sum(bool(_NUMBERED_RE.match(title)) for title, _ in found)
    sections: list[OutlineSection] = []
    for title, notes in found:
        if numbered >= 3 and sections and not _NUMBERED_RE.match(title):
            notes = [title, *notes]
            last = sections.pop()
            sections.append(OutlineSection(last.title, " ".join([last.notes, *notes]).strip()))
        else:
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


# the sections every guide of the series has; a small model leaves them out when it runs long
_ARC_CLOSING = (
    ("comparison", "Comparativa con alternativas", "Cuándo conviene cada opción y cómo elegir."),
    ("limitations", "Limitaciones y cuándo no usarla", "Dónde falla y qué errores son comunes."),
    ("summary", "Resumen y framework de decisión", "Síntesis y árbol de decisión de cierre."),
)


def ensure_closing_sections(outline: Outline, max_sections: int) -> Outline:
    """A comparison, the limitations and the summary with its decision tree, in that order at the
    end (ADR-061). Those the model wrote stay as it wrote them; those it left out are added, and
    the last body sections give them room when the outline is at its maximum."""
    wanted = [role for role, _, _ in _ARC_CLOSING]
    closing = {
        role: next((x for x in outline.sections if role_of(x.title) == role), None)
        for role in wanted
    }
    body = [x for x in outline.sections if role_of(x.title) not in ("limitations", "summary")]
    if closing["comparison"] is not None:  # a comparison stays where the model put it
        wanted.remove("comparison")
    tail = [closing[r] or OutlineSection(t, n) for r, t, n in _ARC_CLOSING if r in wanted]
    while len(body) + len(tail) > max_sections:
        spare = [i for i, x in enumerate(body) if role_of(x.title) != "comparison"]
        if not spare:
            break
        body.pop(spare[-1])
    return Outline(outline.topic, (*body, *tail), outline.siblings)


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
        translate: Callable[[str], str | None] | None = None,
    ) -> None:
        self._translate = translate
        self._translated: dict[str, str | None] = {}
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

    def _english(self, query: str) -> str | None:
        """The query in English when the books are (ADR-052), once per query."""
        if self._translate is None or in_english(query):
            return None
        if query not in self._translated:
            text = self._translate(query)
            self._translated[query] = text if text and text.casefold() != query.casefold() else None
        return self._translated[query]

    def _evidence(self, query: str) -> list[Hit]:
        """The query and its English twin, fused; from the books only when `draft.evidence` says
        so; at most `draft.per_document` fragments of one chapter (ADR-061)."""
        d, r = self._config.draft, self._config.retrieval
        queries = [query]
        if (english := self._english(query)) is not None:
            queries.append(english)
        pools = [self._retriever.search(q, r.final_k * 4) for q in queries]
        hits = rrf(pools, r.rrf_k) if len(pools) > 1 else pools[0]
        folder = self._config.library.books_folder
        if d.evidence == "books":
            hits = [h for h in hits if book_id(h.doc_path, folder) is not None]
        taken: Counter[str] = Counter()
        out = []
        for hit in hits:
            if taken[hit.doc_path] < d.per_document:
                taken[hit.doc_path] += 1
                out.append(hit)
        return out[: r.final_k]

    def _context(self, query: str, system: str, user_template: str, **fields: str) -> Context:
        hits = self._evidence(query)
        fixed = self._counter.count(system) + self._counter.count(
            user_template.format(evidence="", **fields)
        )
        cfg = self._config.llm
        return assemble(hits, self._counter, cfg.context_budget_tokens, cfg.max_fragments, fixed)

    def propose_outline(self, topic: str, on_token: Callable[[str], None] | None = None) -> Outline:
        d = self._config.draft
        # the style is the user's own guides: books are the evidence, never the model to imitate
        folder = self._config.library.books_folder
        own = [h for h in self._retriever.search(topic, 40) if book_id(h.doc_path, folder) is None]
        siblings = list(dict.fromkeys(h.doc_path for h in own))[: d.sibling_guides]
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
        with_closing = ensure_closing_sections(outline, d.max_sections)
        return Outline(topic, with_closing.sections, tuple(siblings))

    def frame(
        self, outline: Outline, results: Sequence[SectionResult], registry: SourceRegistry
    ) -> GuideMeta:
        """The epigraph and the key concepts, from what the sections ended up saying."""
        gist = "\n".join(
            f"- {s.title}: {_CITATION_GROUP_RE.sub('', s.body)[:300]}"
            for s in results
            if not s.missing_evidence
        )
        system = load_prompt("draft_frame_system")
        user = load_prompt("draft_frame_user").format(topic=outline.topic, gist=gist or "(nada)")
        reply = self._generate(system, user, None)
        return guide_meta(self._config, outline.topic, registry, _json_object(reply))

    def write_section(
        self,
        outline: Outline,
        section: OutlineSection,
        registry: SourceRegistry,
        on_token: Callable[[str], None] | None = None,
    ) -> SectionResult:
        role = role_of(section.title)
        system = (
            load_prompt("draft_section_system").format(missing=self.missing)
            + "\n\n"
            + load_prompt(f"draft_role_{role}")
        )
        user_template = load_prompt("draft_section_user")
        plan = "\n".join(f"- {s.title}" for s in outline.sections)
        notes = f"Notas: {section.notes}" if section.notes else ""
        fields = {"topic": outline.topic, "outline": plan, "section": section.title, "notes": notes}
        title = re.sub(r"^\s*\d+[.)]\s*", "", section.title)
        context = self._context(f"{outline.topic} {title}", system, user_template, **fields)
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


def _json_object(text: str) -> dict[str, object] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def guide_meta(
    config: Config, topic: str, registry: SourceRegistry, frame: dict[str, object] | None
) -> GuideMeta:
    """The header's fields: the area most of the sources come from, and the model's epigraph and
    key concepts when it gave them in the shape asked."""
    d = config.draft
    folder = config.library.books_folder
    areas = Counter(
        book_area(key, folder)
        for f in registry.fragments
        if (key := book_id(f.doc_path, folder)) is not None
    )
    area = areas.most_common(1)[0][0] if areas else "Reference"
    concepts: tuple[str, ...] = ()
    epigraph = ""
    if frame:
        raw = frame.get("conceptos")
        if isinstance(raw, list):
            concepts = tuple(str(c).strip() for c in raw if str(c).strip())[:4]
        text = frame.get("epigrafe")
        epigraph = text.strip() if isinstance(text, str) else ""
    return GuideMeta(d.author, area, f"{area} → {topic}", d.level, concepts, epigraph)


def strip_repeated_title(text: str, title: str) -> str:
    """Drop a leading heading that repeats the section title (the draft writes `## title`)."""
    lines = text.lstrip().splitlines()
    if lines and lines[0].lstrip().startswith("#"):
        heading = lines[0].lstrip("# ").strip().strip("*").strip()
        bare = re.sub(r"^\s*\d+[.)]\s*", "", title).strip().casefold()
        if heading.casefold() in (title.strip().casefold(), bare):
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
    topic: str,
    sections: Sequence[SectionResult],
    registry: SourceRegistry,
    today: date,
    meta: GuideMeta | None = None,
) -> str:
    """The draft in the series' shape (ADR-061); every claim keeps its [n] until `clean_draft`."""
    meta = meta or GuideMeta(series=topic)
    titles = [numbered_title(i, s.title) for i, s in enumerate(sections, start=1)]
    lines = render_header(
        topic, meta.track, meta.author, meta.series, meta.concepts, meta.level, today.year
    )
    lines += [
        "",
        f"<!-- Borrador generado por `biblio draft` el {today.isoformat()}: un esqueleto para",
        "     editar, no una guía terminada (ADR-008). Cada afirmación cita su fuente con [n];",
        "     revísala contra el libro antes de darla por buena. Al aceptarlo, quita las citas con",
        "     `biblio draft-clean`. -->",
    ]
    lines += render_epigraph(meta.epigraph or "[Epígrafe por escribir]")
    lines += render_toc(titles)
    for title, s in zip(titles, sections, strict=True):
        lines += [f"## {title}", "", s.body, ""]
        if s.uncited:
            lines += [
                "> ⚠ Esta sección no cita ninguna fuente: no la des por buena sin revisarla.",
                "",
            ]
        if s.invalid_citations:
            lines += ["> ⚠ El modelo citó fragmentos inexistentes, marcados como [?].", ""]
    lines += [SOURCES_TITLE, ""]
    for n, fragment in enumerate(registry.fragments, start=1):
        lines.append(f"{n}. {fragment.citation} (líneas {fragment.start_line}-{fragment.end_line})")
    if not registry.fragments:
        lines.append("(ninguna)")
    lines += ["", *render_footer(meta.author)]
    return "\n".join(lines) + "\n"


_CITES_RE = re.compile(r"[ \t]*(?:\[(?:\d+|\?)\])+")
_BANNER_RE = re.compile(r"<!-- Borrador generado.*?-->\n?", re.DOTALL)
_SOURCE_ITEM_RE = re.compile(r"^\d+\. (.*?) \(líneas \d+-\d+\)\s*$")


def clean_draft(markdown: str) -> str:
    """The draft as the series publishes it: no [n], no review flags, no banner, and the
    references as a list of the chapters used instead of numbered fragments."""
    out: list[str] = []
    in_sources = False
    seen: set[str] = set()
    for line in _BANNER_RE.sub("", markdown).splitlines():
        if line.startswith(OUTLINE_SECTION_PREFIX):
            in_sources = line.strip() == SOURCES_TITLE
        if line.startswith("> ⚠"):
            continue
        if in_sources and (m := _SOURCE_ITEM_RE.match(line)):
            if m.group(1) not in seen:
                seen.add(m.group(1))
                out.append(f"- {m.group(1)}")
            continue
        out.append(_CITES_RE.sub("", line) if not in_sources else line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).rstrip() + "\n"
