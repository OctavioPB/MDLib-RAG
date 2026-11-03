"""The shape of a guide in the user's OPB Reference Series (ADR-061).

A guide has six blocks, in this order: header, epigraph, table of contents, numbered sections,
sources and footer. The checker below tells whether a Markdown file has them; the renderers write
them. The order of the sections follows an arc (a conceptual frame, foundations, mechanics,
complexity, a comparison, applications, limitations and a decision tree) that the outline asks for
and the section prompts know by role (`role_of`).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

SERIES_LINE = "### OPB Reference Series"
FOOTER_SERIES = "**OPB Reference Series**"
FOOTER_TAGLINE = "*From Pipeline to Decision.*"
TOC_TITLE = "## 0. Tabla de Contenidos"
SOURCES_TITLE = "## Referencias"

_SECTION_RE = re.compile(r"^## (\d+)\.\s+(.+?)\s*$", re.MULTILINE)
_TABLE_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", re.MULTILINE)
_NUMBER_RE = re.compile(r"^\s*\d+[.)]\s*")
_EPIGRAPH_RE = re.compile(r'^> \*"[\s\S]+?"\*\s*\n> — OPB', re.MULTILINE)
_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def fold(text: str) -> str:
    plain = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in plain if not unicodedata.combining(ch))


# --- the role of a section, read from its title -------------------------------------------

# (role, words of the title that name it); the first that matches wins
_ROLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("limitations", ("limitacion", "cuando no", "cuando falla", "donde falla", "errores comunes")),
    ("summary", ("resumen", "framework de decision", "arbol de decision", "cuando usar")),
    ("comparison", ("comparativa", "comparacion", " vs", "versus", "alternativas")),
    ("complexity", ("complejidad", "costos", "rendimiento")),
    ("applications", ("aplicacion", "casos de uso", "uso en")),
    ("concept", ("marco conceptual", "que es", "conceptual")),
)


def role_of(title: str) -> str:
    """What a section is for, from its title: the prompts ask each role for its own shape."""
    name = f" {fold(title)}"
    for role, words in _ROLES:
        if any(w in name for w in words):
            return role
    return "body"


# --- the checker ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class StructureReport:
    checks: tuple[tuple[str, bool], ...]
    sections: int

    @property
    def passed(self) -> int:
        return sum(ok for _, ok in self.checks)

    @property
    def score(self) -> float:
        return self.passed / len(self.checks)

    def failed(self) -> list[str]:
        return [name for name, ok in self.checks if not ok]


def check_structure(
    markdown: str, min_sections: int = 10, max_sections: int = 13
) -> StructureReport:
    """Which of the series' blocks a guide has. Eleven yes/no checks, each one a visible part of
    the style; whether the content is faithful is for the reader (and the citations), not here."""
    lines = markdown.strip().splitlines()
    sections = _SECTION_RE.findall(markdown)
    titles = [t for _, t in sections]
    last_body = markdown.split(TOC_TITLE, 1)[-1]
    numbered = [m for m in _SECTION_RE.finditer(markdown) if int(m.group(1)) > 0]
    closing = ""
    if numbered:
        end = markdown.find("\n## ", numbered[-1].end())
        closing = markdown[numbered[-1].end() : end if end > 0 else len(markdown)]
    blocks = [b.lower() for b in _FENCE_RE.findall(closing)]
    tail = [ln for ln in lines if ln.strip()][-2:]
    checks = (
        ("title", bool(lines) and lines[0].startswith("# ")),
        ("series line", any(ln.startswith(SERIES_LINE) for ln in lines[:6])),
        ("metadata", all(k in markdown[:900] for k in ("**Serie:**", "**Tema:**", "**Nivel:**"))),
        ("epigraph", bool(_EPIGRAPH_RE.search(markdown))),
        (
            "table of contents",
            TOC_TITLE in markdown and bool(re.search(r"\]\(#[^)]+\)", last_body)),
        ),
        ("10-13 sections", min_sections <= len(numbered) <= max_sections),
        ("table", bool(_TABLE_RE.search(markdown))),
        ("comparison", any(role_of(t) == "comparison" for t in titles)),
        ("limitations", any(role_of(t) == "limitations" for t in titles)),
        ("decision tree", any(("si" in b or "sí" in b) and "no" in b for b in blocks)),
        (
            "footer",
            len(tail) == 2 and FOOTER_SERIES in tail[0] and tail[1].strip() == FOOTER_TAGLINE,
        ),
    )
    return StructureReport(checks, len(numbered))


# --- the renderers -------------------------------------------------------------------------------


def anchor(title: str) -> str:
    """The GitHub / Obsidian-style anchor of a heading."""
    plain = re.sub(r"[^\w\s-]", "", title.casefold())
    return re.sub(r"\s+", "-", plain.strip())


def numbered_title(index: int, title: str) -> str:
    """«3. Algo»; a number the outline already wrote is replaced by its place."""
    return f"{index}. {_NUMBER_RE.sub('', title).strip()}"


def render_header(
    topic: str, track: str, author: str, series: str, concepts: Sequence[str], level: str, year: int
) -> list[str]:
    lines = [f"# {topic}", f"{SERIES_LINE} — {track}", ""]
    if author:
        lines.append(f"**Autor:** {author}")
    lines += [
        f"**Serie:** {series}",
        f"**Tema:** {' · '.join(concepts) if concepts else topic}",
        f"**Nivel:** {level}",
        f"**Versión:** 1.0 · {year}",
    ]
    return lines


def render_epigraph(text: str) -> list[str]:
    quote = " ".join(text.split()).strip().strip('"“”«»*')
    return ["", "---", "", f'> *"{quote}"*', "> — OPB", "", "---", ""]


def render_toc(titles: Sequence[str]) -> list[str]:
    lines = [TOC_TITLE, ""]
    lines += [f"{i}. [{t}](#{anchor(t)})" for i, t in enumerate(titles, start=1)]
    return [*lines, ""]


def render_footer(author_line: str) -> list[str]:
    name = author_line.replace(" · ", " — ", 1)
    return ["---", "", f"{FOOTER_SERIES} · {name}".rstrip(" ·"), FOOTER_TAGLINE]
