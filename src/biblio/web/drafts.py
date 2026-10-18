"""The drafts folder as the Guías page sees it (ADR-019): the same files `biblio draft` writes.

`<slug>.outline.md` is the editable outline and `<slug>.md` the written draft. Only names made by
`slugify` are accepted, so a request can never reach a file outside the drafts folder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from biblio.generate.draft import OUTLINE_SECTION_PREFIX, Outline, parse_outline

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,59}$")
OUTLINE_SUFFIX = ".outline.md"
_OUTLINE_TITLE = "# Esquema: "
_SOURCES_HEADING = "## Fuentes"
_SOURCE_LINE_RE = re.compile(r"^\d+\. ")


@dataclass(frozen=True)
class DraftSummary:
    slug: str
    topic: str
    has_outline: bool
    has_draft: bool
    updated: str  # ISO date-time of the newest of its files
    sections: int  # in the draft if written, else in the outline
    sources: int


@dataclass(frozen=True)
class DraftDetail:
    slug: str
    topic: str
    outline: Outline | None
    markdown: str | None


def valid_slug(slug: str) -> bool:
    return bool(SLUG_RE.fullmatch(slug))


def outline_path(drafts_dir: Path, slug: str) -> Path:
    return drafts_dir / f"{slug}{OUTLINE_SUFFIX}"


def draft_path(drafts_dir: Path, slug: str) -> Path:
    return drafts_dir / f"{slug}.md"


def _topic_from_outline(text: str) -> str | None:
    first = text.lstrip().splitlines()[0] if text.strip() else ""
    return first[len(_OUTLINE_TITLE) :].strip() if first.startswith(_OUTLINE_TITLE) else None


def _topic_from_draft(text: str) -> str | None:
    first = text.lstrip().splitlines()[0] if text.strip() else ""
    return first[2:].strip() if first.startswith("# ") else None


def draft_counts(markdown: str) -> tuple[int, int]:
    """(sections, sources) of a written draft."""
    sections = sources = 0
    in_sources = False
    for line in markdown.splitlines():
        if line.startswith(OUTLINE_SECTION_PREFIX):
            in_sources = line.strip() == _SOURCES_HEADING
            sections += not in_sources
        elif in_sources and _SOURCE_LINE_RE.match(line):
            sources += 1
    return sections, sources


def read_draft(drafts_dir: Path, slug: str) -> DraftDetail | None:
    if not valid_slug(slug):
        return None
    o_path, d_path = outline_path(drafts_dir, slug), draft_path(drafts_dir, slug)
    o_text = o_path.read_text(encoding="utf-8") if o_path.is_file() else None
    d_text = d_path.read_text(encoding="utf-8") if d_path.is_file() else None
    if o_text is None and d_text is None:
        return None
    topic = (
        (o_text and _topic_from_outline(o_text)) or (d_text and _topic_from_draft(d_text)) or slug
    )
    outline = parse_outline(o_text, topic) if o_text is not None else None
    return DraftDetail(slug, topic, outline, d_text)


def list_drafts(drafts_dir: Path) -> list[DraftSummary]:
    """Every guide in the drafts folder, newest first."""
    if not drafts_dir.is_dir():
        return []
    slugs = set()
    for path in drafts_dir.glob("*.md"):
        name = path.name
        slug = name[: -len(OUTLINE_SUFFIX)] if name.endswith(OUTLINE_SUFFIX) else name[:-3]
        if valid_slug(slug):
            slugs.add(slug)
    summaries = []
    for slug in slugs:
        detail = read_draft(drafts_dir, slug)
        if detail is None:
            continue
        files = [p for p in (outline_path(drafts_dir, slug), draft_path(drafts_dir, slug))]
        newest = max(p.stat().st_mtime for p in files if p.is_file())
        if detail.markdown is not None:
            sections, sources = draft_counts(detail.markdown)
        else:
            sections, sources = len(detail.outline.sections) if detail.outline else 0, 0
        summaries.append(
            DraftSummary(
                slug=slug,
                topic=detail.topic,
                has_outline=detail.outline is not None,
                has_draft=detail.markdown is not None,
                updated=datetime.fromtimestamp(newest).astimezone().isoformat(timespec="seconds"),
                sections=sections,
                sources=sources,
            )
        )
    return sorted(summaries, key=lambda s: s.updated, reverse=True)
