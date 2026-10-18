"""Domain objects of the ingestion layer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

BREADCRUMB_SEP = " › "


@dataclass(frozen=True)
class Block:
    """A top-level Markdown block, located by 0-based line indexes in the file (`end` exclusive)."""

    kind: str  # markdown-it token type without `_open`: "paragraph", "fence", "heading", ...
    start: int
    end: int
    atomic: bool  # never split inside (code, tables, math, HTML)
    heading_level: int = 0  # > 0 only for headings
    title: str = ""  # heading text
    links: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedDocument:
    doc_path: str  # relative to the library root, "/" separator
    title: str  # root of every heading_path
    metadata: Mapping[str, Any]  # YAML frontmatter, empty if absent or invalid
    lines: tuple[str, ...]  # whole file, frontmatter included, without line terminators
    body_start: int  # 0-based index of the first line after the frontmatter
    blocks: tuple[Block, ...]


@dataclass(frozen=True)
class Chunk:
    doc_path: str
    ordinal: int  # 0-based position within the document
    heading_path: tuple[str, ...]
    text: str  # original content, without the breadcrumb
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    token_count: int  # embedding-tokenizer tokens of `text`
    links: tuple[str, ...]

    @property
    def breadcrumb(self) -> str:
        return BREADCRUMB_SEP.join(self.heading_path)

    @property
    def embedding_text(self) -> str:
        return f"{self.breadcrumb}\n\n{self.text}"
