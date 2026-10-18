"""Markdown → `ParsedDocument`: frontmatter, title and top-level blocks with line ranges.

Headings, code, tables and math are recognized from `markdown-it-py` tokens, never with regexes
(ADR-003): `# comment` inside a fence is not a heading, setext headings are.
"""

from __future__ import annotations

import logging
import re
from functools import cache
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from markdown_it import MarkdownIt
from markdown_it.token import Token
from mdit_py_plugins.dollarmath import dollarmath_plugin

from biblio.ingest.links import markdown_link_target, wikilink_targets
from biblio.ingest.types import Block, ParsedDocument

log = logging.getLogger(__name__)

_LINE_BREAK_RE = re.compile(r"\r\n|\r|\n")
_FRONTMATTER_OPEN = "---"
_FRONTMATTER_CLOSE = ("---", "...")
_ATOMIC_KINDS = frozenset(
    {"fence", "code_block", "table", "math_block", "math_block_label", "html_block"}
)
_LIST_KINDS = frozenset({"bullet_list", "ordered_list"})


@cache
def _markdown() -> MarkdownIt:
    return MarkdownIt("commonmark").enable("table").use(dollarmath_plugin)


def parse_file(path: Path, doc_path: str) -> ParsedDocument:
    return parse_bytes(path.read_bytes(), doc_path)


def parse_bytes(raw: bytes, doc_path: str) -> ParsedDocument:
    """Parse already-read bytes (the indexer hashes and parses the same read)."""
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        log.warning("%s is not valid UTF-8; undecodable bytes replaced", doc_path)
        text = raw.decode("utf-8-sig", errors="replace")
    return parse_text(text, doc_path)


def parse_text(text: str, doc_path: str) -> ParsedDocument:
    lines = _LINE_BREAK_RE.split(text)
    if lines and lines[-1] == "":
        lines.pop()  # the terminator of the last line does not start a new one

    metadata, body_start = _frontmatter(lines, doc_path)
    body = "\n".join(lines[body_start:]) + "\n"
    blocks = tuple(_blocks(_markdown().parse(body), body_start, doc_path))
    _warn_unclosed_fences(blocks, lines, doc_path)
    return ParsedDocument(
        doc_path=doc_path,
        title=_title(metadata, blocks, doc_path),
        metadata=metadata,
        lines=tuple(lines),
        body_start=body_start,
        blocks=blocks,
    )


def _frontmatter(lines: list[str], doc_path: str) -> tuple[dict[str, Any], int]:
    if not lines or lines[0].strip() != _FRONTMATTER_OPEN:
        return {}, 0
    close = next((i for i in range(1, len(lines)) if lines[i].strip() in _FRONTMATTER_CLOSE), None)
    if close is None:
        return {}, 0  # a lone `---` is a thematic break, not frontmatter
    try:
        data = yaml.safe_load("\n".join(lines[1:close]))
    except yaml.YAMLError as exc:
        log.warning("%s: invalid YAML frontmatter ignored (%s)", doc_path, exc)
        data = None
    if data is not None and not isinstance(data, dict):
        log.warning("%s: frontmatter is not a mapping; ignored", doc_path)
        data = None
    return (data or {}), close + 1


def _title(metadata: dict[str, Any], blocks: tuple[Block, ...], doc_path: str) -> str:
    title = metadata.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    for block in blocks:
        if block.heading_level == 1:
            return block.title
    return PurePosixPath(doc_path).stem


def _blocks(tokens: list[Token], offset: int, doc_path: str) -> list[Block]:
    """Top-level blocks. Lists yield one block per item, so long lists can split between items."""
    blocks: list[Block] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.level != 0 or token.map is None or token.nesting == -1:
            i += 1
            continue
        kind = token.type.removesuffix("_open")
        if kind in _LIST_KINDS:
            blocks += _list_items(tokens, i, offset, doc_path)
            i = _closing_index(tokens, i) + 1
            continue
        end = _closing_index(tokens, i)
        links = _links(tokens[i : end + 1], doc_path)
        level, title = 0, ""
        if kind == "heading":
            title = tokens[i + 1].content.strip()
            if title:
                level = int(token.tag[1:])
            else:
                kind = "empty_heading"  # carries no name; treated as plain content
        blocks.append(_block(kind, token.map, offset, links, level, title))
        i = end + 1
    return blocks


def _list_items(tokens: list[Token], i: int, offset: int, doc_path: str) -> list[Block]:
    items = []
    end = _closing_index(tokens, i)
    j = i + 1
    while j < end:
        token = tokens[j]
        if token.type == "list_item_open" and token.level == 1 and token.map is not None:
            item_end = _closing_index(tokens, j)
            links = _links(tokens[j : item_end + 1], doc_path)
            items.append(_block("list_item", token.map, offset, links))
            j = item_end + 1
        else:
            j += 1
    return items


def _block(
    kind: str,
    line_map: list[int],
    offset: int,
    links: tuple[str, ...],
    heading_level: int = 0,
    title: str = "",
) -> Block:
    return Block(
        kind=kind,
        start=line_map[0] + offset,
        end=line_map[1] + offset,
        atomic=kind in _ATOMIC_KINDS,
        heading_level=heading_level,
        title=title,
        links=links,
    )


def _warn_unclosed_fences(blocks: tuple[Block, ...], lines: list[str], doc_path: str) -> None:
    """An unclosed fence turns the rest of the note into code (Obsidian renders it the same way)."""
    for block in blocks:
        if block.kind != "fence":
            continue
        opening = lines[block.start].strip()
        fence_char = opening[0]
        fence_len = len(opening) - len(opening.lstrip(fence_char))
        closing = lines[block.end - 1].strip()
        closed = (
            block.end - block.start >= 2
            and closing.startswith(fence_char * fence_len)
            and not closing.strip(fence_char)
        )
        if not closed:
            log.warning(
                "%s:%d: code fence is never closed; everything after it is treated as code",
                doc_path,
                block.start + 1,
            )


def _closing_index(tokens: list[Token], i: int) -> int:
    """Index of the token that closes `tokens[i]` (itself for self-contained tokens)."""
    if tokens[i].nesting != 1:
        return i
    depth = 0
    for j in range(i, len(tokens)):
        depth += tokens[j].nesting
        if depth == 0:
            return j
    return len(tokens) - 1


def _links(tokens: list[Token], doc_path: str) -> tuple[str, ...]:
    found: list[str] = []
    for token in tokens:
        if token.type != "inline" or not token.children:
            continue
        # Text children only: links inside inline code are not links.
        text = "".join(c.content for c in token.children if c.type == "text")
        found += wikilink_targets(text)
        for child in token.children:
            href = child.attrs.get("href") if child.type == "link_open" else None
            if isinstance(href, str) and (target := markdown_link_target(href, doc_path)):
                found.append(target)
    return tuple(dict.fromkeys(found))
