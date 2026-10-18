"""Structural chunker (ADR-003): sections by heading, long ones split by blocks, tiny ones merged.

Chunks partition the document body by line ranges: every line after the frontmatter belongs to
exactly one chunk (blank lines at chunk edges are trimmed), so nothing is lost or duplicated.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from biblio.config import ChunkingConfig
from biblio.ingest.tokens import TokenCounter
from biblio.ingest.types import Block, Chunk, ParsedDocument

# Bump on any change that alters the chunks produced, default thresholds included.
# The index refuses to mix chunker versions (CLAUDE.md, rule 6).
CHUNKER_VERSION = 1

log = logging.getLogger(__name__)

_Tokens = Callable[[list[Block]], int]


@dataclass
class _Section:
    path: tuple[str, ...]
    blocks: list[Block]


def chunk_document(
    doc: ParsedDocument, counter: TokenCounter, config: ChunkingConfig
) -> list[Chunk]:
    if not doc.blocks:
        return []
    sizes = {b.start: counter.count(_lines(doc, b.start, b.end)) for b in doc.blocks}

    def tokens(blocks: list[Block]) -> int:
        return sum(sizes[b.start] for b in blocks)

    sections = _absorb_bodyless(_sections(doc))
    sections = _merge_tiny(sections, tokens, config.min_tokens)
    pieces = [piece for s in sections for piece in _split(doc, s, tokens, config)]
    return _materialize(doc, pieces, counter)


def _lines(doc: ParsedDocument, start: int, end: int) -> str:
    return "\n".join(doc.lines[start:end])


def _sections(doc: ParsedDocument) -> list[_Section]:
    """One section per heading, plus the content before the first heading."""
    root = doc.title
    stack: list[tuple[int, str]] = []
    current = _Section((root,), [])
    sections = [current]
    for block in doc.blocks:
        if block.heading_level:
            while stack and stack[-1][0] >= block.heading_level:
                stack.pop()
            stack.append((block.heading_level, block.title))
            names = [name for _, name in stack]
            if names[0] == root:
                names = names[1:]  # the H1 that names the guide is already the root
            current = _Section((root, *names), [block])
            sections.append(current)
        else:
            current.blocks.append(block)
    return [s for s in sections if s.blocks]


def _has_body(section: _Section) -> bool:
    return any(not b.heading_level for b in section.blocks)


def _absorb_bodyless(sections: list[_Section]) -> list[_Section]:
    """A heading without body only groups subsections: its line travels with the next section."""
    out: list[_Section] = []
    carried: list[Block] = []
    for section in sections:
        if not _has_body(section):
            carried += section.blocks
            continue
        section.blocks = carried + section.blocks
        carried = []
        out.append(section)
    if carried:
        if out:
            out[-1].blocks += carried
        else:
            out.append(_Section((sections[0].path[0],), carried))
    return out


def _common_path(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[str, ...]:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return a[:n]


def _merge_tiny(sections: list[_Section], tokens: _Tokens, min_tokens: int) -> list[_Section]:
    """Tiny sections merge into the next one (the last one into the previous one); the merged
    chunk is cited by the common ancestor so the citation does not lie."""
    out = list(sections)
    i = 0
    while i < len(out) - 1:
        if tokens(out[i].blocks) < min_tokens:
            nxt = out[i + 1]
            out[i + 1] = _Section(_common_path(out[i].path, nxt.path), out[i].blocks + nxt.blocks)
            del out[i]
        else:
            i += 1
    if len(out) > 1 and tokens(out[-1].blocks) < min_tokens:
        last = out.pop()
        prev = out[-1]
        out[-1] = _Section(_common_path(prev.path, last.path), prev.blocks + last.blocks)
    return out


def _split(
    doc: ParsedDocument, section: _Section, tokens: _Tokens, config: ChunkingConfig
) -> list[_Section]:
    """Split a long section between blocks, accumulating up to `target_tokens`, no overlap."""
    if tokens(section.blocks) <= config.max_tokens:
        return [section]

    units: list[list[Block]] = []
    for block in section.blocks:
        # Headings and lead-in sentences ("... se calcula como:") stay with what they introduce.
        if units and _introduces_next(doc, units[-1][-1]):
            units[-1].append(block)
        else:
            units.append([block])

    pieces: list[list[Block]] = []
    current: list[Block] = []
    for unit in units:
        size = tokens(unit)
        if size > config.max_tokens:
            log.warning(
                "%s:%d: %s of %d tokens exceeds max_tokens=%d; kept whole",
                doc.doc_path,
                unit[-1].start + 1,
                unit[-1].kind,
                size,
                config.max_tokens,
            )
        # A piece is closed only once it has reached min_tokens, so no split produces a fragment.
        if (
            current
            and tokens(current) >= config.min_tokens
            and tokens(current) + size > config.target_tokens
        ):
            pieces.append(current)
            current = []
        current = current + unit
    pieces.append(current)

    if len(pieces) > 1 and tokens(pieces[-1]) < config.min_tokens:
        pieces[-2:] = [pieces[-2] + pieces[-1]]
    return [_Section(section.path, piece) for piece in pieces]


def _introduces_next(doc: ParsedDocument, block: Block) -> bool:
    if block.heading_level:
        return True
    if block.kind not in ("paragraph", "list_item"):
        return False
    last_line = doc.lines[block.end - 1].rstrip().rstrip("*_")
    return last_line.endswith(":")


def _materialize(doc: ParsedDocument, pieces: list[_Section], counter: TokenCounter) -> list[Chunk]:
    chunks: list[Chunk] = []
    for i, piece in enumerate(pieces):
        # Each chunk owns every line up to the next chunk's first block, so lines without tokens
        # (link reference definitions, blank lines) are never lost.
        start = doc.body_start if i == 0 else piece.blocks[0].start
        end = pieces[i + 1].blocks[0].start if i + 1 < len(pieces) else len(doc.lines)
        while start < end and not doc.lines[start].strip():
            start += 1
        while end > start and not doc.lines[end - 1].strip():
            end -= 1
        if start == end:
            continue
        text = _lines(doc, start, end)
        links = tuple(dict.fromkeys(link for b in piece.blocks for link in b.links))
        chunks.append(
            Chunk(
                doc_path=doc.doc_path,
                ordinal=len(chunks),
                heading_path=piece.path,
                text=text,
                start_line=start + 1,
                end_line=end,
                token_count=counter.count(text),
                links=links,
            )
        )
    return chunks
