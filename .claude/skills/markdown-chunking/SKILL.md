---
name: markdown-chunking
description: Rules and procedure for parsing and chunking the library's Markdown documents. Use it ALWAYS when working in src/biblio/ingest/, modifying the parser or the chunker, adjusting the [chunking] thresholds in config, investigating a search result that looks cut off or misplaced, or adding ingestion test fixtures — even if the change looks small.
---

# Structural Markdown chunking

Search quality is bounded by chunk quality: a fragment cut in the middle of a derivation or separated
from its heading is unrecoverable for the reranker and misleading for the LLM.
Base decision: ADR-003.

## Data model

```python
@dataclass(frozen=True)
class Chunk:
    doc_path: str  # relative to the library root, "/" separator
    ordinal: int  # position within the document
    heading_path: tuple[str, ...]  # ("Guide title", "Section", "Subsection")
    text: str  # original content, without the breadcrumb
    start_line: int  # 1-based, inclusive
    end_line: int
    token_count: int  # tokens of the embedding tokenizer, not words
    links: tuple[str, ...]  # normalized internal link targets
```

The embedded text is `" › ".join(heading_path) + "\n\n" + text`. The breadcrumb is not stored inside
`text`, because it is shown and cited separately.

## Algorithm

1. **Frontmatter.** If the file starts with `---`, parse YAML up to the next `---`. A `title` in the
   frontmatter takes priority as the root of `heading_path`. If there is none, use the first H1; if there
   is none either, the file name without extension. Invalid YAML does not stop ingestion: log a warning and continue.
2. **Tokenize with `markdown-it-py`** (`MarkdownIt("commonmark").enable("table")`). Use `token.map`
   for line numbers. Do not use regexes to detect headings: `# comment` inside a code block is not a
   heading, and setext headings (`Title\n=====`) are.
3. **Build sections.** Walk the tokens keeping a stack of headings by level. A level-n heading closes all
   open sections of level ≥ n. Content before the first heading forms a section with `heading_path = (root,)`.
4. **Atomic units.** Never split inside: fenced or indented code blocks, tables, `$$ … $$` blocks, HTML
   blocks, or ASCII diagrams (code blocks without a language or with `text`). If an atomic unit alone
   exceeds `max_tokens`, it becomes its own oversized chunk and is logged; do not split it.
5. **Long sections** (> `max_tokens`): split at paragraph or atomic-unit boundaries, accumulating up to
   `target_tokens`. All sub-chunks inherit the same `heading_path`. Do not use overlap: the breadcrumb
   already provides the context that overlap tries to recover.
6. **Tiny sections** (< `min_tokens`): merge with the next sibling section. If it is the last one, merge
   with the previous one. A heading without a body (that only groups subsections) does not produce its own
   chunk. When merging, the `heading_path` of the result is the common ancestor, so the citation does not lie.
7. **Internal links.** Extract `[[target]]`, `[[target|alias]]` and Markdown links whose target ends in
   `.md` (with or without `#anchor`). Normalize to a path relative to the root. Ignore URLs with a scheme (`http:`, `https:`).

## Versioning

`CHUNKER_VERSION` lives in `src/biblio/ingest/chunker.py`. Increment the version on any change that
alters the chunks produced (default thresholds included). The indexer refuses to mix versions and
requires `--full` (CLAUDE.md, rule 6).

## Tests

- Every pathological case has a fixture in `tests/fixtures/` and a test that checks the number of chunks,
  the `heading_path`s and that no atomic block is split.
- Minimum cases: file without headings; frontmatter only; code with `#` lines; setext heading;
  level jumps (H1 → H3); huge table; multiline `$$`; duplicate headings in the same guide;
  tiny section at the end; empty document.
- Property test: the concatenation of a document's `text`s (in order) covers the whole body, without loss
  or duplication, except for the frontmatter.
- When a chunking bug shows up in the evaluation, first write the fixture that reproduces it and then the fix.

## Implementation notes (chunker v1)

Choices made where the algorithm above leaves room; changing any of them changes the chunks, so it
requires bumping `CHUNKER_VERSION`.

- **Code map:** `ingest/discovery.py` (globs, symlinks) → `ingest/parser.py` (frontmatter, title,
  top-level `Block`s with line ranges) → `ingest/chunker.py` (sections, merge, split, line ranges).
  Token counts come from `ingest/tokens.py` (bge-m3 `tokenizer.json`, special tokens excluded).
- **`$$` blocks** are parsed by `mdit-py-plugins`' `dollarmath`, so a blank line inside a formula
  does not split it.
- **Chunks are line ranges.** Each chunk owns every line from its first block up to the next chunk's
  first block, trimmed of blank edge lines. Lines that produce no tokens (link reference definitions)
  are therefore never lost, and `text` is always an exact slice of the file.
- **Heading lines stay in `text`** (they help BM25); the breadcrumb is still kept out of it.
- **The H1 that names the guide** (or a heading equal to the frontmatter title) is not repeated in
  `heading_path`.
- **A bodyless heading** is carried into the next section, which keeps its own, more specific path.
  At the end of the document it joins the previous section.
- **Empty headings** (`#` alone) are plain content, not section boundaries.
- **Lists** are split into one block per top-level item, so a long list splits between items.
  Blockquotes are one block.
- **Headings and lead-ins are glued** to the block that follows them when a long section is split:
  a heading, or a paragraph/list item ending in `:` ("… se calcula como:", "**Ejemplo:**"), always
  travels with what it introduces.
- **No split produces a fragment:** a piece is only closed once it reaches `min_tokens`, and a
  trailing piece under `min_tokens` folds into the previous one. Hence a chunk below `min_tokens`
  only exists when it is the whole document (property-tested).
- **Unclosed code fences** are logged with their line: they turn the rest of the note into code.
- **Wikilinks** are stored as written plus `.md` (`[[Bayes]]` → `Bayes.md`), not resolved to a path:
  they resolve by name, which is the link graph's job (Phase 7). Links inside inline code are ignored.
- **Discovery** matches globs case-insensitively; `x/**` exclusions also prune the walk.

## Manual inspection

`biblio chunks <file>` must show each chunk with its ordinal, line range, tokens and breadcrumb.
Use it to review by eye before closing changes in this layer.
