"""The built-in reader (ADR-031): a note as a tree of typed blocks and inline nodes.

The note is parsed with the indexer's own `markdown-it-py` setup, so every block carries the same
source lines the chunks and citations use. The page builds the DOM from this tree with
`textContent`; nothing here is HTML.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from markdown_it.token import Token

from biblio.ingest.links import WIKILINK_RE, markdown_link_target
from biblio.ingest.parser import _markdown, parse_text

Node = dict[str, Any]
Resolve = Callable[[str], "str | None"]  # a link target (`name.md` or a path) → an indexed note

_EXTERNAL = re.compile(r"^https?://", re.IGNORECASE)
_CONTAINERS = {
    "bullet_list": "ul",
    "ordered_list": "ol",
    "list_item": "li",
    "blockquote": "blockquote",
    "table": "table",
    "thead": "thead",
    "tbody": "tbody",
    "tr": "tr",
}
_INLINE_SPANS = {"strong", "em", "s", "link"}


def note_tree(text: str, doc_path: str, resolve: Resolve, depth: int = 3) -> dict[str, Any]:
    """{title, blocks, outline, words}: blocks with `lines` [first, last] (1-based, like chunks)."""
    doc = parse_text(text, doc_path)
    body = "\n".join(doc.lines[doc.body_start :]) + "\n"
    tokens = _markdown().parse(body)
    blocks, _ = _blocks(tokens, 0, doc.body_start, doc_path, resolve, None)
    return {
        "title": doc.title,
        "blocks": blocks,
        "outline": outline(blocks, depth, doc.title),
        "words": len(body.split()),
    }


# Headings that are a line of code, not a title: exports that turned `# import numpy` comments into
# headings. Kept in the text as the note writes them, left out of the reader's outline.
_CODE_HEADING = re.compile(
    r"^(?:import\s|from\s+[\w.]+\s+import\s|def\s|class\s+\w+\s*[(:]|return\b|print\s*\(|"
    r"for\s+\w+(?:\s*,\s*\w+)*\s+in\s|if\s.*:$|elif\s|else\s*:|while\s|with\s.*:$|try\s*:|except\b|"
    r"raise\s|assert\s|lambda\b|async\s+def\s|await\s|const\s|let\s|var\s|function\s|"
    r"pip\s+install\s|conda\s|npm\s|\$\s|>>>|"
    # x = f(…), df['a'] = [...], n += 1; «EBITDA = EBIT + D&A» is a formula, kept
    r"[\w.\[\]'\"]+\s*[-+*/]?=(?!=)\s*(?:.*[(\[]|[-\d.]+$|['\"])|"
    r"[\w.]+\.\w+\(.*\)\s*$|\w+\(.*\)\s*$)"  # plt.show(), model.fit(x, y), main()
)


def _looks_like_code(text: str) -> bool:
    return bool(_CODE_HEADING.match(text.strip()))


def _plain(inlines: list[Node], code: bool = True) -> str:
    """The text of inline nodes; `code=False` leaves out code spans."""
    out = []
    for n in inlines:
        if n.get("t") == "code":
            out.append(n["text"] if code else " ")
        elif "children" in n:
            out.append(_plain(n["children"], code))
        elif n.get("t") == "math":
            out.append(n["tex"])
        else:
            out.append(str(n.get("text", "")))
    return "".join(out)


def outline(blocks: list[Node], depth: int, title: str = "") -> list[Node]:
    """The note's sections for the reader's index: {level, text, line}, top `depth` levels.

    The note's own H1 is the page title, so the outline starts below it, also when some text
    comes before it. A heading whose written
    text (code spans aside: `pandas.read_csv` is a real title) is a line of code is left out.
    """
    heads = [b for b in blocks if b["type"] == "heading" and b.get("lines")]
    if heads and heads[0]["level"] == 1 and blocks and blocks[0] is heads[0]:
        heads = heads[1:]
    heads = [h for h in heads if not (h["level"] == 1 and _plain(h["inlines"]).strip() == title)]
    out: list[Node] = []
    for h in heads:
        text = " ".join(_plain(h["inlines"]).split())
        prose = " ".join(_plain(h["inlines"], code=False).split())
        if not text or (prose and _looks_like_code(prose)):
            continue
        out.append({"level": h["level"], "text": text, "line": h["lines"][0]})
    if not out:
        return out
    top = min(h["level"] for h in out)
    return [h for h in out if h["level"] < top + depth]


def _lines(token: Token, offset: int) -> list[int] | None:
    if token.map is None:
        return None
    start, end = token.map
    return [start + offset + 1, max(end + offset, start + offset + 1)]


def _blocks(
    tokens: list[Token],
    i: int,
    offset: int,
    doc_path: str,
    resolve: Resolve,
    close: str | None,
) -> tuple[list[Node], int]:
    """Blocks from `tokens[i]` until the `close` token type; returns them and the next index."""
    out: list[Node] = []
    while i < len(tokens):
        tok = tokens[i]
        if close is not None and tok.type == close:
            return out, i + 1
        kind = tok.type.removesuffix("_open")
        if tok.type.endswith("_open") and kind in _CONTAINERS:
            children, i = _blocks(tokens, i + 1, offset, doc_path, resolve, f"{kind}_close")
            node: Node = {"type": _CONTAINERS[kind], "lines": _lines(tok, offset)}
            if kind == "ordered_list" and tok.attrs.get("start"):
                node["start"] = tok.attrs["start"]
            node["children"] = children
            out.append(node)
            continue
        if tok.type == "heading_open":
            inline = tokens[i + 1]
            out.append(
                {
                    "type": "heading",
                    "level": int(tok.tag[1]),
                    "lines": _lines(tok, offset),
                    "title": inline.content.strip(),  # as in a chunk's heading path (ADR-050)
                    "inlines": _inlines(inline.children or [], doc_path, resolve),
                }
            )
            i += 3
            continue
        if tok.type in ("paragraph_open", "th_open", "td_open"):
            inline = tokens[i + 1]
            node = {
                "type": {"paragraph_open": "p", "th_open": "th", "td_open": "td"}[tok.type],
                "lines": _lines(tok, offset),
                "inlines": _inlines(inline.children or [], doc_path, resolve),
            }
            align = tok.attrs.get("style")
            if isinstance(align, str) and "text-align:" in align:
                node["align"] = align.split("text-align:", 1)[1].strip(" ;")
            out.append(node)
            i += 3
            continue
        if tok.type in ("fence", "code_block"):
            info = tok.info.strip().split(" ", 1)[0] if tok.type == "fence" else ""
            out.append(
                {"type": "code", "lang": info, "text": tok.content, "lines": _lines(tok, offset)}
            )
        elif tok.type in ("math_block", "math_block_label"):
            out.append({"type": "math", "tex": tok.content, "lines": _lines(tok, offset)})
        elif tok.type == "html_block":
            out.append(
                {"type": "code", "lang": "html", "text": tok.content, "lines": _lines(tok, offset)}
            )
        elif tok.type == "hr":
            out.append({"type": "hr", "lines": _lines(tok, offset)})
        i += 1
    return out, i


def _inlines(children: list[Token], doc_path: str, resolve: Resolve) -> list[Node]:
    root: list[Node] = []
    stack = [root]
    for tok in children:
        kind = tok.type
        if kind == "text":
            stack[-1].extend(_text_nodes(tok.content, resolve))
        elif kind == "softbreak":
            stack[-1].append({"t": "text", "text": "\n"})
        elif kind == "hardbreak":
            stack[-1].append({"t": "br"})
        elif kind == "code_inline":
            stack[-1].append({"t": "code", "text": tok.content})
        elif kind in ("math_inline", "math_inline_double"):
            stack[-1].append({"t": "math", "tex": tok.content, "display": kind.endswith("double")})
        elif kind == "html_inline":
            stack[-1].append({"t": "code", "text": tok.content})
        elif kind == "image":
            src = str(tok.attrs.get("src", ""))
            alt = "".join(c.content for c in tok.children or []) or src
            stack[-1].append(
                {"t": "image", "alt": alt, "src": src}
            )  # shown by the reader (ADR-048)
        elif kind.endswith("_open") and kind.removesuffix("_open") in _INLINE_SPANS:
            node: Node = {"t": kind.removesuffix("_open"), "children": []}
            if node["t"] == "link":
                node.update(_link(str(tok.attrs.get("href", "")), doc_path, resolve))
            stack[-1].append(node)
            stack.append(node["children"])
        elif kind.endswith("_close") and kind.removesuffix("_close") in _INLINE_SPANS:
            if len(stack) > 1:
                stack.pop()
        elif tok.content:
            stack[-1].append({"t": "text", "text": tok.content})
    return root


def _link(href: str, doc_path: str, resolve: Resolve) -> Node:
    if _EXTERNAL.match(href):
        return {"href": href}
    target = markdown_link_target(href, doc_path)
    return {"note": resolve(target) if target else None}


# what browsers draw; any other extension is checked by its first bytes (ADR-048)
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "webp", "avif", "bmp", "svg"}


def _text_nodes(text: str, resolve: Resolve) -> list[Node]:
    """Plain text with `[[target|alias]]` turned into links to the notes they name."""
    out: list[Node] = []
    last = 0
    for match in WIKILINK_RE.finditer(text):
        embed = match.start() > 0 and text[match.start() - 1] == "!"  # Obsidian's ![[…]]
        before = text[last : match.start() - 1] if embed else text[last : match.start()]
        if before:
            out.append({"t": "text", "text": before})
        inner = match.group(1)
        target, _, alias = inner.partition("|")
        name = target.split("#", 1)[0].strip()
        label = alias.strip() or target.strip()
        if embed and name.rsplit(".", 1)[-1].lower() in IMAGE_EXTENSIONS:
            # ![[figure.png|300]]: an image found by name (ADR-048); a size alias is not alt text
            alt = alias.strip() if alias.strip() and not alias.strip().isdigit() else name
            out.append({"t": "image", "alt": alt, "embed": name})
            last = match.end()
            continue
        if name and not name.lower().endswith(".md"):
            name += ".md"
        out.append({"t": "wikilink", "text": label, "note": resolve(name) if name else None})
        last = match.end()
    if last < len(text):
        out.append({"t": "text", "text": text[last:]})
    return out
