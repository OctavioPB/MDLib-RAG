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


def note_tree(text: str, doc_path: str, resolve: Resolve) -> dict[str, Any]:
    """{title, blocks}: blocks with `lines` [first, last] (1-based, like chunks)."""
    doc = parse_text(text, doc_path)
    body = "\n".join(doc.lines[doc.body_start :]) + "\n"
    tokens = _markdown().parse(body)
    blocks, _ = _blocks(tokens, 0, doc.body_start, doc_path, resolve, None)
    return {"title": doc.title, "blocks": blocks}


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
            alt = "".join(c.content for c in tok.children or []) or str(tok.attrs.get("src", ""))
            stack[-1].append({"t": "image", "alt": alt})
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


def _text_nodes(text: str, resolve: Resolve) -> list[Node]:
    """Plain text with `[[target|alias]]` turned into links to the notes they name."""
    out: list[Node] = []
    last = 0
    for match in WIKILINK_RE.finditer(text):
        if match.start() > last:
            out.append({"t": "text", "text": text[last : match.start()]})
        inner = match.group(1)
        target, _, alias = inner.partition("|")
        name = target.split("#", 1)[0].strip()
        label = alias.strip() or target.strip()
        if name and not name.lower().endswith(".md"):
            name += ".md"
        out.append({"t": "wikilink", "text": label, "note": resolve(name) if name else None})
        last = match.end()
    if last < len(text):
        out.append({"t": "text", "text": text[last:]})
    return out
