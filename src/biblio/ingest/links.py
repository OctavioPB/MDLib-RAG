"""Internal link extraction and normalization (used by the link graph in Phase 7)."""

from __future__ import annotations

import posixpath
import re
from urllib.parse import unquote, urlsplit

WIKILINK_RE = re.compile(r"\[\[([^\[\]]+?)\]\]")
_MD_SUFFIX = ".md"


def wikilink_targets(text: str) -> list[str]:
    """`[[target]]`, `[[target|alias]]`, `[[target#heading]]` → `target.md`.

    Wikilinks are resolved by name, not by path, so they are kept as written (plus `.md`); matching
    them to a document is left to the link graph.
    """
    targets = []
    for match in WIKILINK_RE.finditer(text):
        target = match.group(1).split("|", 1)[0].split("#", 1)[0].strip()
        if not target:
            continue
        if not target.lower().endswith(_MD_SUFFIX):
            target += _MD_SUFFIX
        targets.append(target)
    return targets


def markdown_link_target(href: str, doc_path: str) -> str | None:
    """Normalize a Markdown link to a `.md` file into a path relative to the library root.

    Returns None for URLs with a scheme, non-Markdown targets and paths that escape the root.
    """
    parts = urlsplit(href)
    if parts.scheme or parts.netloc:
        return None
    path = unquote(parts.path)
    if not path.lower().endswith(_MD_SUFFIX):
        return None
    if path.startswith("/"):
        normalized = posixpath.normpath(path.lstrip("/"))
    else:
        normalized = posixpath.normpath(posixpath.join(posixpath.dirname(doc_path), path))
    if normalized == ".." or normalized.startswith("../"):
        return None
    return normalized
