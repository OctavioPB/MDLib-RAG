"""Tables of contents are never search results (ADR-044).

Almost every note opens with a section «Índice» / «Tabla de contenidos» (2,172 in the library): a
list of every heading of the note, so it matches any term those headings use, and competes with
the very section it points to. Only the heading decides: a short introduction merged with its
index by the chunker keeps real content and stays searchable.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import replace
from typing import Any

from biblio.retrieve.types import Hit, Retriever

NAVIGATION_HEADINGS = frozenset(
    {
        "indice",
        "indice de contenidos",
        "indice de contenido",
        "tabla de contenido",
        "tabla de contenidos",
        "contenido",
        "contenidos",
        "table of contents",
        "contents",
        "toc",
    }
)


def _plain(text: str) -> str:
    """Lower case, no accents, numbering or punctuation: «0. Tabla de Contenidos» → ..."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().casefold()
    return " ".join(re.sub(r"^[\d.\s]+|[^\w\s]", " ", ascii_text).split())


def is_navigation(heading_path: tuple[str, ...]) -> bool:
    """A section whose own heading names it a table of contents (not the note's title)."""
    return len(heading_path) > 1 and _plain(heading_path[-1]) in NAVIGATION_HEADINGS


class SkipNavigation:
    """A retriever without tables of contents: asks its base for `margin` more hits and drops
    them, so `k` results still come back."""

    def __init__(self, base: Retriever, margin: int) -> None:
        self._base = base
        self._margin = margin
        self.name = base.name

    def search(self, query: str, k: int) -> list[Hit]:
        hits = [
            h
            for h in self._base.search(query, k + self._margin)
            if not is_navigation(h.heading_path)
        ]
        return [replace(h, rank=rank) for rank, h in enumerate(hits[:k], start=1)]

    def close(self) -> None:
        self._base.close()

    def __getattr__(self, name: str) -> Any:  # reranker_device etc., read by `biblio eval`
        return getattr(self._base, name)
