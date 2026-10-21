"""Lexical retrieval: BM25 over FTS5 with accent folding (ADR-005, ADR-006).

Natural-language queries ("¿Qué es RRF k=60?") are not valid FTS5 syntax, so each word becomes a
quoted term and the terms are OR-ed: BM25's IDF then weighs rare terms over common ones. There is no
Spanish stemmer and none is emulated (CLAUDE.md): dense search covers morphology.
"""

from __future__ import annotations

import itertools
import json
import re
import sqlite3

from biblio.retrieve.store import load_hits, scope_chunk_ids_json
from biblio.retrieve.types import Hit

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def fts_query(query: str) -> str | None:
    """`¿Qué es RRF k=60?` → `"qué" OR "es" OR "rrf" OR "k" OR "60"`; None if no words."""
    words = list(dict.fromkeys(w.lower() for w in _WORD_RE.findall(query)))
    if not words:
        return None
    return " OR ".join(f'"{w}"' for w in words)


class BM25Retriever:
    name = "bm25"

    def __init__(
        self, conn: sqlite3.Connection, scope: str | None = None, heading_weight: float = 1.0
    ) -> None:
        """`scope` limits the search to one library folder (ADR-019); None = the whole library.
        `heading_weight` weighs the heading path (title › sections) against the text (ADR-044)."""
        self._conn = conn
        self._scope = scope
        self._rank = f"bm25(chunks_fts, {float(heading_weight)!r}, 1.0)"

    def search(self, query: str, k: int) -> list[Hit]:
        match = fts_query(query)
        if match is None:
            return []
        if self._scope is None:
            rows = self._conn.execute(
                f"SELECT rowid, {self._rank} FROM chunks_fts WHERE chunks_fts MATCH ? "
                f"ORDER BY {self._rank} LIMIT ?",
                (match, k),
            ).fetchall()
        else:
            # Rank every match, then keep the folder's: a `rowid IN (...)` constraint makes FTS5
            # test the match chunk by chunk (26 s for a 10k-chunk folder vs 0.08 s this way).
            in_scope = set(json.loads(scope_chunk_ids_json(self._conn, self._scope)))
            ranked = self._conn.execute(
                f"SELECT rowid, {self._rank} FROM chunks_fts WHERE chunks_fts MATCH ? "
                f"ORDER BY {self._rank}",
                (match,),
            )
            rows = list(itertools.islice((r for r in ranked if r[0] in in_scope), k))
        # FTS5's bm25() is lower-is-better; negate so higher is better.
        return load_hits(self._conn, [(rowid, -score) for rowid, score in rows], self.name)

    def close(self) -> None:
        pass
