"""Dense retrieval: exact cosine search over bge-m3 vectors with sqlite-vec (ADR-004, ADR-005)."""

from __future__ import annotations

import sqlite3

from biblio.index.embedder import Embedder
from biblio.retrieve.store import load_hits, scope_chunk_ids_json
from biblio.retrieve.types import Hit


class DenseRetriever:
    name = "dense"

    def __init__(
        self, conn: sqlite3.Connection, embedder: Embedder, scope: str | None = None
    ) -> None:
        """`scope` limits the search to one library folder (ADR-019); None = the whole library."""
        self._conn = conn
        self._embedder = embedder
        self._scope = scope

    def search(self, query: str, k: int) -> list[Hit]:
        vector = self._embedder.embed([query])[0]
        if self._scope is None:
            rows = self._conn.execute(
                "SELECT chunk_id, distance FROM vec_chunks WHERE embedding MATCH ? AND k = ? "
                "ORDER BY distance",
                (vector.tobytes(), k),
            ).fetchall()
        else:
            # sqlite-vec restricts the exact KNN scan to the given ids (~0.15 s for 10k chunks).
            rows = self._conn.execute(
                "SELECT chunk_id, distance FROM vec_chunks WHERE embedding MATCH ? AND k = ? "
                "AND chunk_id IN (SELECT value FROM json_each(?)) ORDER BY distance",
                (vector.tobytes(), k, scope_chunk_ids_json(self._conn, self._scope)),
            ).fetchall()
        # cosine distance → cosine similarity, so higher is better like every other score
        return load_hits(self._conn, [(cid, 1.0 - dist) for cid, dist in rows], self.name)

    def close(self) -> None:
        self._embedder.close()
