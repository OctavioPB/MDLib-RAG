"""Hybrid retrieval: Reciprocal Rank Fusion of dense and BM25 rankings (ADR-006).

RRF uses only ranks, so dense cosine similarities and BM25 scores never need to be calibrated
against each other: score(d) = Σ 1 / (k + rank_i(d)) over the rankings that contain d.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from biblio.config import RetrievalConfig
from biblio.retrieve.types import Hit, Retriever


def rrf(rankings: Sequence[Sequence[Hit]], k: int) -> list[Hit]:
    """Fuse rankings into one, best first. Ties keep the order of first appearance."""
    scores: dict[int, float] = {}
    first_seen: dict[int, Hit] = {}
    for ranking in rankings:
        for hit in ranking:
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + hit.rank)
            first_seen.setdefault(hit.chunk_id, hit)
    order = sorted(scores, key=lambda cid: -scores[cid])  # stable: preserves first-seen order
    return [
        replace(first_seen[cid], rank=rank, score=scores[cid], source="hybrid")
        for rank, cid in enumerate(order, start=1)
    ]


class HybridRetriever:
    name = "hybrid"

    def __init__(self, dense: Retriever, bm25: Retriever, config: RetrievalConfig) -> None:
        self._dense = dense
        self._bm25 = bm25
        self._config = config

    def search(self, query: str, k: int) -> list[Hit]:
        dense_hits = self._dense.search(query, max(k, self._config.dense_top_n))
        bm25_hits = self._bm25.search(query, max(k, self._config.bm25_top_n))
        return rrf([dense_hits, bm25_hits], self._config.rrf_k)[:k]

    def close(self) -> None:
        self._dense.close()
        self._bm25.close()
