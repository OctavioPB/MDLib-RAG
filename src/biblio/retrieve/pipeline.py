"""Builds the retriever for a mode.

Each mode adds one piece over the previous ones so every piece can be measured on its own
(PLAN Phase 4): dense → bm25 → hybrid (RRF) → + reranker.
"""

from __future__ import annotations

from collections.abc import Callable

from biblio.config import RETRIEVAL_MODES, Config
from biblio.index.embedder import Embedder, load_query_embedder
from biblio.retrieve.bm25 import BM25Retriever
from biblio.retrieve.dense import DenseRetriever
from biblio.retrieve.fusion import HybridRetriever
from biblio.retrieve.navigation import SkipNavigation
from biblio.retrieve.rerank import (
    Reranker,
    RerankingRetriever,
    RerankPlan,
    load_reranker,
    resolve_rerank_plan,
)
from biblio.retrieve.store import check_query_embedder, open_index
from biblio.retrieve.types import Retriever

MODES = RETRIEVAL_MODES


def build_retriever(
    config: Config,
    mode: str,
    embedder_factory: Callable[[Config], Embedder] = load_query_embedder,
    reranker_factory: Callable[[Config, str], Reranker] = load_reranker,
    plan_resolver: Callable[[Config], RerankPlan] = resolve_rerank_plan,
) -> Retriever:
    if mode not in MODES:
        raise ValueError(f"Unknown retrieval mode {mode!r}; expected one of {', '.join(MODES)}")
    conn = open_index(config)
    base_mode, _, rerank = mode.partition("-")
    if base_mode != "bm25":
        check_query_embedder(conn, config)

    base: Retriever
    r = config.retrieval
    if base_mode == "bm25":
        base = BM25Retriever(conn, heading_weight=r.bm25_heading_weight)
    elif base_mode == "dense":
        base = DenseRetriever(conn, embedder_factory(config))
    else:
        dense = DenseRetriever(conn, embedder_factory(config))
        base = HybridRetriever(dense, BM25Retriever(conn, heading_weight=r.bm25_heading_weight), r)
    if r.skip_navigation:  # before reranking: no candidate slot is spent on an index (ADR-044)
        base = SkipNavigation(base, r.navigation_margin)

    if rerank:
        plan = plan_resolver(config)
        if plan.device is None:
            return base  # GPU not free: plain first stage (ADR-014); its name says so
        return RerankingRetriever(
            base, reranker_factory(config, plan.device), plan.candidates, plan.device
        )
    return base
