"""Cross-encoder reranking of the best candidates (ADR-006); bge-reranker-v2-m3 (ADR-013).

The reranker reads each (query, chunk) pair jointly, which is far more precise than comparing two
independent embeddings, and far slower: it only sees the top `rerank_candidates` of a first stage.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from biblio.config import Config
from biblio.models import local_model_path, model_specs
from biblio.offline import loopback_port_open
from biblio.retrieve.types import Hit, Retriever

log = logging.getLogger(__name__)


class Reranker(Protocol):
    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Relevance of each passage to the query; higher is better."""
        ...

    def close(self) -> None: ...


class CrossEncoderReranker:
    """A sentence-transformers cross-encoder loaded from `models/`, never from the Hub."""

    def __init__(self, config: Config, device: str) -> None:
        from sentence_transformers import CrossEncoder

        spec = next(s for s in model_specs(config) if s.role == "reranker")
        path = local_model_path(spec, config.paths.models_dir)
        self._model: Any = CrossEncoder(
            str(path),
            device=device,
            max_length=config.reranker.max_tokens,
            local_files_only=True,
        )
        self._batch_size = config.reranker.batch_size
        self._device = device

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        scores = self._model.predict(
            [(query, p) for p in passages], batch_size=self._batch_size, show_progress_bar=False
        )
        return [float(s) for s in scores]

    def close(self) -> None:
        del self._model
        gc.collect()
        if self._device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()


def load_reranker(config: Config, device: str) -> CrossEncoderReranker:
    return CrossEncoderReranker(config, device)


@dataclass(frozen=True)
class RerankPlan:
    device: str | None  # None = skip the reranker (plain first stage)
    candidates: int
    fallback_reason: str = ""  # why the preferred GPU is not used; empty when it is


def _cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def resolve_rerank_plan(
    config: Config,
    cuda_available: Callable[[], bool] = _cuda_available,
    llm_running: Callable[[], bool] | None = None,
) -> RerankPlan:
    """GPU when free; otherwise skip the reranker, or rerank on CPU if configured (ADR-014)."""
    r = config.reranker
    if not r.device.startswith("cuda"):
        return RerankPlan(r.device, r.cpu_candidates)
    if llm_running is None:

        def llm_running() -> bool:
            return loopback_port_open(config.llm.base_url, config.llm.probe_timeout_s)

    reason = ""
    if not cuda_available():
        reason = "CUDA is not available"
    elif llm_running():
        reason = f"an LLM server is running on {config.llm.base_url}"
    if not reason:
        return RerankPlan(r.device, config.retrieval.rerank_candidates)
    if r.fallback == "cpu":
        log.warning("Reranker on CPU with %d candidates: %s (ADR-014)", r.cpu_candidates, reason)
        return RerankPlan("cpu", r.cpu_candidates, reason)
    log.warning("Reranker skipped, plain hybrid ranking: %s (ADR-014)", reason)
    return RerankPlan(None, 0, reason)


def passage(hit: Hit) -> str:
    """What the reranker reads: the breadcrumb gives the chunk its context, as in the embedding."""
    return f"{hit.breadcrumb}\n\n{hit.text}"


class RerankingRetriever:
    def __init__(
        self, base: Retriever, reranker: Reranker, candidates: int, device: str = ""
    ) -> None:
        self._base = base
        self._reranker = reranker
        self._candidates = candidates
        self.name = f"{base.name}-rerank"
        self.reranker_device = device  # recorded by `biblio eval`
        self.rerank_candidates = candidates

    def search(self, query: str, k: int) -> list[Hit]:
        """Rerank only the first `candidates` hits; deeper hits (asked for by `biblio eval` at
        k=20) follow in first-stage order, so the top-k is exactly what `search` returns."""
        first_stage = self._base.search(query, max(k, self._candidates))
        head, tail = first_stage[: self._candidates], first_stage[self._candidates :]
        scores = self._reranker.score(query, [passage(h) for h in head])
        order = sorted(range(len(head)), key=lambda i: -scores[i])
        reranked = [replace(head[i], score=scores[i], source=self.name) for i in order]
        return [replace(hit, rank=rank) for rank, hit in enumerate([*reranked, *tail][:k], start=1)]

    def close(self) -> None:
        self._base.close()
        self._reranker.close()
