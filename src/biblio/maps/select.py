"""Which sections a concept map reads (ADR-051, stage 1): those that explain the topic first.

A one-word topic searched as written attracts sections that only mention it (a project's `api/`
folder, a glossary, an installation guide). The candidates are the bare topic's and a
definitional question's, fused; the reranker orders them against the question; sections whose
heading names the topic go first. Measured on 8 topics labelled general / specific / incidental
(`eval/maps_centrality.jsonl`): precision of the sections read 0.44 → 0.70, contamination
0.21 → 0.08, coverage of the general sections 0.45 → 0.69.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from biblio.config import MapsConfig
from biblio.generate.context import Fragment
from biblio.maps.build import topic_in_text
from biblio.retrieve.fusion import rrf
from biblio.retrieve.rerank import passage
from biblio.retrieve.types import Hit

Search = Callable[[str, int], list[Hit]]
Score = Callable[[str, Sequence[str]], list[float]]


def definitional_query(topic: str, cfg: MapsConfig) -> str:
    return cfg.define_query.format(topic=topic)


def map_candidates(search: Search, topic: str, cfg: MapsConfig, rrf_k: int) -> list[Hit]:
    """The bare topic's and the definitional question's first-stage hits, fused (RRF)."""
    bare = search(topic, cfg.candidates)
    defined = search(definitional_query(topic, cfg), cfg.candidates)
    return rrf([bare, defined], rrf_k)


def order_sections(topic: str, hits: Sequence[Hit], scores: Sequence[float] | None) -> list[Hit]:
    """By reranker score when there is one, then those whose heading names the topic first
    (stable: each group keeps its order). The guide's title does not count as a heading."""
    ranked = list(hits)
    if scores is not None:
        ranked = [h for _, h in sorted(zip(scores, ranked, strict=True), key=lambda p: -p[0])]
    named = [h for h in ranked if topic_in_text(topic, " ".join(h.heading_path[1:]))]
    return named + [h for h in ranked if h not in named]


def rerank_scores(score: Score, topic: str, hits: Sequence[Hit], cfg: MapsConfig) -> list[float]:
    return score(definitional_query(topic, cfg), [passage(h) for h in hits])


def count_general(
    topic: str,
    fragments: Sequence[Fragment],
    scores: dict[tuple[str, int], float],
    threshold: float,
    focus: Callable[[Fragment], str | None],
) -> int:
    """How many fragments explain the topic in general (ADR-051 stage 3): the reranker scores
    them against the definitional question at `threshold` or more, or their heading names the
    topic and the LLM (`focus`, asked only then) calls them general. Measured on the labelled
    pool: 0.78 accuracy, against 0.62 for the LLM alone (it calls almost everything general)."""
    count = 0
    for f in fragments:
        scored = scores.get((f.doc_path, f.start_line), 0.0) >= threshold
        named = topic_in_text(topic, " ".join(f.heading_path[1:]))
        count += scored or (named and focus(f) == "general")  # the LLM only when it can decide
    return count
