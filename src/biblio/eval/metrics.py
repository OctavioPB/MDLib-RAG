"""Retrieval metrics at file and section granularity (`retrieval-eval` skill).

A hit matches an expected *section* when it is from the expected file and either its heading path
ends with the expected heading, or the chunk contains that heading as a Markdown heading line (a
tiny section merged into a neighbor is cited by the common ancestor but still contains its text).
Each expected item is credited once, at its best rank, so nDCG never exceeds 1 when a section was
split into several chunks.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from biblio.eval.questions import Expected, Location
from biblio.retrieve.types import Hit

_HEADING_LINE_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
_SPACES_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Case-, accent- and whitespace-insensitive form for matching user-typed headings and paths."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _SPACES_RE.sub(" ", stripped).strip().casefold()


def matches_file(hit: Hit, expected: Expected) -> bool:
    """The hit is from the expected file or from the file of any alternative."""
    return any(_in_file(hit, option) for option in expected.options)


def matches_section(hit: Hit, expected: Expected) -> bool:
    """The hit is the expected section or any alternative section."""
    return any(matches_location(hit, option) for option in expected.options)


def _in_file(hit: Hit, location: Location) -> bool:
    return normalize(hit.doc_path) == normalize(location.path)


def matches_location(hit: Hit, location: Location) -> bool:
    if not _in_file(hit, location):
        return False
    target = normalize(location.heading)
    if hit.heading_path and normalize(hit.heading_path[-1]) == target:
        return True
    for line in hit.text.splitlines():
        m = _HEADING_LINE_RE.match(line)
        if m and normalize(m.group(1)) == target:
            return True
    return False


@dataclass(frozen=True)
class QuestionScore:
    """Ranks are 1-based positions in the hit list; None = not retrieved within the list."""

    first_rank: int | None  # first hit matching any expected item
    expected_ranks: tuple[int | None, ...]  # best rank of each expected item (files deduplicated)


def score_question(
    hits: Sequence[Hit], expected: Sequence[Expected], granularity: str
) -> QuestionScore:
    if granularity == "section":
        items = list(expected)
        match = matches_section
    elif granularity == "file":
        # Expected sections with the same acceptable files are one expected file.
        items = list(
            {frozenset(normalize(o.path) for o in e.options): e for e in expected}.values()
        )
        match = matches_file
    else:
        raise ValueError(f"unknown granularity {granularity!r}")
    best: list[int | None] = [None] * len(items)
    for hit in hits:
        for i, item in enumerate(items):
            if best[i] is None and match(hit, item):
                best[i] = hit.rank
    found = [r for r in best if r is not None]
    return QuestionScore(first_rank=min(found) if found else None, expected_ranks=tuple(best))


def recall_at(score: QuestionScore, k: int) -> float:
    ranks = score.expected_ranks
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def all_found_at(score: QuestionScore, k: int) -> float:
    return float(all(r is not None and r <= k for r in score.expected_ranks))


def reciprocal_rank(score: QuestionScore) -> float:
    return 1.0 / score.first_rank if score.first_rank else 0.0


def ndcg_at(score: QuestionScore, k: int) -> float:
    """Binary relevance; each expected item contributes once, at its best rank."""
    dcg = sum(1.0 / math.log2(r + 1) for r in score.expected_ranks if r is not None and r <= k)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(score.expected_ranks), k)))
    return dcg / ideal if ideal else 0.0
