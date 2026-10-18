"""Candidate alternatives for the questions a run missed, for the *user* to judge.

Pooling for relevance judgment: the retriever proposes, a human decides. Nothing here edits the
question set: the skill forbids Claude (or the tool) from deciding what counts as a right answer.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from biblio.eval.metrics import matches_section
from biblio.eval.questions import Question
from biblio.retrieve.types import Hit, Retriever


@dataclass(frozen=True)
class Review:
    question: Question
    hits: tuple[Hit, ...]  # top hits that are not already an expected section


def missed_questions(
    record: dict[str, Any], questions: Sequence[Question], k: int
) -> list[Question]:
    """Questions of the run whose expected sections all stayed out of the top-k."""
    by_id = {q.id: q for q in questions}
    missed = []
    for qid, data in record["per_question"].items():
        ranks = [r for r in data["section_ranks"] if r is not None]
        if qid in by_id and (not ranks or min(ranks) > k):
            missed.append(by_id[qid])
    return missed


def collect(retriever: Retriever, questions: Sequence[Question], top: int) -> list[Review]:
    reviews = []
    for q in questions:
        hits = retriever.search(q.question, top + len(q.expected))
        fresh = [h for h in hits if not any(matches_section(h, e) for e in q.expected)]
        reviews.append(Review(q, tuple(fresh[:top])))
    return reviews


def to_markdown(reviews: Sequence[Review], tag: str, mode: str) -> str:
    lines = [
        f"# Candidate alternatives — run `{tag}` ({mode})",
        "",
        "For each missed question, the sections the search returned instead. If one of them",
        "**really answers the question**, copy its JSON line into that expected item's",
        '`"alternatives": [...]` in `eval/questions.jsonl`. Leave the rest out: an alternative',
        "that only mentions the topic makes the evaluation lenient, not fair.",
        "",
    ]
    for review in reviews:
        q = review.question
        lines += [f"## {q.id} ({q.type}) — {q.question}", "", "Expected:"]
        lines += [f"- `{e.path}` › {e.heading}" for e in q.expected]
        lines += ["", "Returned instead:", ""]
        for hit in review.hits:
            excerpt = " ".join(
                ln.strip() for ln in hit.text.splitlines() if ln.strip() and not ln.startswith("#")
            )[:160]
            location = {"path": hit.doc_path, "heading": hit.heading_path[-1]}
            lines += [
                f"{hit.rank}. `{hit.citation}`  ",
                f"   {excerpt}…",
                "",
                "   ```json",
                f"   {json.dumps(location, ensure_ascii=False)}",
                "   ```",
                "",
            ]
    return "\n".join(lines) + "\n"
