"""Context assembly for grounded generation (`rag-prompting` skill).

Fragments keep the retriever's order and are numbered [1]..[n]. Hits that are contiguous in the same
document are joined first, so a section split in two chunks reads as one fragment. The token budget
is counted with the LLM's own tokenizer; a fragment that does not fit is dropped whole, never cut.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from biblio.ingest.tokens import TokenCounter
from biblio.retrieve.types import Hit

# Blank lines between two chunks of the same document still count as contiguous.
_MAX_LINE_GAP = 2


@dataclass(frozen=True)
class Fragment:
    number: int  # [n] as shown to the LLM
    doc_path: str
    heading_path: tuple[str, ...]
    text: str
    start_line: int
    end_line: int

    @property
    def citation(self) -> str:
        sections = " › ".join(self.heading_path[1:]) or self.heading_path[0]
        return f"{self.doc_path} › {sections}"

    def render(self) -> str:
        lines = f"líneas {self.start_line}-{self.end_line}"
        return f"[{self.number}] {self.citation} ({lines})\n{self.text}"


@dataclass(frozen=True)
class Context:
    fragments: tuple[Fragment, ...]
    tokens: int
    dropped: int  # hits left out by the budget or the fragment limit
    over_budget: int = 0  # of those, left out because they did not fit the token budget

    def render(self) -> str:
        return "\n\n".join(f.render() for f in self.fragments)


def merge_contiguous(hits: Sequence[Hit]) -> list[Hit]:
    """Join hits from the same document whose line ranges touch, keeping the first one's place."""
    merged: list[Hit] = []
    for hit in hits:
        for i, prev in enumerate(merged):
            if prev.doc_path != hit.doc_path:
                continue
            first, second = (prev, hit) if prev.start_line <= hit.start_line else (hit, prev)
            if second.start_line - first.end_line <= _MAX_LINE_GAP:
                merged[i] = Hit(
                    chunk_id=prev.chunk_id,
                    doc_path=prev.doc_path,
                    heading_path=first.heading_path,
                    text=f"{first.text}\n\n{second.text}",
                    start_line=first.start_line,
                    end_line=max(first.end_line, second.end_line),
                    rank=prev.rank,
                    score=prev.score,
                    source=prev.source,
                )
                break
        else:
            merged.append(hit)
    return merged


def assemble(
    hits: Sequence[Hit],
    counter: TokenCounter,
    budget_tokens: int,
    max_fragments: int,
    fixed_tokens: int,
    merge: bool = True,
) -> Context:
    """Fill `budget_tokens` (the prompt itself costs `fixed_tokens`) with whole fragments.
    `merge=False` keeps the hits as given (a pinned passage and its section, ADR-031)."""
    candidates = merge_contiguous(hits) if merge else list(hits)
    fragments: list[Fragment] = []
    used = fixed_tokens
    over_budget = 0
    for hit in candidates:
        if len(fragments) == max_fragments:
            break
        fragment = Fragment(
            number=len(fragments) + 1,
            doc_path=hit.doc_path,
            heading_path=hit.heading_path,
            text=hit.text,
            start_line=hit.start_line,
            end_line=hit.end_line,
        )
        cost = counter.count(fragment.render()) + 2  # the blank line between fragments
        if used + cost > budget_tokens:
            over_budget += 1
            continue  # too big: try the next, smaller fragment instead of cutting this one
        fragments.append(fragment)
        used += cost
    return Context(tuple(fragments), used, len(candidates) - len(fragments), over_budget)
