"""Domain objects and interfaces of the retrieval layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from biblio.ingest.types import BREADCRUMB_SEP


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    doc_path: str
    heading_path: tuple[str, ...]
    text: str
    start_line: int
    end_line: int
    rank: int  # 1-based
    score: float  # higher is better; comparable only within one retriever
    source: str  # "dense", "bm25", ...

    @property
    def breadcrumb(self) -> str:
        return BREADCRUMB_SEP.join(self.heading_path)

    @property
    def citation(self) -> str:
        """`path/file.md › Section › Subsection` (CLAUDE.md, rule 2)."""
        sections = BREADCRUMB_SEP.join(self.heading_path[1:]) or self.heading_path[0]
        return f"{self.doc_path} › {sections}"


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int) -> list[Hit]: ...

    def close(self) -> None: ...
