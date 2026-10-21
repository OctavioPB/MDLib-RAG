"""Search results grouped by note (ADR-045).

A big note answers a query with many of its sections; in 30 of the 64 evaluation questions one
note filled three or more of the top 8. Grouped, the page shows one line per note (its best
section) with the others a click away, and asks for more hits so the same lines hold more notes.
"""

from __future__ import annotations

from collections.abc import Sequence

from biblio.retrieve.types import Hit


def group_by_note(hits: Sequence[Hit], notes: int) -> list[list[Hit]]:
    """The hits by note, notes in the order of their best hit, each note's hits in rank order;
    at most `notes` groups."""
    groups: dict[str, list[Hit]] = {}
    for hit in hits:
        if hit.doc_path not in groups:
            if len(groups) == notes:
                continue
            groups[hit.doc_path] = []
        groups[hit.doc_path].append(hit)
    return list(groups.values())
