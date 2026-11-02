"""Summaries of a chapter or note and of a whole book, made on demand and kept (ADR-056, stage E).

The library holds ~190 M tokens (17,816 book chapters, 2,736 notes): reading it all ahead took ~11 h
on the 4B and ~5 days on the «Razonar» model, so a summary is made the first time it is asked for
and kept in `understanding.db`, keyed by the text's hash, so a changed note is summarized again.

- A document: its sections in order, in windows of `window_tokens`; each window gives a short
  summary and up to `part_points` key points, each with the number of its section and a literal
  quote checked with `practice.feynman.contains` (a point whose quote is not in its section is
  dropped); several windows are then joined into one summary by one more call.
- A book: its chapters' summaries (missing ones are made on the way), then one overview citing
  chapters by number; a number with no chapter is reported, as the tutor's citations are.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from biblio.config import UnderstandingConfig
from biblio.generate.ask import load_prompt
from biblio.generate.citations import extract_citations
from biblio.generate.context import Fragment
from biblio.generate.llm import LLMClient
from biblio.ingest.tokens import TokenCounter
from biblio.practice.feynman import contains
from biblio.practice.quiz import _complete, _json_object

CHAPTER = "chapter"
BOOK = "book"
_MAX_TEXT = 500  # characters of a summary point or a quote


@dataclass(frozen=True)
class Point:
    text: str
    heading: str  # the section's own title
    start_line: int
    end_line: int
    quote: str  # literal text of that section


@dataclass(frozen=True)
class ChapterLine:
    path: str
    title: str
    overview: str


@dataclass(frozen=True)
class Summary:
    kind: str  # chapter (a book's chapter or a note) | book
    key: str  # the document's path or the book's id
    title: str
    source_hash: str  # the document's SHA-256, or a hash of the chapters' ones
    model: str
    overview: str
    points: tuple[Point, ...] = ()  # a document's key points
    chapters: tuple[ChapterLine, ...] = ()  # a book's chapters, in order
    invalid: tuple[int, ...] = ()  # chapter numbers the book overview cites that do not exist
    created: str = ""


def book_hash(chapter_hashes: Sequence[str]) -> str:
    return sha256("\n".join(chapter_hashes).encode("utf-8")).hexdigest()


# --- a document ------------------------------------------------------------------------------


def windows(
    fragments: Sequence[Fragment], counter: TokenCounter, budget: int
) -> list[list[Fragment]]:
    """The sections in order, grouped up to `budget` tokens; a longer section goes alone."""
    groups: list[list[Fragment]] = []
    current: list[Fragment] = []
    used = 0
    for fragment in fragments:
        cost = counter.count(fragment.render())
        if current and used + cost > budget:
            groups.append(current)
            current, used = [], 0
        current.append(fragment)
        used += cost
    if current:
        groups.append(current)
    return groups


def _clean(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split()).strip(" «»\"'")
    return text if 0 < len(text) <= _MAX_TEXT else None


def parse_part(
    reply: str, fragments: Sequence[Fragment], max_points: int
) -> tuple[str, list[Point]] | None:
    """(summary, points with checked quotes); None when the reply is unusable."""
    data = _json_object(reply)
    if data is None:
        return None
    overview = data.get("resumen")
    raw = data.get("puntos", [])
    if not isinstance(overview, str) or not overview.strip() or not isinstance(raw, list):
        return None
    local = {i: f for i, f in enumerate(fragments, start=1)}
    points: list[Point] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text, quote = _clean(item.get("punto")), _clean(item.get("cita"))
        number = item.get("fragmento")
        fragment = local.get(number) if isinstance(number, int) else None
        if text is None or quote is None or fragment is None:
            continue
        if not contains(fragment.text, quote):
            continue  # not in the section it names: dropped (rule 2)
        points.append(
            Point(text, fragment.heading_path[-1], fragment.start_line, fragment.end_line, quote)
        )
        if len(points) == max_points:
            break
    return " ".join(overview.split()), points


def summarize_part(
    llm: LLMClient, cfg: UnderstandingConfig, title: str, fragments: Sequence[Fragment]
) -> tuple[str, list[Point]] | None:
    numbered = "\n\n".join(
        f"[{i}] {f.heading_path[-1]}\n{f.text[: cfg.max_fragment_chars]}"
        for i, f in enumerate(fragments, start=1)
    )
    system = load_prompt("summary_part_system").format(max_points=cfg.part_points)
    user = load_prompt("summary_part_user").format(title=title, fragments=numbered)
    for _ in range(2):  # one retry
        parsed = parse_part(
            _complete(llm, system, user, cfg.part_max_tokens), fragments, cfg.part_points
        )
        if parsed is not None:
            return parsed
    return None


def summarize_document(
    llm: LLMClient,
    cfg: UnderstandingConfig,
    title: str,
    fragments: Sequence[Fragment],
    counter: TokenCounter,
    progress: Callable[[int, int], None] = lambda _d, _t: None,
) -> tuple[str, list[Point]] | None:
    """A chapter's or note's summary and key points; None if no part could be summarized."""
    groups = windows(fragments, counter, cfg.window_tokens)
    parts: list[tuple[str, list[Point]]] = []
    for i, group in enumerate(groups, start=1):
        parsed = summarize_part(llm, cfg, title, group)
        if parsed is not None:
            parts.append(parsed)
        progress(i, len(groups))
    if not parts:
        return None
    points = [p for _, part_points in parts for p in part_points][: cfg.max_points]
    if len(parts) == 1:
        return parts[0][0], points
    listed = "\n\n".join(f"Parte {i}: {overview}" for i, (overview, _) in enumerate(parts, 1))
    user = load_prompt("summary_combine_user").format(title=title, parts=listed)
    joined = _complete(llm, load_prompt("summary_combine_system"), user, cfg.combine_max_tokens)
    overview = " ".join(joined.split()) or " ".join(o for o, _ in parts)
    return overview, points


# --- a book ----------------------------------------------------------------------------------


def summarize_book(
    llm: LLMClient, cfg: UnderstandingConfig, title: str, chapters: Sequence[ChapterLine]
) -> tuple[str, tuple[int, ...]]:
    """The book's overview from its chapters' summaries, citing chapters as [n]; and the cited
    numbers that are no chapter."""
    listed = "\n\n".join(f"[{i}] {c.title}: {c.overview}" for i, c in enumerate(chapters, 1))
    user = load_prompt("summary_book_user").format(title=title, chapters=listed)
    text = _complete(llm, load_prompt("summary_book_system"), user, cfg.book_max_tokens).strip()
    invalid = tuple(n for n in extract_citations(text) if not 1 <= n <= len(chapters))
    return text, invalid


# --- the store -------------------------------------------------------------------------------


class SummaryStore:
    """`understanding.db`: one row per summary, apart from the index (app memory, ADR-040)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode = WAL")
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS summaries (
                    kind        TEXT NOT NULL,
                    key         TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    model       TEXT NOT NULL,
                    created     TEXT NOT NULL,
                    payload     TEXT NOT NULL,   -- the Summary as JSON
                    PRIMARY KEY (kind, key)
                )
                """
            )

    def get(self, kind: str, key: str) -> Summary | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM summaries WHERE kind = ? AND key = ?", (kind, key)
            ).fetchone()
        return summary_from_json(json.loads(row[0])) if row else None

    def put(self, summary: Summary) -> Summary:
        stamped = replace(summary, created=_now())
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO summaries VALUES (?, ?, ?, ?, ?, ?)",
                (
                    stamped.kind,
                    stamped.key,
                    stamped.source_hash,
                    stamped.model,
                    stamped.created,
                    json.dumps(asdict(stamped), ensure_ascii=False),
                ),
            )
        return stamped

    def all(self) -> list[Summary]:
        """Every kept summary, newest first (the library's «Resúmenes» tab)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload FROM summaries ORDER BY created DESC"
            ).fetchall()
        return [summary_from_json(json.loads(row[0])) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def summary_from_json(data: dict[str, Any]) -> Summary:
    return Summary(
        kind=data["kind"],
        key=data["key"],
        title=data["title"],
        source_hash=data["source_hash"],
        model=data["model"],
        overview=data["overview"],
        points=tuple(Point(**p) for p in data.get("points", ())),
        chapters=tuple(ChapterLine(**c) for c in data.get("chapters", ())),
        invalid=tuple(data.get("invalid", ())),
        created=data.get("created", ""),
    )
