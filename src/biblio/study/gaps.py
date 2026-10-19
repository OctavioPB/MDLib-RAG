"""Gap detector (ADR-029, ADR-030): what the library could not answer becomes a pending topic.

The topics live in `data/practice.db` with every occurrence (the question, or the thin map) that
raised them. Embeddings shortlist the known topics close to a new one and the LLM decides whether
it is the same subject; if so, the occurrence is added to that topic.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from biblio.config import GapsConfig
from biblio.generate.ask import load_prompt
from biblio.generate.llm import LLMClient
from biblio.practice.quiz import _complete, _json_object

STATUSES = ("open", "draft", "resolved", "dismissed")
SOURCES = ("ask", "map")
_MAX_TOPIC = 120
Embed = Callable[[Sequence[str]], np.ndarray]


def fallback_topic(question: str) -> str:
    """The question itself, without question marks, when no topic could be taken from it."""
    text = " ".join(question.split()).strip(" ¿?¡!.")
    return text[:_MAX_TOPIC]


def parse_topic(reply: str, question: str, max_words: int) -> str | None:
    data = _json_object(reply)
    topic = data.get("tema") if data is not None else None
    if not isinstance(topic, str):
        return None
    text = " ".join(topic.split()).strip(" ¿?¡!.«»\"'")
    words = len(re.findall(r"\w+", text))
    if not 1 <= words <= max_words or len(text) > _MAX_TOPIC:
        return None
    return text


def extract_topic(llm: LLMClient, cfg: GapsConfig, question: str) -> str:
    user = load_prompt("gap_topic_user").format(question=question)
    reply = _complete(llm, load_prompt("gap_topic_system"), user, cfg.topic_max_tokens)
    return parse_topic(reply, question, cfg.topic_max_words) or fallback_topic(question)


def parse_same(reply: str, candidates: int) -> int | None:
    """The 0-based index of the candidate the model chose; None for none or an unusable reply."""
    data = _json_object(reply)
    value = data.get("mismo") if data is not None else None
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= candidates:
        return None
    return value - 1


def same_topic(
    llm: LLMClient, cfg: GapsConfig, text: str, topic: str, candidates: Sequence[str]
) -> int | None:
    listed = "\n".join(f"{n}. {c}" for n, c in enumerate(candidates, start=1))
    user = load_prompt("gap_same_user").format(text=text, topic=topic, candidates=listed)
    reply = _complete(llm, load_prompt("gap_same_system"), user, cfg.same_max_tokens)
    return parse_same(reply, len(candidates))


@dataclass(frozen=True)
class Candidate:
    gap_id: int
    topic: str
    score: float


@dataclass(frozen=True)
class Recorded:
    gap_id: int
    event_id: int
    topic: str  # the gap's topic: the one it was merged into, if merged
    count: int  # occurrences so far
    merged: bool
    status: str  # after recording: open, draft or dismissed


def _unit(vector: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32).reshape(-1)
    return v / (np.linalg.norm(v) or 1.0)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class GapStore:
    """Tables in the practice database (ADR-029); shares its connection and lock."""

    def __init__(self, conn: sqlite3.Connection, lock: threading.Lock) -> None:
        self._conn = conn
        self._lock = lock
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS gaps (
                    id      INTEGER PRIMARY KEY,
                    created TEXT NOT NULL,
                    updated TEXT NOT NULL,
                    topic   TEXT NOT NULL,
                    status  TEXT NOT NULL,              -- open | draft | resolved | dismissed
                    vector  BLOB NOT NULL,              -- float32 embedding of the topic
                    draft   TEXT                        -- slug of the draft written from it
                );
                CREATE TABLE IF NOT EXISTS gap_events (
                    id     INTEGER PRIMARY KEY,
                    gap_id INTEGER NOT NULL REFERENCES gaps(id),
                    at     TEXT NOT NULL,
                    source TEXT NOT NULL,               -- ask | map
                    text   TEXT NOT NULL,               -- the question, or the map's topic
                    detail TEXT NOT NULL DEFAULT ''     -- e.g. "3 conceptos"
                );
                CREATE INDEX IF NOT EXISTS gap_events_gap ON gap_events(gap_id);
                """
            )

    def candidates(self, vector: np.ndarray, min_score: float, limit: int) -> list[Candidate]:
        """Known topics at least this close to the vector, the closest first. Dismissed ones are
        included: a question on a dismissed subject is counted there and not shown again."""
        v = _unit(vector)
        with self._lock:
            rows = self._conn.execute("SELECT id, topic, vector FROM gaps").fetchall()
        found = []
        for gid, topic, blob in rows:
            other = np.frombuffer(blob, dtype=np.float32)
            if other.shape == v.shape and (score := float(v @ other)) >= min_score:
                found.append(Candidate(gid, topic, score))
        return sorted(found, key=lambda c: -c.score)[:limit]

    def record(
        self,
        topic: str,
        vector: np.ndarray,
        source: str,
        text: str,
        detail: str,
        into: int | None = None,
    ) -> Recorded:
        """One occurrence: added to the topic `into` (the same subject), or a new topic."""
        if source not in SOURCES:
            raise ValueError(f"unknown gap source {source!r}")
        v = _unit(vector)
        now = _now()
        with self._lock, self._conn:
            row = (
                self._conn.execute(
                    "SELECT topic, status FROM gaps WHERE id = ?", (into,)
                ).fetchone()
                if into is not None
                else None
            )
            if row is None:
                cur = self._conn.execute(
                    "INSERT INTO gaps (created, updated, topic, status, vector) "
                    "VALUES (?, ?, ?, 'open', ?)",
                    (now, now, topic, v.tobytes()),
                )
                gap_id, gap_topic, status, merged = int(cur.lastrowid or 0), topic, "open", False
            else:
                assert into is not None
                gap_id, (gap_topic, status) = into, row
                if status == "resolved":  # asked again: the library still lacks it
                    status = "open"
                self._conn.execute(
                    "UPDATE gaps SET updated = ?, status = ? WHERE id = ?", (now, status, gap_id)
                )
                merged = True
            cur = self._conn.execute(
                "INSERT INTO gap_events (gap_id, at, source, text, detail) VALUES (?, ?, ?, ?, ?)",
                (gap_id, now, source, text, detail),
            )
            event_id = int(cur.lastrowid or 0)
            (count,) = self._conn.execute(
                "SELECT count(*) FROM gap_events WHERE gap_id = ?", (gap_id,)
            ).fetchone()
        return Recorded(gap_id, event_id, gap_topic, int(count), merged, status)

    def undo(self, event_id: int) -> bool:
        """Remove one occurrence; the topic goes too when it was its only one."""
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT gap_id FROM gap_events WHERE id = ?", (event_id,)
            ).fetchone()
            if row is None:
                return False
            self._conn.execute("DELETE FROM gap_events WHERE id = ?", (event_id,))
            (left,) = self._conn.execute(
                "SELECT count(*) FROM gap_events WHERE gap_id = ?", (row[0],)
            ).fetchone()
            if not left:
                self._conn.execute("DELETE FROM gaps WHERE id = ?", (row[0],))
        return True

    def set_status(self, gap_id: int, status: str, draft: str | None = None) -> bool:
        if status not in STATUSES:
            raise ValueError(f"unknown gap status {status!r}")
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE gaps SET status = ?, updated = ?, draft = coalesce(?, draft) WHERE id = ?",
                (status, _now(), draft, gap_id),
            )
        return cur.rowcount > 0

    def gaps(self, statuses: Sequence[str] = ("open", "draft")) -> list[dict[str, object]]:
        """Gaps in those states, the most asked first, each with its latest occurrences."""
        marks = ",".join("?" * len(statuses))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT g.id, g.topic, g.status, g.created, g.updated, g.draft, count(e.id) "
                f"FROM gaps g JOIN gap_events e ON e.gap_id = g.id WHERE g.status IN ({marks}) "
                f"GROUP BY g.id ORDER BY count(e.id) DESC, g.updated DESC",
                tuple(statuses),
            ).fetchall()
            out = []
            for gid, topic, status, created, updated, draft, count in rows:
                events = self._conn.execute(
                    "SELECT source, text, detail, at FROM gap_events WHERE gap_id = ? "
                    "ORDER BY id DESC LIMIT 3",
                    (gid,),
                ).fetchall()
                out.append(
                    {
                        "id": gid,
                        "topic": topic,
                        "status": status,
                        "created": created,
                        "updated": updated,
                        "draft": draft,
                        "count": count,
                        "events": [
                            {"source": s, "text": t, "detail": d, "at": a} for s, t, d, a in events
                        ],
                    }
                )
        return out
