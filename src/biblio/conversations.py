"""Conversar history (ADR-023): `data/conversations.db`, apart from the index.

Each message is stored when its answer is complete, with what the page shows: question, mode,
answer and the full text of its sources (an opaque JSON `payload`, rendered by the page as it was).
Search is FTS5 over questions and answers with the index's accent folding.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_MAX_TITLE = 120


@dataclass(frozen=True)
class NewTurn:
    question: str
    mode: str  # "explain" | "find"
    scope: str | None
    strategy: str
    status: str  # "done" | "error"
    error: str = ""
    answer: str = ""  # the explanation, "" for a search
    payload: dict[str, Any] | None = None  # what the page renders: fragments / hits, timings…


@dataclass(frozen=True)
class Turn:
    id: int
    created: str
    question: str
    mode: str
    scope: str | None
    strategy: str
    status: str
    error: str
    answer: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class ConversationSummary:
    id: int
    title: str
    created: str
    updated: str
    turns: int
    snippet: str = ""  # where a search matched
    frame: str = ""  # the conversation's topic, kept by the planner and editable (ADR-038)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _title(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= _MAX_TITLE else flat[: _MAX_TITLE - 1] + "…"


def fts_all_words(query: str) -> str | None:
    """`naive bayes` → `"naive" "bayes"`: every word must appear (FTS5 implicit AND)."""
    words = list(dict.fromkeys(w.lower() for w in _WORD_RE.findall(query)))
    return " ".join(f'"{w}"' for w in words) if words else None


class ConversationStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id      INTEGER PRIMARY KEY,
                    title   TEXT NOT NULL,
                    created TEXT NOT NULL,
                    updated TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS turns (
                    id              INTEGER PRIMARY KEY,
                    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    created         TEXT NOT NULL,
                    question        TEXT NOT NULL,
                    mode            TEXT NOT NULL,
                    scope           TEXT,
                    strategy        TEXT NOT NULL,
                    status          TEXT NOT NULL,
                    error           TEXT NOT NULL,
                    answer          TEXT NOT NULL,
                    payload         TEXT NOT NULL      -- JSON, rendered by the page as it was
                );
                CREATE INDEX IF NOT EXISTS turns_conversation ON turns(conversation_id, id);
                CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
                    question, answer, content='turns', content_rowid='id',
                    tokenize='unicode61 remove_diacritics 2'
                );
                CREATE TRIGGER IF NOT EXISTS turns_ai AFTER INSERT ON turns BEGIN
                    INSERT INTO turns_fts(rowid, question, answer)
                    VALUES (new.id, new.question, new.answer);
                END;
                CREATE TRIGGER IF NOT EXISTS turns_ad AFTER DELETE ON turns BEGIN
                    INSERT INTO turns_fts(turns_fts, rowid, question, answer)
                    VALUES ('delete', old.id, old.question, old.answer);
                END;
                """
            )
            columns = {row[1] for row in self._conn.execute("PRAGMA table_info(conversations)")}
            if "frame" not in columns:  # conversations.db created before ADR-038
                self._conn.execute(
                    "ALTER TABLE conversations ADD COLUMN frame TEXT NOT NULL DEFAULT ''"
                )

    def close(self) -> None:
        self._conn.close()

    # --- writing ------------------------------------------------------------------------------

    def add_turn(self, conversation_id: int | None, turn: NewTurn) -> tuple[int, int]:
        """Append a message; a missing or unknown conversation starts a new one, titled with the
        question. Returns (conversation id, turn id)."""
        now = _now()
        with self._lock, self._conn:
            exists = (
                conversation_id is not None
                and self._conn.execute(
                    "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
                ).fetchone()
            )
            if not exists:
                cur = self._conn.execute(
                    "INSERT INTO conversations (title, created, updated) VALUES (?, ?, ?)",
                    (_title(turn.question), now, now),
                )
                conversation_id = int(cur.lastrowid or 0)
            else:
                self._conn.execute(
                    "UPDATE conversations SET updated = ? WHERE id = ?", (now, conversation_id)
                )
            cur = self._conn.execute(
                "INSERT INTO turns (conversation_id, created, question, mode, scope, strategy, "
                "status, error, answer, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    conversation_id,
                    now,
                    turn.question,
                    turn.mode,
                    turn.scope,
                    turn.strategy,
                    turn.status,
                    turn.error,
                    turn.answer,
                    json.dumps(turn.payload or {}, ensure_ascii=False),
                ),
            )
            assert conversation_id is not None
            return conversation_id, int(cur.lastrowid or 0)

    def set_frame(self, conversation_id: int, frame: str) -> bool:
        """The conversation's topic (ADR-038): set by the planner, or edited by the user."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE conversations SET frame = ? WHERE id = ?",
                (" ".join(frame.split()), conversation_id),
            )
            return cur.rowcount == 1

    def rename(self, conversation_id: int, title: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ?", (_title(title), conversation_id)
            )
            return cur.rowcount == 1

    def delete(self, conversation_id: int) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            return cur.rowcount == 1

    # --- reading ------------------------------------------------------------------------------

    def _summaries(self, ids: list[int] | None, limit: int) -> list[ConversationSummary]:
        where = "" if ids is None else f"WHERE c.id IN ({','.join('?' * len(ids))})"
        rows = self._conn.execute(
            "SELECT c.id, c.title, c.created, c.updated, count(t.id), c.frame FROM conversations c "
            f"LEFT JOIN turns t ON t.conversation_id = c.id {where} "
            "GROUP BY c.id ORDER BY c.updated DESC, c.id DESC LIMIT ?",
            (*(ids or []), limit),
        ).fetchall()
        return [ConversationSummary(r[0], r[1], r[2], r[3], r[4], frame=r[5]) for r in rows]

    def conversations(self, limit: int, query: str | None = None) -> list[ConversationSummary]:
        """Newest first; with `query`, only conversations where every word appears in a question
        or an answer (accents and case ignored), with a snippet of the first match."""
        with self._lock:
            if not query or (match := fts_all_words(query)) is None:
                return self._summaries(None, limit)
            hits = self._conn.execute(
                "SELECT t.conversation_id, snippet(turns_fts, -1, '«', '»', '…', 12) "
                "FROM turns_fts JOIN turns t ON t.id = turns_fts.rowid "
                "WHERE turns_fts MATCH ? ORDER BY rank",
                (match,),
            ).fetchall()
            snippets: dict[int, str] = {}
            for conversation_id, snippet in hits:
                snippets.setdefault(conversation_id, snippet)
            if not snippets:
                return []
            found = self._summaries(list(snippets), limit)
        return [
            ConversationSummary(
                s.id, s.title, s.created, s.updated, s.turns, snippets[s.id], s.frame
            )
            for s in found
        ]

    def get(self, conversation_id: int) -> tuple[ConversationSummary, list[Turn]] | None:
        with self._lock:
            summary = self._summaries([conversation_id], 1)
            if not summary:
                return None
            rows = self._conn.execute(
                "SELECT id, created, question, mode, scope, strategy, status, error, answer, "
                "payload FROM turns WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()
        turns = [
            Turn(
                id=row[0],
                created=row[1],
                question=row[2],
                mode=row[3],
                scope=row[4],
                strategy=row[5],
                status=row[6],
                error=row[7],
                answer=row[8],
                payload=json.loads(row[9]),
            )
            for row in rows
        ]
        return summary[0], turns


# --- export -------------------------------------------------------------------------------------


def to_markdown(summary: ConversationSummary, turns: list[Turn], today: date) -> str:
    """One conversation as a note: each question, its answer or results, and their sources."""
    lines = [
        f"# {summary.title}",
        "",
        f"> **Conversación exportada de Biblioteca el {today.isoformat()}.** Cada respuesta cita",
        "> con [n] los fragmentos de tus notas que aparecen debajo de ella.",
        "",
    ]
    for turn in turns:
        lines += [f"## {turn.question}", ""]
        where = f" · en {turn.scope}" if turn.scope else ""
        kind = "Explicación" if turn.mode == "explain" else "Ubicación"
        lines += [f"*{kind} · {turn.created[:16].replace('T', ' ')}{where}*", ""]
        if turn.status == "error":
            lines += [f"> Error: {turn.error}", ""]
            continue
        if turn.mode == "explain":
            lines += [turn.answer.strip() or "(sin respuesta)", ""]
            fragments = turn.payload.get("fragments") or []
            if fragments:
                lines += ["**Fuentes**", ""]
                for f in fragments:
                    a, b = f["lines"]
                    lines.append(f"{f['number']}. {f['citation']} (líneas {a}-{b})")
                lines.append("")
        else:
            hits = turn.payload.get("hits") or []
            for h in hits:
                a, b = h["lines"]
                lines.append(f"{h['rank']}. {h['citation']} (líneas {a}-{b})")
            lines.append("" if hits else "(sin resultados)\n")
    return "\n".join(lines).rstrip("\n") + "\n"
