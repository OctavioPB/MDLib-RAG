"""The practice history: `data/practice.db`, apart from the index so a full reindex never deletes
it (ADR-020). Questions remember their section by path and heading, not by chunk id, because chunk
ids change when the index is rebuilt."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

KINDS = ("mc", "tf", "open_self", "open_model")  # multiple choice, true/false, open answers
RESULTS = ("correct", "partial", "wrong")
_SCORE = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}


@dataclass(frozen=True)
class Section:
    doc_path: str
    heading_path: tuple[str, ...]
    start_line: int
    end_line: int
    text: str

    @property
    def guide(self) -> str:
        return self.heading_path[0] if self.heading_path else self.doc_path

    @property
    def title(self) -> str:
        return self.heading_path[-1] if len(self.heading_path) > 1 else self.guide

    @property
    def citation(self) -> str:
        return f"{self.doc_path} › {' › '.join(self.heading_path[1:]) or self.guide}"


@dataclass(frozen=True)
class QuestionContent:
    """What the model wrote. Only the fields of its kind are filled."""

    text: str  # the question, or the statement for true/false
    explanation: str
    options: tuple[str, ...] = ()  # mc
    correct: int | None = None  # mc: index into options
    truth: bool | None = None  # tf
    reference: str = ""  # open answers: the reference answer


@dataclass(frozen=True)
class StoredQuestion:
    id: int
    quiz_id: int
    kind: str
    section: Section
    content: QuestionContent
    flagged: bool


@dataclass(frozen=True)
class GuideProgress:
    doc_path: str
    title: str
    answered: int
    score: float  # correct = 1, partial = 0.5
    last: str
    to_review: tuple[str, ...]  # section titles whose latest answer was not correct


@dataclass(frozen=True)
class Progress:
    guides: tuple[GuideProgress, ...]
    to_review: int  # sections whose latest answer was not correct, across the library
    questions: int
    flagged: int
    grader_reviewed: int  # model grades the user confirmed or changed
    grader_agreed: int


@dataclass(frozen=True)
class FeynmanSession:
    id: int
    concept: str
    fragments: list[dict[str, Any]]
    ideas: list[dict[str, Any]]
    explanation: str | None
    review: dict[str, Any] | None
    proposed: str | None
    result: str | None


@dataclass
class _Tally:
    title: str
    answered: int = 0
    score: float = 0.0
    last: str = ""


def _dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class PracticeStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._create()

    def _create(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS quizzes (
                    id         INTEGER PRIMARY KEY,
                    created    TEXT NOT NULL,
                    scope_kind TEXT NOT NULL,           -- guide | folder | topic | review
                    scope      TEXT NOT NULL,
                    requested  INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS questions (
                    id           INTEGER PRIMARY KEY,
                    quiz_id      INTEGER NOT NULL REFERENCES quizzes(id),
                    kind         TEXT NOT NULL,
                    doc_path     TEXT NOT NULL,
                    heading_path TEXT NOT NULL,         -- JSON array
                    start_line   INTEGER NOT NULL,
                    end_line     INTEGER NOT NULL,
                    section_text TEXT NOT NULL,
                    content      TEXT NOT NULL,         -- JSON of QuestionContent
                    created      TEXT NOT NULL,
                    flagged      INTEGER NOT NULL DEFAULT 0,
                    flag_note    TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    id           INTEGER PRIMARY KEY,
                    question_id  INTEGER NOT NULL REFERENCES questions(id),
                    answered     TEXT NOT NULL,
                    answer       TEXT NOT NULL,
                    result       TEXT NOT NULL,         -- correct | partial | wrong
                    graded_by    TEXT NOT NULL,         -- code | self | model
                    model_result TEXT,                  -- the model's grade, kept after a change
                    model_reason TEXT,
                    user_agrees  INTEGER                -- NULL until the user reviews a model grade
                );
                CREATE TABLE IF NOT EXISTS feynman (    -- «Explícamelo tú» sessions (ADR-027)
                    id          INTEGER PRIMARY KEY,
                    created     TEXT NOT NULL,
                    concept     TEXT NOT NULL,
                    fragments   TEXT NOT NULL,          -- JSON: the notes compared with
                    ideas       TEXT NOT NULL,          -- JSON: step 1
                    explanation TEXT,
                    review      TEXT,                   -- JSON: step 2
                    proposed    TEXT,                   -- the model's grade
                    result      TEXT,                   -- the grade the user kept
                    quiz_id     INTEGER REFERENCES quizzes(id)
                );
                CREATE TABLE IF NOT EXISTS feynman_votes (  -- «¿Es justo?» per reported point
                    session_id INTEGER NOT NULL REFERENCES feynman(id),
                    point      TEXT NOT NULL,           -- "idea:<i>" or "claim:<i>"
                    fair       INTEGER NOT NULL,
                    voted      TEXT NOT NULL,
                    PRIMARY KEY (session_id, point)
                );
                CREATE TABLE IF NOT EXISTS study_plans (  -- study plan per folder (ADR-028)
                    folder  TEXT PRIMARY KEY,
                    created TEXT NOT NULL,
                    items   TEXT NOT NULL    -- JSON: [{doc_path, reason, module, placed}]
                );
                CREATE TABLE IF NOT EXISTS dictionaries (  -- «Diccionario» (ADR-034)
                    id        INTEGER PRIMARY KEY,
                    created   TEXT NOT NULL,
                    kind      TEXT NOT NULL,          -- topic | guide
                    source    TEXT NOT NULL,          -- the topic, or the guide's path
                    title     TEXT NOT NULL,
                    entries   TEXT NOT NULL,          -- JSON
                    fragments TEXT NOT NULL           -- JSON: the notes the entries cite
                );
                CREATE TABLE IF NOT EXISTS dictionary_cards (  -- last flashcard result per entry
                    dictionary_id INTEGER NOT NULL REFERENCES dictionaries(id) ON DELETE CASCADE,
                    entry         INTEGER NOT NULL,     -- index in the dictionary's entries
                    result        TEXT NOT NULL,        -- correct | partial | wrong
                    answered      TEXT NOT NULL,
                    PRIMARY KEY (dictionary_id, entry)
                );
                CREATE TABLE IF NOT EXISTS guide_reads (  -- «Marcar como leída»
                    doc_path TEXT PRIMARY KEY,
                    marked   TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS questions_section ON questions(doc_path, heading_path);
                CREATE INDEX IF NOT EXISTS attempts_question ON attempts(question_id);
                """
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
            )

    def close(self) -> None:
        self._conn.close()

    def shared(self) -> tuple[sqlite3.Connection, threading.Lock]:
        """The connection and its lock, for stores kept in the same file (gaps, ADR-029)."""
        return self._conn, self._lock

    # --- writing ------------------------------------------------------------------------------

    def quiz_scope(self, quiz_id: int) -> tuple[str, str] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT scope_kind, scope FROM quizzes WHERE id = ?", (quiz_id,)
            ).fetchone()
        return (row[0], row[1]) if row else None

    def new_quiz(self, scope_kind: str, scope: str, requested: int) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO quizzes (created, scope_kind, scope, requested) VALUES (?, ?, ?, ?)",
                (_now(), scope_kind, scope, requested),
            )
            return int(cur.lastrowid or 0)

    def add_question(
        self, quiz_id: int, kind: str, section: Section, content: QuestionContent
    ) -> int:
        if kind not in KINDS:
            raise ValueError(f"unknown question kind {kind!r}")
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO questions (quiz_id, kind, doc_path, heading_path, start_line, "
                "end_line, section_text, content, created) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    quiz_id,
                    kind,
                    section.doc_path,
                    json.dumps(list(section.heading_path), ensure_ascii=False),
                    section.start_line,
                    section.end_line,
                    section.text,
                    json.dumps(
                        {
                            "text": content.text,
                            "explanation": content.explanation,
                            "options": list(content.options),
                            "correct": content.correct,
                            "truth": content.truth,
                            "reference": content.reference,
                        },
                        ensure_ascii=False,
                    ),
                    _now(),
                ),
            )
            return int(cur.lastrowid or 0)

    def record_attempt(
        self,
        question_id: int,
        answer: str,
        result: str,
        graded_by: str,
        model_reason: str | None = None,
    ) -> int:
        if result not in RESULTS:
            raise ValueError(f"unknown result {result!r}")
        model_result = result if graded_by == "model" else None
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO attempts (question_id, answered, answer, result, graded_by, "
                "model_result, model_reason) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (question_id, _now(), answer, result, graded_by, model_result, model_reason),
            )
            return int(cur.lastrowid or 0)

    def review_model_grade(self, attempt_id: int, result: str) -> bool:
        """The user's verdict on a model grade: kept as the result; agreement is recorded."""
        if result not in RESULTS:
            raise ValueError(f"unknown result {result!r}")
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT model_result FROM attempts WHERE id = ? AND graded_by = 'model'",
                (attempt_id,),
            ).fetchone()
            if row is None:
                return False
            self._conn.execute(
                "UPDATE attempts SET result = ?, user_agrees = ? WHERE id = ?",
                (result, int(result == row[0]), attempt_id),
            )
            return True

    def flag(self, question_id: int, note: str = "") -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE questions SET flagged = 1, flag_note = ? WHERE id = ?", (note, question_id)
            )
            return cur.rowcount == 1

    # --- reading ------------------------------------------------------------------------------

    def question(self, question_id: int) -> StoredQuestion | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, quiz_id, kind, doc_path, heading_path, start_line, end_line, "
                "section_text, content, flagged FROM questions WHERE id = ?",
                (question_id,),
            ).fetchone()
        if row is None:
            return None
        qid, quiz_id, kind, path, heading, start, end, text, content_json, flagged = row
        c = json.loads(content_json)
        content = QuestionContent(
            text=c["text"],
            explanation=c["explanation"],
            options=tuple(c["options"]),
            correct=c["correct"],
            truth=c["truth"],
            reference=c["reference"],
        )
        section = Section(path, tuple(json.loads(heading)), start, end, text)
        return StoredQuestion(qid, quiz_id, kind, section, content, bool(flagged))

    def latest_results(self) -> dict[tuple[str, str], tuple[str, str]]:
        """{(doc_path, heading_path JSON): (latest result, when)} for every section answered."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT q.doc_path, q.heading_path, a.result, a.answered FROM attempts a "
                "JOIN questions q ON q.id = a.question_id ORDER BY a.id"
            ).fetchall()
        return {(path, heading): (result, when) for path, heading, result, when in rows}

    def section_histories(self) -> dict[tuple[str, str], list[tuple[datetime, str]]]:
        """{(doc_path, heading_path JSON): [(when, result), …] oldest first}, for FSRS (ADR-025)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT q.doc_path, q.heading_path, a.answered, a.result FROM attempts a "
                "JOIN questions q ON q.id = a.question_id ORDER BY a.id"
            ).fetchall()
        out: dict[tuple[str, str], list[tuple[datetime, str]]] = {}
        for path, heading, answered, result in rows:
            out.setdefault((path, heading), []).append((datetime.fromisoformat(answered), result))
        return out

    def reviews_done_on(self, day: str) -> int:
        """Answers given on `day` (YYYY-MM-DD, local) in spaced-repetition reviews."""
        with self._lock:
            row = self._conn.execute(
                "SELECT count(*) FROM attempts a JOIN questions q ON q.id = a.question_id "
                "JOIN quizzes z ON z.id = q.quiz_id "
                "WHERE z.scope_kind = 'srs' AND substr(a.answered, 1, 10) = ?",
                (day,),
            ).fetchone()
        return int(row[0])

    def latest_question(self, key: tuple[str, str]) -> StoredQuestion | None:
        """The most recent question of a section not marked faulty (review without the LLM)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM questions WHERE doc_path = ? AND heading_path = ? AND flagged = 0 "
                "ORDER BY id DESC LIMIT 1",
                key,
            ).fetchone()
        return self.question(int(row[0])) if row else None

    # --- «Explícamelo tú» (ADR-027) -------------------------------------------------------------

    def new_feynman(
        self, concept: str, fragments: list[dict[str, Any]], ideas: list[dict[str, Any]]
    ) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO feynman (created, concept, fragments, ideas) VALUES (?, ?, ?, ?)",
                (_now(), concept, _dumps(fragments), _dumps(ideas)),
            )
            return int(cur.lastrowid or 0)

    def feynman(self, session_id: int) -> FeynmanSession | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, concept, fragments, ideas, explanation, review, proposed, result "
                "FROM feynman WHERE id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        sid, concept, fragments, ideas, explanation, review, proposed, result = row
        return FeynmanSession(
            sid,
            concept,
            json.loads(fragments),
            json.loads(ideas),
            explanation,
            json.loads(review) if review else None,
            proposed,
            result,
        )

    def set_feynman_review(
        self, session_id: int, explanation: str, review: dict[str, Any], proposed: str
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE feynman SET explanation = ?, review = ?, proposed = ? WHERE id = ?",
                (explanation, _dumps(review), proposed, session_id),
            )
            # votes were on the points of the previous check
            self._conn.execute("DELETE FROM feynman_votes WHERE session_id = ?", (session_id,))

    def set_feynman_result(self, session_id: int, result: str, quiz_id: int | None) -> None:
        if result not in RESULTS:
            raise ValueError(f"unknown result {result!r}")
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE feynman SET result = ?, quiz_id = ? WHERE id = ?",
                (result, quiz_id, session_id),
            )

    def vote_feynman(self, session_id: int, point: str, fair: bool) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO feynman_votes VALUES (?, ?, ?, ?) "
                "ON CONFLICT (session_id, point) DO UPDATE SET fair = excluded.fair, "
                "voted = excluded.voted",
                (session_id, point, int(fair), _now()),
            )

    def feynman_agreement(self) -> tuple[int, int]:
        """(votes, of them «fair»): how often the user agrees with the reported points."""
        with self._lock:
            votes, fair = self._conn.execute(
                "SELECT count(*), coalesce(sum(fair), 0) FROM feynman_votes"
            ).fetchone()
        return int(votes), int(fair)

    # --- study plans (ADR-028) -----------------------------------------------------------------

    def save_plan(self, folder: str, items: list[dict[str, Any]]) -> str:
        created = _now()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO study_plans VALUES (?, ?, ?) ON CONFLICT (folder) DO UPDATE SET "
                "created = excluded.created, items = excluded.items",
                (folder, created, _dumps(items)),
            )
        return created

    def plan(self, folder: str) -> tuple[str, list[dict[str, Any]]] | None:
        """(created, items) of the folder's plan."""
        with self._lock:
            row = self._conn.execute(
                "SELECT created, items FROM study_plans WHERE folder = ?", (folder,)
            ).fetchone()
        return (row[0], json.loads(row[1])) if row else None

    def plans(self) -> list[str]:
        """Folders with a plan, the most recent first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT folder FROM study_plans ORDER BY created DESC"
            ).fetchall()
        return [r[0] for r in rows]

    def set_plan_items(self, folder: str, items: list[dict[str, Any]]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE study_plans SET items = ? WHERE folder = ?", (_dumps(items), folder)
            )

    def delete_plan(self, folder: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM study_plans WHERE folder = ?", (folder,))
        return cur.rowcount > 0

    # --- dictionaries (ADR-034) ----------------------------------------------------------------

    def add_dictionary(
        self,
        kind: str,
        source: str,
        title: str,
        entries: list[dict[str, Any]],
        fragments: list[dict[str, Any]],
    ) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO dictionaries (created, kind, source, title, entries, fragments) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (_now(), kind, source, title, _dumps(entries), _dumps(fragments)),
            )
            return int(cur.lastrowid or 0)

    def dictionary(self, dictionary_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, created, kind, source, title, entries, fragments FROM dictionaries "
                "WHERE id = ?",
                (dictionary_id,),
            ).fetchone()
        if row is None:
            return None
        did, created, kind, source, title, entries, fragments = row
        with self._lock:
            marks = dict(
                self._conn.execute(
                    "SELECT entry, result FROM dictionary_cards WHERE dictionary_id = ?", (did,)
                ).fetchall()
            )
        return {
            "id": did,
            "created": created,
            "kind": kind,
            "source": source,
            "title": title,
            "entries": json.loads(entries),
            "fragments": json.loads(fragments),
            "marks": {str(k): v for k, v in marks.items()},  # entry index → last card result
        }

    def dictionaries(self, limit: int, query: str | None = None) -> list[dict[str, Any]]:
        """Saved dictionaries, newest first; `query` matches the title or any entry's term."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, created, kind, source, title, entries FROM dictionaries "
                "ORDER BY id DESC"
            ).fetchall()
        needle = query.casefold().strip() if query else ""
        out = []
        for did, created, kind, source, title, entries in rows:
            items = json.loads(entries)
            if (
                needle
                and needle not in title.casefold()
                and not any(needle in str(e.get("term", "")).casefold() for e in items)
            ):
                continue
            counts = {
                k: sum(1 for e in items if e.get("kind") == k)
                for k in ("concept", "function", "formula")
            }
            out.append(
                {
                    "id": did,
                    "created": created,
                    "kind": kind,
                    "source": source,
                    "title": title,
                    "counts": counts,
                }
            )
            if len(out) == limit:
                break
        return out

    def mark_card(self, dictionary_id: int, entry: int, result: str) -> None:
        if result not in RESULTS:
            raise ValueError(f"unknown result {result!r}")
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO dictionary_cards VALUES (?, ?, ?, ?) "
                "ON CONFLICT (dictionary_id, entry) "
                "DO UPDATE SET result = excluded.result, answered = excluded.answered",
                (dictionary_id, entry, result, _now()),
            )

    def delete_dictionary(self, dictionary_id: int) -> bool:
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM dictionary_cards WHERE dictionary_id = ?", (dictionary_id,)
            )
            cur = self._conn.execute("DELETE FROM dictionaries WHERE id = ?", (dictionary_id,))
        return cur.rowcount > 0

    def set_read(self, doc_path: str, read: bool) -> None:
        with self._lock, self._conn:
            if read:
                self._conn.execute(
                    "INSERT OR REPLACE INTO guide_reads VALUES (?, ?)", (doc_path, _now())
                )
            else:
                self._conn.execute("DELETE FROM guide_reads WHERE doc_path = ?", (doc_path,))

    def reads(self) -> set[str]:
        with self._lock:
            return {r[0] for r in self._conn.execute("SELECT doc_path FROM guide_reads")}

    def flagged_sections(self) -> set[tuple[str, str]]:
        """Sections with a question marked faulty: they get no new questions (ADR-020)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT doc_path, heading_path FROM questions WHERE flagged = 1"
            ).fetchall()
        return {(path, heading) for path, heading in rows}

    def progress(self) -> Progress:
        with self._lock:
            rows = self._conn.execute(
                "SELECT q.doc_path, q.heading_path, a.result, a.answered FROM attempts a "
                "JOIN questions q ON q.id = a.question_id ORDER BY a.id"
            ).fetchall()
            questions, flagged = self._conn.execute(
                "SELECT count(*), coalesce(sum(flagged), 0) FROM questions"
            ).fetchone()
            reviewed, agreed = self._conn.execute(
                "SELECT count(user_agrees), coalesce(sum(user_agrees), 0) FROM attempts "
                "WHERE graded_by = 'model'"
            ).fetchone()
        per_guide: dict[str, _Tally] = {}
        latest: dict[tuple[str, str], str] = {}
        for path, heading, result, when in rows:
            tally = per_guide.setdefault(path, _Tally(json.loads(heading)[0]))
            tally.answered += 1
            tally.score += _SCORE[result]
            tally.last = when
            latest[(path, heading)] = result
        failing: dict[str, list[str]] = {}
        for (path, heading), result in latest.items():
            if result != "correct":
                titles = json.loads(heading)
                failing.setdefault(path, []).append(titles[-1] if len(titles) > 1 else titles[0])
        guides = sorted(
            (
                GuideProgress(
                    doc_path=path,
                    title=t.title,
                    answered=t.answered,
                    score=t.score,
                    last=t.last,
                    to_review=tuple(failing.get(path, ())),
                )
                for path, t in per_guide.items()
            ),
            key=lambda g: g.last,
            reverse=True,
        )
        return Progress(
            guides=tuple(guides),
            to_review=sum(1 for r in latest.values() if r != "correct"),
            questions=int(questions),
            flagged=int(flagged),
            grader_reviewed=int(reviewed),
            grader_agreed=int(agreed),
        )
