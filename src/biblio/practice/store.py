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


@dataclass
class _Tally:
    title: str
    answered: int = 0
    score: float = 0.0
    last: str = ""


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
                CREATE INDEX IF NOT EXISTS questions_section ON questions(doc_path, heading_path);
                CREATE INDEX IF NOT EXISTS attempts_question ON attempts(question_id);
                """
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
            )

    def close(self) -> None:
        self._conn.close()

    # --- writing ------------------------------------------------------------------------------

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
