"""Evaluation set: `eval/questions.jsonl`, written by the user (`retrieval-eval` skill)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

QUESTION_TYPES = ("lookup", "exact_term", "synthesis", "absent")


class QuestionsError(ValueError):
    """The evaluation file is malformed; the message names the line."""


@dataclass(frozen=True)
class Location:
    path: str  # relative to the library root, "/" separator
    heading: str  # last element of the section's heading_path


@dataclass(frozen=True)
class Expected:
    """One section the answer needs. Finding the primary location or any alternative satisfies it
    (other guides often answer the same thing: the skill's "ambiguity" case)."""

    path: str
    heading: str
    alternatives: tuple[Location, ...] = ()

    @property
    def options(self) -> tuple[Location, ...]:
        return (Location(self.path, self.heading), *self.alternatives)


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    question: str
    expected: tuple[Expected, ...]
    notes: str = ""
    key_points: tuple[str, ...] = ()  # ideas a good answer must contain (answer eval)


def load_questions(path: Path) -> list[Question]:
    if not path.is_file():
        raise QuestionsError(
            f"{path} does not exist. Write it following eval/questions.example.jsonl."
        )
    questions: list[Question] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path.name}:{lineno}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise QuestionsError(f"{where}: invalid JSON ({exc.msg})") from exc
        if not isinstance(raw, dict):
            raise QuestionsError(f"{where}: each line must be a JSON object")
        question = _question(raw, where)
        if question.id in seen:
            raise QuestionsError(f"{where}: duplicate id {question.id!r}")
        seen.add(question.id)
        questions.append(question)
    if not questions:
        raise QuestionsError(f"{path} has no questions")
    return questions


def _question(raw: dict[str, object], where: str) -> Question:
    for key in ("id", "type", "question"):
        if not isinstance(raw.get(key), str) or not str(raw[key]).strip():
            raise QuestionsError(f"{where}: '{key}' must be a non-empty string")
    qtype = str(raw["type"])
    if qtype not in QUESTION_TYPES:
        raise QuestionsError(f"{where}: type {qtype!r} is not one of {', '.join(QUESTION_TYPES)}")
    expected_raw = raw.get("expected", [])
    if not isinstance(expected_raw, list):
        raise QuestionsError(f"{where}: 'expected' must be a list")
    expected = []
    for item in expected_raw:
        location = _location(item, where, "expected item")
        alternatives_raw = item.get("alternatives", [])
        if not isinstance(alternatives_raw, list):
            raise QuestionsError(f"{where}: 'alternatives' must be a list")
        alternatives = tuple(_location(a, where, "alternative") for a in alternatives_raw)
        expected.append(Expected(location.path, location.heading, alternatives))
    if qtype == "absent" and expected:
        raise QuestionsError(f"{where}: an 'absent' question cannot have expected sections")
    if qtype != "absent" and not expected:
        raise QuestionsError(f"{where}: a {qtype!r} question needs at least one expected section")
    if qtype == "synthesis" and len(expected) < 2:
        raise QuestionsError(f"{where}: a 'synthesis' question needs 2+ expected sections")
    notes = raw.get("notes", "")
    key_points = raw.get("key_points", [])
    if not isinstance(key_points, list) or not all(isinstance(k, str) for k in key_points):
        raise QuestionsError(f"{where}: 'key_points' must be a list of strings")
    return Question(
        id=str(raw["id"]),
        type=qtype,
        question=str(raw["question"]),
        expected=tuple(expected),
        notes=notes if isinstance(notes, str) else "",
        key_points=tuple(key_points),
    )


def _location(item: object, where: str, what: str) -> Location:
    if (
        not isinstance(item, dict)
        or not isinstance(item.get("path"), str)
        or not isinstance(item.get("heading"), str)
    ):
        raise QuestionsError(f"{where}: each {what} needs 'path' and 'heading' strings")
    return Location(path=item["path"].replace("\\", "/"), heading=item["heading"])
