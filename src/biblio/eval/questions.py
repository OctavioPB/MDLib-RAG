"""Evaluation set: `eval/questions.jsonl`, written by the user (`retrieval-eval` skill)."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

QUESTION_TYPES = ("lookup", "exact_term", "synthesis", "absent")
# The reasoning set (ADR-056, `eval/reasoning.jsonl`): what kind of thinking a question asks for,
# and what the answer must do: use the notes, add marked general knowledge, or abstain.
REASONING_TYPES = (
    "inference",
    "relation",
    "application",
    "comprehension",
    "general",
    "boundary",
    "absent",
)
BEHAVIORS = ("notes", "general", "abstain")


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
    behavior: str = ""  # reasoning set only: one of BEHAVIORS

    @property
    def should_abstain(self) -> bool:
        return self.type == "absent" or self.behavior == "abstain"


def load_questions(path: Path, reasoning: bool = False) -> list[Question]:
    """The retrieval and answer sets, or with `reasoning` the reasoning set (ADR-056)."""
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
        question = _reasoning_question(raw, where) if reasoning else _question(raw, where)
        if question.id in seen:
            raise QuestionsError(f"{where}: duplicate id {question.id!r}")
        seen.add(question.id)
        questions.append(question)
    if not questions:
        raise QuestionsError(f"{path} has no questions")
    return questions


def _question(raw: dict[str, object], where: str) -> Question:
    question = _parse(raw, where, QUESTION_TYPES)
    if question.type == "absent" and question.expected:
        raise QuestionsError(f"{where}: an 'absent' question cannot have expected sections")
    if question.type != "absent" and not question.expected:
        raise QuestionsError(
            f"{where}: a {question.type!r} question needs at least one expected section"
        )
    if question.type == "synthesis" and len(question.expected) < 2:
        raise QuestionsError(f"{where}: a 'synthesis' question needs 2+ expected sections")
    return question


def _reasoning_question(raw: dict[str, object], where: str) -> Question:
    """An answer that must abstain has no sections to find; every other one names its premises."""
    question = _parse(raw, where, REASONING_TYPES)
    behavior = raw.get("behavior")
    if behavior not in BEHAVIORS:
        raise QuestionsError(f"{where}: 'behavior' must be one of {', '.join(BEHAVIORS)}")
    if question.type in ("boundary", "absent") and behavior != "abstain":
        raise QuestionsError(f"{where}: a {question.type!r} question must have behavior 'abstain'")
    if behavior == "abstain" and question.expected:
        raise QuestionsError(f"{where}: a question that must abstain cannot have expected sections")
    if behavior != "abstain" and not question.expected:
        raise QuestionsError(f"{where}: a {question.type!r} question needs its premises' sections")
    if question.type == "relation" and len(question.expected) < 2:
        raise QuestionsError(f"{where}: a 'relation' question needs 2+ expected sections")
    return replace(question, behavior=str(behavior))


def _parse(raw: dict[str, object], where: str, types: tuple[str, ...]) -> Question:
    for key in ("id", "type", "question"):
        if not isinstance(raw.get(key), str) or not str(raw[key]).strip():
            raise QuestionsError(f"{where}: '{key}' must be a non-empty string")
    qtype = str(raw["type"])
    if qtype not in types:
        raise QuestionsError(f"{where}: type {qtype!r} is not one of {', '.join(types)}")
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
