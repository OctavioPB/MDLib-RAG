from __future__ import annotations

import json
from pathlib import Path

import pytest

from biblio.config import PROJECT_ROOT
from biblio.eval.questions import REASONING_TYPES, QuestionsError, load_questions


def _write(tmp_path: Path, *rows: object) -> Path:
    path = tmp_path / "questions.jsonl"
    path.write_text(
        "\n".join(r if isinstance(r, str) else json.dumps(r, ensure_ascii=False) for r in rows),
        encoding="utf-8",
    )
    return path


LOOKUP = {
    "id": "q1",
    "type": "lookup",
    "question": "¿Dónde explico SVD?",
    "expected": [{"path": "algebra\\svd.md", "heading": "Definición"}],
}


def test_example_file_is_valid() -> None:
    questions = load_questions(PROJECT_ROOT / "eval" / "questions.example.jsonl")
    assert {q.type for q in questions} == {"lookup", "exact_term", "synthesis", "absent"}


def test_loads_and_normalizes_paths(tmp_path: Path) -> None:
    (question,) = load_questions(_write(tmp_path, LOOKUP, ""))
    assert question.expected[0].path == "algebra/svd.md"


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("{not json", "invalid JSON"),
        ([1, 2], "JSON object"),
        ({**LOOKUP, "type": "other"}, "not one of"),
        ({**LOOKUP, "question": ""}, "'question'"),
        ({**LOOKUP, "expected": []}, "at least one"),
        ({**LOOKUP, "type": "absent"}, "cannot have expected"),
        ({**LOOKUP, "type": "synthesis"}, "2\\+ expected"),
        ({**LOOKUP, "expected": [{"path": "a.md"}]}, "'heading'"),
    ],
)
def test_invalid_rows_name_the_line(tmp_path: Path, row: object, message: str) -> None:
    with pytest.raises(QuestionsError, match=message) as info:
        load_questions(_write(tmp_path, LOOKUP | {"id": "ok"}, row))
    assert "questions.jsonl:2" in str(info.value)


def test_duplicate_ids_and_missing_file(tmp_path: Path) -> None:
    with pytest.raises(QuestionsError, match="duplicate"):
        load_questions(_write(tmp_path, LOOKUP, LOOKUP))
    with pytest.raises(QuestionsError, match="does not exist"):
        load_questions(tmp_path / "nope.jsonl")


def test_alternatives_are_parsed_and_validated(tmp_path: Path) -> None:
    row = {
        **LOOKUP,
        "expected": [
            {
                "path": "a.md",
                "heading": "A",
                "alternatives": [{"path": "dir\\b.md", "heading": "B"}],
            }
        ],
    }
    (question,) = load_questions(_write(tmp_path, row))
    assert question.expected[0].alternatives[0].path == "dir/b.md"
    assert [o.heading for o in question.expected[0].options] == ["A", "B"]

    bad = {
        **LOOKUP,
        "expected": [{"path": "a.md", "heading": "A", "alternatives": [{"path": "b.md"}]}],
    }
    with pytest.raises(QuestionsError, match="alternative"):
        load_questions(_write(tmp_path, bad))


# --- the reasoning set (ADR-056) -----------------------------------------------------------------


def test_reasoning_example_file_is_valid() -> None:
    questions = load_questions(PROJECT_ROOT / "eval" / "reasoning.example.jsonl", reasoning=True)
    assert {q.type for q in questions} == set(REASONING_TYPES)
    assert [q.id for q in questions if q.should_abstain] == ["r06", "r07"]
    assert {q.behavior for q in questions} == {"notes", "general", "abstain"}


REASON = {
    "id": "r1",
    "type": "relation",
    "behavior": "notes",
    "question": "¿Qué comparten?",
    "expected": [{"path": "a.md", "heading": "A"}, {"path": "b.md", "heading": "B"}],
}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"behavior": "maybe"}, "'behavior' must be one of"),
        ({"type": "lookup"}, "is not one of inference"),
        ({"type": "boundary"}, "must have behavior 'abstain'"),
        ({"behavior": "abstain"}, "cannot have expected sections"),
        ({"expected": [{"path": "a.md", "heading": "A"}]}, r"needs 2\+ expected sections"),
        ({"type": "inference", "expected": []}, "needs its premises' sections"),
    ],
)
def test_invalid_reasoning_rows_name_the_line(
    tmp_path: Path, change: dict[str, object], message: str
) -> None:
    with pytest.raises(QuestionsError, match=f"questions.jsonl:1: .*{message}"):
        load_questions(_write(tmp_path, {**REASON, **change}), reasoning=True)


def test_the_reasoning_types_are_not_answer_set_types(tmp_path: Path) -> None:
    with pytest.raises(QuestionsError, match="is not one of lookup"):
        load_questions(_write(tmp_path, REASON))
    (question,) = load_questions(_write(tmp_path, LOOKUP))
    assert question.behavior == "" and not question.should_abstain
