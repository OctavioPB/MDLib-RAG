from __future__ import annotations

import json
from pathlib import Path

import pytest

from biblio.config import PROJECT_ROOT
from biblio.eval.questions import QuestionsError, load_questions


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
