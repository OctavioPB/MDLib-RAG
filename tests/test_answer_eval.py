from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from biblio.config import Config
from biblio.eval.questions import Expected, Question, QuestionsError, load_questions
from biblio.generate.answer_eval import parse_grades, review_markdown, run_answers, score
from biblio.generate.ask import MAP_REDUCE, Tutor
from biblio.generate.llm import Message
from biblio.retrieve.types import Hit

from fakes import WordCounter


def hit(rank: int, path: str, text: str = "texto breve") -> Hit:
    return Hit(
        rank, path, (path.upper(), f"S{rank}"), text, rank * 10, rank * 10 + 5, rank, 1.0, "t"
    )


class _Retriever:
    name = "fake"

    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.asked: list[int] = []

    def search(self, query: str, k: int) -> list[Hit]:
        self.asked.append(k)
        return self.hits[:k]

    def close(self) -> None:
        pass


class _LLM:
    """Answers map calls by guide and the final call with a fixed text."""

    def __init__(self, final: str, map_reply: dict[str, str] | None = None) -> None:
        self.final = final
        self.map_reply = map_reply or {}
        self.calls: list[Sequence[Message]] = []

    def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
        self.calls.append(messages)
        user = messages[1].content
        for guide, reply in self.map_reply.items():
            if f"Fragmentos de la guía «{guide.upper()} ({guide})»" in user:
                yield reply
                return
        yield self.final

    def close(self) -> None:
        pass


THREE_GUIDES = [hit(1, "a.md"), hit(2, "b.md"), hit(3, "c.md"), hit(4, "a.md")]


def _cfg(config: Config, **llm: object) -> Config:
    return replace(config, llm=replace(config.llm, **llm))  # type: ignore[arg-type]


# --- map-reduce ---------------------------------------------------------------------------------


def test_auto_map_reduce_when_evidence_spans_three_guides(config: Config) -> None:
    llm = _LLM(
        "A y B se relacionan [1][2].",
        {
            "a.md": "- A dice algo [1]",
            "b.md": "- B dice otra cosa [2]",
            "c.md": "Sin información relevante.",
        },
    )
    answer = Tutor(config, _Retriever(THREE_GUIDES), llm, WordCounter(), "auto").ask("¿relación?")
    assert answer.strategy == MAP_REDUCE
    assert [n.guide for n in answer.notes] == ["A.MD (a.md)", "B.MD (b.md)"]  # empty note skipped
    assert len(llm.calls) == 4  # three maps + one reduce
    reduce_user = llm.calls[-1][1].content
    assert "Guía «A.MD (a.md)»:\n- A dice algo [1]" in reduce_user
    assert answer.validation.grounded and answer.validation.cited == (1, 2)


def test_map_reduce_numbers_fragments_globally(config: Config) -> None:
    llm = _LLM("ok [4] y [5]", {"a.md": "- x [1][4]", "b.md": "- y [2]", "c.md": "- z [3]"})
    answer = Tutor(config, _Retriever(THREE_GUIDES), llm, WordCounter(), "on").ask("q")
    assert [f.number for f in answer.context.fragments] == [1, 2, 3, 4]
    a_call = next(c for c in llm.calls if "a.md)»" in c[1].content)
    assert "[1] a.md" in a_call[1].content and "[4] a.md" in a_call[1].content
    assert answer.validation.cited == (4,)  # a.md's second fragment keeps its global number
    assert answer.validation.invalid == (5,)


def test_map_reduce_off_and_few_guides_stay_direct(config: Config) -> None:
    llm = _LLM("Respuesta [1].")
    retriever = _Retriever(THREE_GUIDES)
    answer = Tutor(config, retriever, llm, WordCounter(), "off").ask("q")
    assert answer.strategy == "direct" and len(llm.calls) == 1
    assert retriever.asked == [config.retrieval.final_k]  # no deeper retrieval when off

    two = [hit(1, "a.md"), hit(2, "b.md")]
    assert (
        Tutor(config, _Retriever(two), _LLM("R [1]."), WordCounter()).ask("q").strategy == "direct"
    )


def test_map_reduce_abstains_when_every_map_is_empty(config: Config) -> None:
    llm = _LLM(
        "no debería llegar", {g: "Sin información relevante." for g in ("a.md", "b.md", "c.md")}
    )
    answer = Tutor(config, _Retriever(THREE_GUIDES), llm, WordCounter(), "auto").ask("q")
    assert answer.validation.abstained
    assert len(llm.calls) == 3  # no reduce call


def test_budget_overflow_triggers_map_reduce(config: Config) -> None:
    big = " ".join(["palabra"] * 400)
    hits = [hit(1, "a.md", big), hit(2, "b.md", big)]
    cfg = _cfg(config, context_budget_tokens=700)  # one fragment fits, the other does not
    llm = _LLM("R [1].", {"a.md": "- a [1]", "b.md": "- b [2]"})
    assert Tutor(cfg, _Retriever(hits), llm, WordCounter(), "auto").ask("q").strategy == MAP_REDUCE


# --- questions with key points ------------------------------------------------------------------


def test_key_points_are_loaded_and_validated(tmp_path: Path) -> None:
    row = {
        "id": "a1",
        "type": "lookup",
        "question": "q",
        "expected": [{"path": "a.md", "heading": "A"}],
        "key_points": ["uno", "dos"],
    }
    path = tmp_path / "answers.jsonl"
    path.write_text(json.dumps(row), encoding="utf-8")
    assert load_questions(path)[0].key_points == ("uno", "dos")
    path.write_text(json.dumps({**row, "key_points": "uno"}), encoding="utf-8")
    with pytest.raises(QuestionsError, match="key_points"):
        load_questions(path)


# --- review sheet and scoring -------------------------------------------------------------------


def _records(config: Config) -> list[dict[str, object]]:
    questions = [
        Question("a1", "lookup", "¿q1?", (Expected("a.md", "S1"),), key_points=("uno", "dos")),
        Question("a2", "absent", "¿kubeadm?", ()),
    ]

    class _Mixed(_LLM):
        def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
            yield (
                "Respuesta [1]."
                if "q1" in messages[1].content
                else "No encuentro esto en tu biblioteca."
            )

    tutor = Tutor(config, _Retriever([hit(1, "a.md")]), _Mixed(""), WordCounter(), "off")
    return run_answers(tutor, questions)


def test_review_sheet_and_score(config: Config) -> None:
    records = _records(config)
    sheet = review_markdown(records, "t")
    assert "## a1 (lookup) — ¿q1?" in sheet
    assert "**[1] a.md › S1** (líneas 10-15)" in sheet
    assert "- fidelidad: ? de ?\n- cobertura: ? de 2\n  - [ ] uno" in sheet
    graded = sheet.replace("- fidelidad: ? de ?", "- fidelidad: 3 de 4", 1).replace(
        "- cobertura: ? de 2", "- cobertura: 1 de 2"
    )
    grades = parse_grades(graded)
    assert grades == {"a1": {"fidelidad": (3, 4), "cobertura": (1, 2)}}
    result = score(records, grades)["metrics"]
    assert result["faithfulness"] == 0.75 and result["fully_faithful"] == 0.0
    assert result["coverage"] == 0.5
    assert result["abstained_when_absent"] == 1.0 and result["false_abstention"] == 0.0


def test_impossible_grades_are_ignored() -> None:
    sheet = "### Calificación x\n\n- fidelidad: 5 de 4\n- cobertura: 1 de 0\n"
    assert parse_grades(sheet) == {}


def test_hand_made_sheet_format_is_accepted() -> None:
    sheet = "\n".join(
        [
            "## a01 (lookup) — ¿q?",
            "",
            "- **fidelidad: 2 de 2**",
            "- **cobertura: 2 de 3**",
            "",
            "## a06 (synthesis) — ¿q?",
            "",
            "- **fidelidad: ? de ?** — abstención",
            "- **cobertura: 0 de 4**",
        ]
    )
    assert parse_grades(sheet) == {
        "a01": {"fidelidad": (2, 2), "cobertura": (2, 3)},
        "a06": {"cobertura": (0, 4)},
    }


def test_sheet_shows_whole_fragments_and_keeps_grades(config: Config) -> None:
    from biblio.generate.answer_eval import carry_over_grades

    records = _records(config)
    long_text = "inicio " + "palabra " * 300 + "```py\ncodigo\n``` FINAL_DEL_FRAGMENTO"
    records[0]["fragments"][0]["text"] = long_text  # type: ignore[index]
    sheet = review_markdown(records, "t")
    assert "FINAL_DEL_FRAGMENTO" in sheet  # nothing cut away from the grader
    assert "````text" in sheet  # fence longer than the fragment's own code fence
    graded = sheet.replace("- fidelidad: ? de ?", "- fidelidad: 1 de 2\n  - Nota: revisar", 1)
    graded = graded.replace("  - [ ] uno", "  - [x] uno")
    regenerated = carry_over_grades(review_markdown(records, "t"), graded)
    assert parse_grades(regenerated) == {"a1": {"fidelidad": (1, 2)}}
    assert "  - Nota: revisar" in regenerated and "  - [x] uno" in regenerated
    assert regenerated.count("## a1 (lookup)") == 1
