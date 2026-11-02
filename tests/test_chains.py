"""`biblio eval-chains` (ADR-038, stage D of ADR-056): conversations planned turn by turn."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from biblio.config import PROJECT_ROOT, Config
from biblio.eval.chains import load_chains, run_chains
from biblio.eval.questions import QuestionsError
from biblio.generate.llm import Message
from biblio.retrieve.types import Hit


def _hit(rank: int, path: str, heading: str) -> Hit:
    return Hit(rank, path, ("Guía", heading), "texto", 1, 5, rank, 1.0, "t")


class _Retriever:
    name = "fake"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, k: int) -> list[Hit]:
        self.queries.append(query)
        if "Kalman" in query:
            return [_hit(1, "kalman.md", "Filtro")]
        return [_hit(1, "otro.md", "Otra cosa")]

    def close(self) -> None:
        pass


class _Planner:
    """Plans each message: the follow-up keeps the topic only if `keep` is set."""

    def __init__(self, keep: bool) -> None:
        self.keep = keep
        self.users: list[str] = []

    def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
        user = messages[1].content
        self.users.append(user)
        message = user.rsplit("Mensaje nuevo", 1)[-1] if "Mensaje nuevo" in user else user
        if "pan" in message:
            plan = {"relacion": "nuevo", "pregunta": "¿Cómo hago pan?", "busquedas": []}
        elif "varianza" in message:
            question = "¿Varianza del filtro de Kalman?" if self.keep else "¿Qué es la varianza?"
            plan = {"relacion": "sigue", "pregunta": question, "busquedas": []}
        else:
            plan = {
                "relacion": "inicio",
                "pregunta": "¿Qué es el filtro de Kalman?",
                "busquedas": [],
            }
        yield json.dumps({**plan, "tema": "filtro de Kalman", "relacionar": False})

    def close(self) -> None:
        pass


CHAIN = {
    "id": "c1",
    "turns": [
        {
            "message": "¿Qué es el filtro de Kalman?",
            "relation": "inicio",
            "expected": [{"path": "kalman.md", "heading": "Filtro"}],
        },
        {
            "message": "¿Y la varianza?",
            "relation": "sigue",
            "expected": [{"path": "kalman.md", "heading": "Filtro"}],
        },
        {"message": "¿Cómo hago pan casero?", "relation": "nuevo", "expected": []},
    ],
}


def _write(tmp_path: Path, *chains: object) -> Path:
    path = tmp_path / "chains.jsonl"
    path.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chains), encoding="utf-8")
    return path


def test_the_example_and_bad_rows(tmp_path: Path) -> None:
    assert load_chains(PROJECT_ROOT / "eval" / "chains.example.jsonl")[0].id == "c001"
    bad_first = {**CHAIN, "turns": [{**CHAIN["turns"][0], "relation": "sigue"}]}  # type: ignore[dict-item]
    with pytest.raises(QuestionsError, match="«inicio» for the first turn"):
        load_chains(_write(tmp_path, bad_first))
    turns = list(CHAIN["turns"])  # type: ignore[call-overload]
    turns[2] = {**turns[2], "expected": [{"path": "a.md", "heading": "A"}]}
    with pytest.raises(QuestionsError, match="«nuevo» turn has no expected"):
        load_chains(_write(tmp_path, {**CHAIN, "turns": turns}))


@pytest.mark.parametrize(("keep", "followup_found"), [(True, 1), (False, 0)])
def test_followups_are_planned_with_the_previous_turns(
    config: Config, tmp_path: Path, keep: bool, followup_found: int
) -> None:
    chains = load_chains(_write(tmp_path, CHAIN))
    planner = _Planner(keep)
    record = run_chains(config, _Retriever(), planner, chains, "t")
    rows = record["rows"]
    assert [r["found"] for r in rows] == [1, followup_found, 0]
    assert all(r["relation_ok"] for r in rows)
    assert "¿Qué es el filtro de Kalman?" in planner.users[1]  # the previous turn is shown
    m = record["metrics"]
    assert m["recall_first"] == 1.0 and m["recall_followups"] == float(followup_found)
