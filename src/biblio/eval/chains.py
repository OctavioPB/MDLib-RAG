"""Chained conversations (ADR-038, stage D of ADR-056): does each message of a conversation reach
the sections it needs, read with the conversation's topic and previous turns?

`eval/chains.jsonl` is written by the user, one conversation per line:

    {"id": "c01", "turns": [
        {"message": "¿Qué es el filtro de Kalman?", "relation": "inicio",
         "expected": [{"path": "…", "heading": "…", "alternatives": [{"path": …, "heading": …}]}]},
        {"message": "¿Y si el sistema no es lineal?", "relation": "sigue", "expected": [...]}]}

Only planning and retrieval are measured, not answers: each turn is planned with the topic and
the previous questions (a previous answer's gist is left empty, which the app would fill in),
searched as the tutor searches, and scored by whether its expected sections are among the
fragments the tutor reads (`thread.evidence_fragments`). A turn whose `relation` is «nuevo»
expects the planner to say so and is not searched.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from biblio.config import Config
from biblio.eval.metrics import matches_section
from biblio.eval.questions import Expected, QuestionsError, _location
from biblio.eval.runner import _git_state
from biblio.generate.llm import LLMClient
from biblio.generate.thread import RELATIONS, PriorTurn, make_plan, search_plan
from biblio.retrieve.types import Hit, Retriever

RUNS_FILE = "chains.jsonl"
FIRST = "inicio"
# a searcher: (plan's standalone question, plan, k) → hits; the default is `search_plan`
Searcher = Callable[[Retriever, Any, int], list[Hit]]


@dataclass(frozen=True)
class Turn:
    message: str
    relation: str  # inicio | sigue | corrige | nuevo
    expected: tuple[Expected, ...]


@dataclass(frozen=True)
class Chain:
    id: str
    turns: tuple[Turn, ...]


def load_chains(path: Path) -> list[Chain]:
    if not path.is_file():
        raise QuestionsError(f"{path} does not exist. Write it as eval/chains.example.jsonl.")
    chains: list[Chain] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path.name}:{lineno}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise QuestionsError(f"{where}: invalid JSON ({exc.msg})") from exc
        turns_raw = raw.get("turns") if isinstance(raw, dict) else None
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str) or not turns_raw:
            raise QuestionsError(f"{where}: needs an 'id' and a list of 'turns'")
        turns = []
        for i, t in enumerate(turns_raw):
            relation = t.get("relation") if isinstance(t, dict) else None
            if relation not in (FIRST, *RELATIONS) or (relation == FIRST) != (i == 0):
                raise QuestionsError(
                    f"{where}: turn {i + 1}: 'relation' must be «inicio» for the first turn and "
                    "sigue, corrige or nuevo for the others"
                )
            expected = tuple(
                Expected(
                    _location(e, where, "expected item").path,
                    _location(e, where, "expected item").heading,
                    tuple(_location(a, where, "alternative") for a in e.get("alternatives", [])),
                )
                for e in t.get("expected", [])
            )
            if (
                relation == "nuevo" and expected
            ):  # a turn without sections is scored on its relation
                raise QuestionsError(
                    f"{where}: turn {i + 1}: a «nuevo» turn has no expected sections"
                )
            turns.append(Turn(str(t.get("message", "")), relation, expected))
        if raw["id"] in seen:
            raise QuestionsError(f"{where}: duplicate id {raw['id']!r}")
        seen.add(raw["id"])
        chains.append(Chain(raw["id"], tuple(turns)))
    if not chains:
        raise QuestionsError(f"{path} has no conversations")
    return chains


def _default_search(cfg: Config) -> Searcher:
    return lambda retriever, plan, k: search_plan(
        retriever, plan, k, cfg.thread, cfg.retrieval.rrf_k
    )


def run_chains(
    config: Config,
    retriever: Retriever,
    llm: LLMClient,
    chains: Sequence[Chain],
    tag: str,
    progress: Callable[[str, dict[str, Any]], None] = lambda _c, _r: None,
    searcher: Searcher | None = None,
) -> dict[str, Any]:
    search = searcher or _default_search(config)
    k = config.thread.evidence_fragments
    rows: list[dict[str, Any]] = []
    for chain in chains:
        frame = ""
        history: list[PriorTurn] = []
        for n, turn in enumerate(chain.turns, start=1):
            t0 = time.perf_counter()
            plan = make_plan(llm, config.thread, turn.message, frame, history)
            relation_ok = (plan.relation == turn.relation) or (
                turn.relation == FIRST and plan.relation != "nuevo"
            )
            found: list[bool] = []
            if turn.relation != "nuevo" and plan.relation != "nuevo":
                hits = search(retriever, plan, k)[:k]
                found = [any(matches_section(h, e) for h in hits) for e in turn.expected]
            elif turn.relation != "nuevo":  # the planner left the topic: nothing searched
                found = [False] * len(turn.expected)
            row = {
                "chain": chain.id,
                "turn": n,
                "message": turn.message,
                "relation": turn.relation,
                "planned_relation": plan.relation,
                "relation_ok": relation_ok,
                "question": plan.question,
                "searches": list(plan.searches),
                "expected": len(turn.expected),
                "found": sum(found),
                "seconds": round(time.perf_counter() - t0, 1),
            }
            rows.append(row)
            progress(chain.id, row)
            if plan.relation != "nuevo":
                frame = plan.frame
                history.append(PriorTurn(plan.question, ""))

    def recall(subset: Sequence[dict[str, Any]]) -> float | None:
        expected = sum(r["expected"] for r in subset)
        return round(sum(r["found"] for r in subset) / expected, 3) if expected else None

    follow = [r for r in rows if r["turn"] > 1]
    commit, dirty = _git_state(config.root)
    return {
        "tag": tag,
        "date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": commit,
        "dirty": dirty,
        "model": config.llm.model,
        "conversations": len(chains),
        "turns": len(rows),
        "metrics": {
            "recall_all": recall(rows),
            "recall_first": recall([r for r in rows if r["turn"] == 1]),
            "recall_followups": recall(follow),
            "relation_accuracy": round(sum(r["relation_ok"] for r in rows) / len(rows), 3),
        },
        "rows": rows,
    }


def append_chains_run(results_dir: Path, record: dict[str, Any]) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / RUNS_FILE
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path
