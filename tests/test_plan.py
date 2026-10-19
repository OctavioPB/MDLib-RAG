from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator, Sequence
from dataclasses import replace

import pytest
from test_practice import _call, cfg  # noqa: F401  (the fixture is used by name)

from biblio.config import Config
from biblio.generate.llm import Message
from biblio.practice.quiz import candidate_sections
from biblio.practice.store import QuestionContent
from biblio.retrieve.rerank import RerankPlan
from biblio.retrieve.store import open_index
from biblio.study.plan import (
    Guide,
    GuideState,
    PlanItem,
    PlanProgress,
    complete_order,
    next_guide,
    parse_order,
    plan_folder,
)
from biblio.web.engine import Engine
from biblio.web.server import make_server

from fakes import FakeEmbedder, WordCounter


def test_order_reply_is_checked() -> None:
    reply = json.dumps(
        {
            "orden": [
                {"guia": 3, "razon": "  Base de   todo. "},
                {"guia": "1", "razon": "usa 3"},
                {"guia": 3, "razon": "repetida"},
                {"guia": 9, "razon": "no existe"},
                {"guia": True, "razon": "no es un número"},
            ]
        }
    )
    assert parse_order(f"Claro: {reply}", 4, 200) == [(2, "Base de todo."), (0, "usa 3")]
    assert parse_order(reply, 4, 4) == [(2, "Base"), (0, "usa ")]
    assert parse_order("no JSON", 4, 200) is None
    assert parse_order('{"orden": []}', 4, 200) is None
    # the guides the model left out go last, by title, marked as not placed by it
    titles = ["D", "B", "C", "A"]
    assert complete_order([(2, "x")], titles) == [
        (2, "x", True),
        (3, "", False),
        (1, "", False),
        (0, "", False),
    ]


class _Orderer:
    """Orders every list backwards, and records what it was asked."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
        user = messages[1].content
        self.prompts.append(user)
        count = len(re.findall(r"(?m)^\d+\. ", user))
        yield json.dumps({"orden": [{"guia": n, "razon": f"r{n}"} for n in range(count, 0, -1)]})

    def close(self) -> None:
        pass


def _guides(paths: Sequence[str]) -> list[Guide]:
    return [Guide(p, p.rsplit("/", 1)[-1], ("Intro",)) for p in paths]


def test_small_folder_is_ordered_in_one_go(config: Config) -> None:
    llm = _Orderer()
    steps = list(plan_folder(llm, config.plan, "DS", _guides(["DS/a.md", "DS/b.md", "DS/x/c.md"])))
    assert steps[:-1] == [PlanProgress(1, 1, "DS")]
    items = steps[-1]
    assert isinstance(items, list)
    assert [i.doc_path for i in items] == ["DS/x/c.md", "DS/b.md", "DS/a.md"]
    assert all(i.placed and i.module == "" for i in items)
    assert "Carpeta: DS" in llm.prompts[0] and "a.md — Intro" in llm.prompts[0]


def test_big_folder_is_ordered_by_module(config: Config) -> None:
    small = replace(config.plan, max_guides=2)
    paths = [
        "S/root.md",
        "S/LLMs/a.md",
        "S/LLMs/b.md",
        "S/NLP/c.md",
        "S/NLP/d.md",
        "S/NLP/deep/e.md",
    ]
    llm = _Orderer()
    steps = list(plan_folder(llm, small, "S", _guides(paths)))
    progress = [s for s in steps if isinstance(s, PlanProgress)]
    assert progress[0].label == "módulos" and progress[-1].done == progress[-1].total
    items = steps[-1]
    assert isinstance(items, list) and all(isinstance(i, PlanItem) for i in items)
    # the fake orders everything backwards: modules (guides at the root, LLMs, NLP) and, NLP being
    # too big too, its own modules (its guides, deep); then the guides inside each
    assert [(i.doc_path, i.module) for i in items] == [
        ("S/NLP/deep/e.md", "NLP/deep"),
        ("S/NLP/d.md", "NLP"),
        ("S/NLP/c.md", "NLP"),
        ("S/LLMs/b.md", "LLMs"),
        ("S/LLMs/a.md", "LLMs"),
        ("S/root.md", ""),
    ]
    assert sum("(módulos)" in p for p in llm.prompts) == 2


def test_next_guide_is_the_first_not_practised() -> None:
    def st(state: str) -> GuideState:
        return GuideState(state, 4, 0, None)

    order = ["a", "b", "c"]
    assert next_guide(order, {"a": st("mastered"), "b": st("read"), "c": st("unread")}) == (
        "b",
        "next",
    )
    assert next_guide(order, {"a": st("mastered"), "b": st("practiced"), "c": st("mastered")}) == (
        "b",
        "review",
    )
    assert next_guide(order, {k: st("mastered") for k in order}) is None


# --- through the API -----------------------------------------------------------------------------


@pytest.fixture
def server(cfg: Config) -> Iterator[tuple[str, int, Engine]]:  # noqa: F811
    engine = Engine(
        cfg,
        open_index(cfg, shared_across_threads=True),
        FakeEmbedder(),
        plan_resolver=lambda _c: RerankPlan(None, 0, "test"),
        llm_factory=lambda _c: _Orderer(),
        counter_factory=lambda _c: WordCounter(),
    )
    engine.llm_running = lambda: True  # type: ignore[method-assign]
    srv = make_server(engine, "127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield "127.0.0.1", srv.server_address[1], engine
    srv.shutdown()
    srv.server_close()
    engine.close()


def test_plan_states_order_and_reading_marks(server: tuple[str, int, Engine]) -> None:
    engine = server[2]
    assert _call(server, "POST", "/api/plans/generate", {"folder": "nada"})[0] == 400
    assert _call(server, "GET", "/api/plans/view?folder=est")[0] == 404

    # a one-guide folder needs no model call; the root has two guides in two modules
    status, events = _call(server, "POST", "/api/plans/generate", {"folder": "est"})
    assert status == 200 and [n for n, _ in events][-1] == "plan"
    plan = events[-1][1]
    assert [i["doc_path"] for i in plan["items"]] == ["est/bayes.md"]
    assert plan["items"][0]["state"] == "unread" and plan["items"][0]["sections"] >= 2
    nxt = plan["next"]
    assert (nxt["doc_path"], nxt["title"], nxt["why"]) == ("est/bayes.md", "Bayes", "next")
    assert nxt["open_url"].startswith("obsidian://")

    assert (
        _call(server, "POST", "/api/plans/read", {"doc_path": "est/bayes.md", "read": True})[0]
        == 200
    )
    _, view = _call(server, "GET", "/api/plans/view?folder=est")
    assert view["items"][0]["state"] == "read" and view["counts"]["read"] == 1

    # practising one section makes it «practiced»; all of them recalled today makes it «mastered»
    store = engine.practice
    quiz = store.new_quiz("guide", "est/bayes.md", 1)
    with engine.lock:
        found = candidate_sections(engine._conn, engine.config, "guide", "est/bayes.md")
    content = QuestionContent("¿?", "", reference="r")
    for c in found[:1]:
        qid = store.add_question(quiz, "open_self", c.section, content)
        store.record_attempt(qid, "a", "correct", "self")
    _, view = _call(server, "GET", "/api/plans/view?folder=est")
    assert view["items"][0]["state"] == "practiced" and view["next"]["why"] == "review"
    for c in found[1:]:
        qid = store.add_question(quiz, "open_self", c.section, content)
        store.record_attempt(qid, "a", "correct", "self")
    _, view = _call(server, "GET", "/api/plans/view?folder=est")
    assert view["items"][0]["state"] == "mastered" and view["next"] is None

    status, events = _call(server, "POST", "/api/plans/generate", {"folder": ""})
    assert status == 400  # the library root is not a folder
    _, plans = _call(server, "GET", "/api/plans")
    assert [p["folder"] for p in plans["plans"]] == ["est"]

    order = {"folder": "est", "order": ["est/bayes.md", "est/bayes.md"]}
    assert _call(server, "POST", "/api/plans/order", order)[0] == 400
    assert _call(server, "POST", "/api/plans/order", {**order, "order": ["est/bayes.md"]})[0] == 200
    assert _call(server, "POST", "/api/plans/delete", {"folder": "est"})[0] == 200
    assert _call(server, "GET", "/api/plans/view?folder=est")[0] == 404


def test_the_folder_index_is_marked_for_the_model(config: Config) -> None:
    from biblio.study.plan import is_overview

    guides = [
        Guide("L/LLMs.md", "LLMs", (), frozenset({"LLMs-A.md", "LLMs-B.md", "otra.md"})),
        Guide("L/LLMs-A.md", "A", (), frozenset({"LLMs-B.md"})),
        Guide("L/LLMs-B.md", "B", ()),
    ]
    assert is_overview(guides[0], guides)
    assert not is_overview(guides[1], guides)  # one link is not an index
    llm = _Orderer()
    list(plan_folder(llm, config.plan, "L", guides))
    assert "1. LLMs [guía general: enlaza a las demás] — " in llm.prompts[0]
    assert "2. A — " in llm.prompts[0]


class _Forgetful(_Orderer):
    """Leaves the last guide out the first time."""

    def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
        reply = json.loads("".join(super().stream(messages, max_tokens)))
        if len(self.prompts) == 1:
            reply["orden"] = reply["orden"][:-1]
        yield json.dumps(reply)


def test_an_order_that_leaves_guides_out_is_asked_again(config: Config) -> None:
    llm = _Forgetful()
    steps = list(plan_folder(llm, config.plan, "D", _guides(["D/a.md", "D/b.md", "D/c.md"])))
    items = steps[-1]
    assert isinstance(items, list) and all(i.placed for i in items) and len(llm.prompts) == 2
