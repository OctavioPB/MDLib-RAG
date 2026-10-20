"""Conversations that keep their topic, and questions that reach every area they join (ADR-038)."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from biblio.config import Config
from biblio.generate.ask import load_prompt
from biblio.generate.llm import Message
from biblio.generate.thread import (
    Plan,
    PriorTurn,
    gist,
    make_plan,
    parse_plan,
    render_history,
    reserve_and_fill,
    search_plan,
)
from biblio.index.indexer import index_library
from biblio.retrieve.rerank import RerankPlan
from biblio.retrieve.store import open_index
from biblio.retrieve.types import Hit
from biblio.web.engine import Engine
from biblio.web.server import make_server

from fakes import FakeEmbedder, WordCounter


def _hit(chunk: int, doc: str, rank: int) -> Hit:
    return Hit(chunk, doc, ("Guía", f"S{chunk}"), f"texto {chunk}", 1, 2, rank, 1.0, "hybrid")


# --- the plan --------------------------------------------------------------------------------


def test_a_plan_is_validated_and_falls_back_to_the_message(config: Config) -> None:
    cfg = replace(config.thread, max_searches=2)
    reply = json.dumps(
        {
            "relacion": "corrige",
            "pregunta": "¿Qué algoritmos de ciencia de datos se usan en química?",
            "busquedas": [
                "ciencia de datos en química",
                "  ",
                "aprendizaje automático moléculas",
                "x y z",
            ],
            "tema": "Cómo la ciencia de datos aporta a la química",
            "relacionar": True,
        }
    )
    plan = parse_plan(
        f"```json\n{reply}\n```", "hablamos de química", "Ciencia de datos y química", cfg, False
    )
    assert plan is not None and plan.relation == "corrige" and plan.relate
    assert plan.searches == (
        "ciencia de datos en química",
        "aprendizaje automático moléculas",
    )  # capped
    assert plan.frame == "Cómo la ciencia de datos aporta a la química"
    first = parse_plan(reply, "m", "", cfg, True)
    assert first is not None and first.relation == "inicio"  # the first message has no topic yet
    other = parse_plan(reply.replace("corrige", "nuevo"), "¿cómo hago pan?", "Química", cfg, False)
    assert other is not None and other.frame == "Química"  # another topic never replaces it
    odd = parse_plan(reply.replace("corrige", "quizá"), "m", "T", cfg, False)
    assert odd is not None and odd.relation == "sigue"
    assert parse_plan("no es JSON", "m", "T", cfg, False) is None
    assert parse_plan('{"pregunta": "p"}', "m", "T", cfg, False) is None  # no searches list


class _Replies:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[tuple[Message, ...]] = []

    def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
        self.prompts.append(tuple(messages))
        yield self.replies.pop(0) if self.replies else "nada"

    def close(self) -> None:
        pass


def test_the_planner_sees_the_topic_and_the_last_turns(config: Config) -> None:
    llm = _Replies("roto", "otra vez roto")
    history = [PriorTurn("¿Cómo aporta la ciencia de datos a la química?", "Predice propiedades.")]
    plan = make_plan(
        llm, config.thread, "¿Qué métodos se usan?", "Ciencia de datos y química", history
    )
    assert (
        not plan.planned and plan.question == "¿Qué métodos se usan?"
    )  # asked twice, then as written
    assert plan.frame == "Ciencia de datos y química" and len(llm.prompts) == 2
    user = llm.prompts[0][1].content
    assert "Tema de la conversación: Ciencia de datos y química" in user
    assert "Tutor (resumen): Predice propiedades." in user and "¿Qué métodos se usan?" in user
    assert render_history([]) == "(ninguno)"


def test_gist_keeps_the_first_sentences_without_citations() -> None:
    answer = "La química estudia la materia [1]. Se divide en ramas [2][3]. " + "Más texto. " * 40
    short = gist(answer, 60)
    assert short == "La química estudia la materia. Se divide en ramas."
    assert "[" not in gist("Sin cortes [4]", 300)


# --- retrieval ---------------------------------------------------------------------------------


def test_the_question_keeps_its_places_and_the_plan_fills_two() -> None:
    own = [_hit(i, "quimica.md", i) for i in range(1, 9)]
    fused = [_hit(30, "polimeros.md", 1), own[0], _hit(31, "adme.md", 2), own[6]]
    ranked = reserve_and_fill(own, fused, head=6, fill=2)
    assert [h.chunk_id for h in ranked[:8]] == [1, 2, 3, 4, 5, 6, 30, 31]  # the question's six kept
    assert [h.chunk_id for h in ranked[8:]] == [7, 8]  # nothing dropped
    same = reserve_and_fill(own, own, head=6, fill=2)  # nothing new: the same eight
    assert {h.chunk_id for h in same} == {h.chunk_id for h in own}


class _ByQuery:
    name = "fake"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, k: int) -> list[Hit]:
        self.queries.append(query)
        if "polímeros" in query:
            return [_hit(30, "polimeros.md", 1), _hit(31, "adme.md", 2)]
        return [_hit(i, "quimica.md", i) for i in range(1, 9)]

    def close(self) -> None:
        pass


def test_the_plan_searches_reach_the_other_area(config: Config) -> None:
    plan = Plan(
        "inicio",
        "¿Cómo aporta la ciencia de datos a la química?",
        ("predicción de polímeros",),
        "t",
        True,
    )
    retriever = _ByQuery()
    hits = search_plan(retriever, plan, 8, config.thread, 60)
    assert retriever.queries == [plan.question, "predicción de polímeros"]
    docs = [h.doc_path for h in hits[: config.thread.evidence_fragments]]
    assert "polimeros.md" in docs and "adme.md" in docs  # within what the tutor reads
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    alone = search_plan(retriever, replace(plan, searches=()), 8, config.thread, 60)
    assert [h.chunk_id for h in alone] == list(range(1, 9))  # no searches: exactly as before


# --- the web API: topic, history, another topic ---------------------------------------------------

GUIDES = {
    "quimica/ciencia.md": "# Ciencia Química\n\n## Ramas\n\n"
    + "La química estudia la materia y sus transformaciones en ramas como la orgánica. " * 3,
    "quimica/polimeros.md": "# Polímeros\n\n## Predicción de propiedades\n\n"
    + "Los modelos de aprendizaje automático predicen propiedades de polímeros a partir de datos. "
    * 3,
    "ds/algoritmos.md": "# Algoritmos\n\n## Regresión\n\n"
    + "La regresión y los bosques aleatorios son algoritmos de ciencia de datos. " * 3,
}


class _ThreadLLM:
    """Plans by the message; answers citing [1]."""

    def __init__(self) -> None:
        self.answers: list[tuple[Message, ...]] = []

    def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
        if messages[0].content == load_prompt("thread_plan_system"):
            user = messages[1].content
            message = user.split("Mensaje nuevo del usuario: ", 1)[1].split("\n", 1)[0]
            if "pan" in message:
                plan = {
                    "relacion": "nuevo",
                    "pregunta": message,
                    "busquedas": [],
                    "tema": "x",
                    "relacionar": False,
                }
            elif "métodos" in message:
                plan = {
                    "relacion": "sigue",
                    "pregunta": "¿Qué algoritmos de ciencia de datos se usan en química?",
                    "busquedas": [
                        "algoritmos de ciencia de datos",
                        "predicción de propiedades de polímeros",
                    ],
                    "tema": "Ciencia de datos aplicada a la química",
                    "relacionar": True,
                }
            else:
                plan = {
                    "relacion": "inicio",
                    "pregunta": message,
                    "busquedas": ["aprendizaje automático en química"],
                    "tema": "Ciencia de datos aplicada a la química",
                    "relacionar": True,
                }
            yield json.dumps(plan)
            return
        self.answers.append(tuple(messages))
        yield "La regresión predice propiedades de polímeros [1]."

    def close(self) -> None:
        pass


@pytest.fixture
def api(config: Config, tmp_path: Path) -> Iterator[tuple[tuple[str, int], _ThreadLLM]]:
    library = tmp_path / "library"
    for rel, text in GUIDES.items():
        (library / rel).parent.mkdir(parents=True, exist_ok=True)
        (library / rel).write_text(text, encoding="utf-8")
    cfg = replace(
        config,
        chunking=replace(config.chunking, min_tokens=5, target_tokens=60, max_tokens=80),
        paths=replace(config.paths, library=library, index_db=tmp_path / "index.db"),
        conversations=replace(config.conversations, db=tmp_path / "conversations.db"),
        thread=replace(config.thread, plan=True),
    )
    index_library(
        cfg,
        library,
        full=True,
        dtype="float32",
        embedder_factory=FakeEmbedder,
        counter_factory=WordCounter,
    )
    llm = _ThreadLLM()
    engine = Engine(
        cfg,
        open_index(cfg, shared_across_threads=True),
        FakeEmbedder(),
        plan_resolver=lambda _c: RerankPlan(None, 0, "test"),
        llm_factory=lambda _c: llm,
        counter_factory=lambda _c: WordCounter(),
    )
    engine.llm_running = lambda: True  # type: ignore[method-assign]
    srv = make_server(engine, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield ("127.0.0.1", srv.server_address[1]), llm
    srv.shutdown()
    srv.server_close()
    engine.close()


def _post(server: tuple[str, int], path: str, body: Any) -> tuple[int, Any]:
    import http.client

    conn = http.client.HTTPConnection(*server, timeout=10)
    conn.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    raw = resp.read().decode("utf-8")
    conn.close()
    if resp.getheader("Content-Type", "").startswith("text/event-stream"):
        events = []
        for block in raw.strip().split("\n\n"):
            lines = dict(line.split(": ", 1) for line in block.splitlines())
            events.append((lines["event"], json.loads(lines["data"])))
        return resp.status, events
    return resp.status, json.loads(raw)


def _get(server: tuple[str, int], path: str) -> Any:
    import http.client

    conn = http.client.HTTPConnection(*server, timeout=10)
    conn.request("GET", path)
    body = json.loads(conn.getresponse().read().decode("utf-8"))
    conn.close()
    return body


def test_a_conversation_keeps_its_topic_and_asks_before_leaving_it(api: Any) -> None:
    server, llm = api
    _s, events = _post(
        server, "/api/ask", {"question": "¿Cómo aporta la ciencia de datos a la química?"}
    )
    meta = next(d for n, d in events if n == "meta")
    assert meta["plan"]["relation"] == "inicio" and meta["retriever"].endswith("+plan")
    cid = next(d for n, d in events if n == "done")["conversation_id"]
    assert (
        _get(server, f"/api/conversations/{cid}")["frame"]
        == "Ciencia de datos aplicada a la química"
    )
    assert "relaciona áreas distintas" in llm.answers[0][0].content  # the relate rules
    assert "Conversación hasta ahora" not in llm.answers[0][1].content  # nothing before it

    _s, events = _post(
        server, "/api/ask", {"question": "¿Qué métodos se usan?", "conversation_id": cid}
    )
    meta = next(d for n, d in events if n == "meta")
    assert meta["plan"]["question"] == "¿Qué algoritmos de ciencia de datos se usan en química?"
    user = llm.answers[1][1].content
    assert "Tema de la conversación: Ciencia de datos aplicada a la química" in user
    assert "Usuario: ¿Cómo aporta la ciencia de datos a la química?" in user
    assert "Tutor (resumen): La regresión predice propiedades de polímeros." in user
    assert "Pregunta: ¿Qué algoritmos de ciencia de datos se usan en química?" in user
    docs = {f["doc_path"] for f in meta["fragments"]}
    assert len(docs) >= 2  # both areas in the evidence

    turns_before = len(_get(server, f"/api/conversations/{cid}")["turns"])
    _s, events = _post(server, "/api/ask", {"question": "¿Cómo hago pan?", "conversation_id": cid})
    assert [n for n, _ in events] == ["offtopic"]  # nothing answered, nothing stored
    assert events[0][1]["frame"] == "Ciencia de datos aplicada a la química"
    assert len(_get(server, f"/api/conversations/{cid}")["turns"]) == turns_before
    _s, events = _post(
        server, "/api/ask", {"question": "¿Cómo hago pan?", "conversation_id": cid, "force": True}
    )
    assert "done" in [n for n, _ in events]  # «Responder aquí»
    assert "Conversación hasta ahora" not in llm.answers[-1][1].content  # answered on its own
    stored = _get(server, f"/api/conversations/{cid}")
    assert stored["frame"] == "Ciencia de datos aplicada a la química"  # the topic stays

    assert (
        _post(server, "/api/conversations/frame", {"id": cid, "frame": "Química de polímeros"})[0]
        == 200
    )
    assert _get(server, f"/api/conversations/{cid}")["frame"] == "Química de polímeros"
    assert _post(server, "/api/conversations/frame", {"id": 999, "frame": "x"})[0] == 404


def test_an_older_conversations_db_gains_the_frame_column(tmp_path: Path) -> None:
    import sqlite3

    from biblio.conversations import ConversationStore

    path = tmp_path / "conversations.db"
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE conversations (id INTEGER PRIMARY KEY, title TEXT NOT NULL, "
        "created TEXT NOT NULL, updated TEXT NOT NULL)"
    )
    old.execute("INSERT INTO conversations VALUES (1, 't', '2026-10-19', '2026-10-19')")
    old.commit()
    old.close()
    store = ConversationStore(path)
    found = store.get(1)
    assert found is not None and found[0].frame == ""
    assert store.set_frame(1, "  Tema   nuevo ") and store.get(1)[0].frame == "Tema nuevo"  # type: ignore[index]
