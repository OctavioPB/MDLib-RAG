from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence

import numpy as np
import pytest
from test_web import _events, engine, request, server  # noqa: F401  (fixtures used by name)

from biblio.generate.llm import Message
from biblio.study.gaps import GapStore, fallback_topic, parse_same, parse_topic
from biblio.web.engine import Engine


def test_topic_reply_is_checked() -> None:
    q = "¿Cómo configuro Kubernetes en bare metal?"
    assert (
        parse_topic('{"tema": "  Kubernetes en bare metal. "}', q, 8) == "Kubernetes en bare metal"
    )
    assert parse_topic('{"tema": ""}', q, 8) is None
    assert parse_topic('{"tema": "una dos tres cuatro cinco seis siete ocho nueve"}', q, 8) is None
    assert parse_topic("no JSON", q, 8) is None
    assert fallback_topic(q) == "Cómo configuro Kubernetes en bare metal"


def _vec(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


@pytest.fixture
def gaps(tmp_path) -> GapStore:  # type: ignore[no-untyped-def]
    import sqlite3

    conn = sqlite3.connect(tmp_path / "g.db", check_same_thread=False)
    return GapStore(conn, threading.Lock())


def test_same_subject_reply_is_checked() -> None:
    assert parse_same('{"mismo": 2}', 3) == 1
    assert parse_same('{"mismo": "1"}', 3) == 0
    assert parse_same('{"mismo": 0}', 3) is None
    assert parse_same('{"mismo": 4}', 3) is None  # outside the list: none
    assert parse_same("no JSON", 3) is None


def test_candidates_occurrences_and_states(gaps: GapStore) -> None:
    a = gaps.record("Kubernetes en bare metal", _vec(1, 0, 0), "ask", "¿k8s bare metal?", "")
    assert not a.merged and a.count == 1 and a.status == "open"
    near = gaps.candidates(_vec(0.8, 0.6, 0), 0.55, 5)
    assert [(c.gap_id, round(c.score, 1)) for c in near] == [(a.gap_id, 0.8)]
    assert gaps.candidates(_vec(0, 1, 0), 0.55, 5) == []
    b = gaps.record("Kubernetes sin nube", _vec(0.8, 0.6, 0), "ask", "¿k8s sin nube?", "", a.gap_id)
    assert b.merged and b.topic == "Kubernetes en bare metal" and b.count == 2
    c = gaps.record("Roux", _vec(0, 0, 1), "map", "roux", "2 conceptos")
    listed = gaps.gaps()
    assert [g["topic"] for g in listed] == ["Kubernetes en bare metal", "Roux"]  # most asked first
    assert [e["text"] for e in listed[0]["events"]] == ["¿k8s sin nube?", "¿k8s bare metal?"]

    # a resolved topic asked again reopens; a dismissed one stays dismissed, and hidden
    assert gaps.set_status(a.gap_id, "resolved")
    assert gaps.record("K8s", _vec(1, 0, 0), "ask", "¿otra vez?", "", a.gap_id).status == "open"
    assert gaps.set_status(c.gap_id, "dismissed")
    assert [x.gap_id for x in gaps.candidates(_vec(0, 0, 1), 0.55, 5)] == [c.gap_id]
    assert gaps.record("Roux", _vec(0, 0, 1), "map", "roux", "", c.gap_id).status == "dismissed"
    assert [g["topic"] for g in gaps.gaps()] == ["Kubernetes en bare metal"]
    assert gaps.set_status(a.gap_id, "draft", "kubernetes-en-bare-metal")
    assert gaps.gaps()[0]["draft"] == "kubernetes-en-bare-metal"
    with pytest.raises(ValueError):
        gaps.set_status(a.gap_id, "hecho")

    # undo removes one occurrence, and the topic with its last one
    lone = gaps.record("Docker", _vec(0, 1, 0), "ask", "¿docker?", "")
    assert gaps.undo(lone.event_id) and all(g["topic"] != "Docker" for g in gaps.gaps())
    assert gaps.undo(b.event_id) and gaps.gaps()[0]["count"] == 2
    assert not gaps.undo(9999)


class _Abstainer:
    """The tutor finds nothing; the topic prompt gets a topic; the same-subject one says yes."""

    def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
        if "tema de una guía" in messages[0].content:
            yield '{"tema": "Kubernetes en bare metal"}'
        elif "mismo tema" in messages[0].content:
            yield '{"mismo": 1}'
        else:
            yield "No encuentro esto en tu biblioteca."

    def close(self) -> None:
        pass


def test_an_abstention_is_recorded_and_can_be_undone(
    server: tuple[str, int],  # noqa: F811
    engine: Engine,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine, "llm_running", lambda: True)
    monkeypatch.setattr(engine, "_llm_factory", lambda _c: _Abstainer())
    body = {"question": "¿Cómo configuro Kubernetes en bare metal?"}
    _, _, raw = request(server, "POST", "/api/ask", body)
    events = _events(raw)
    names = [n for n, _ in events]
    assert names[-2:] == ["done", "gap"] and events[-2][1]["abstained"]
    gap = events[-1][1]
    assert gap["topic"] == "Kubernetes en bare metal" and gap["count"] == 1 and not gap["merged"]

    _, _, raw = request(server, "POST", "/api/ask", body)  # asked again: the same topic
    again = _events(raw)[-1][1]
    assert again["gap_id"] == gap["gap_id"] and again["count"] == 2 and again["merged"]

    status, _, raw = request(server, "GET", "/api/gaps")
    import json

    listed = json.loads(raw)["gaps"]
    assert status == 200 and listed[0]["count"] == 2 and listed[0]["events"][0]["source"] == "ask"

    assert request(server, "POST", "/api/gaps/undo", {"event_id": again["event_id"]})[0] == 200
    assert (
        request(server, "POST", "/api/gaps/status", {"id": gap["gap_id"], "status": "x"})[0] == 400
    )
    ok = {"id": gap["gap_id"], "status": "draft", "draft": "kubernetes"}
    assert request(server, "POST", "/api/gaps/status", ok)[0] == 200
    listed = json.loads(request(server, "GET", "/api/gaps")[2])["gaps"]
    assert listed[0]["status"] == "draft" and listed[0]["count"] == 1


def test_an_answered_question_records_nothing(
    server: tuple[str, int],  # noqa: F811
    engine: Engine,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine, "llm_running", lambda: True)
    _, _, raw = request(server, "POST", "/api/ask", {"question": "¿Qué hace Bayes?"})
    assert [n for n, _ in _events(raw)][-1] == "done"
    assert engine.gaps.gaps() == []


class _NoConcepts:
    def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
        yield '{"conceptos": [], "relaciones": []}'

    def close(self) -> None:
        pass


def test_a_thin_map_is_recorded(engine: Engine, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    monkeypatch.setattr(engine, "_llm_factory", lambda _c: _NoConcepts())
    events = list(engine.concept_map("bayes"))
    assert [n for n, _ in events][-2:] == ["done", "gap"]
    gap = events[-1][1]
    assert gap["topic"] == "bayes"
    (listed,) = engine.gaps.gaps()
    assert listed["events"][0] == {**listed["events"][0], "source": "map", "detail": "0 conceptos"}


def test_topic_presence_ignores_accents_articles_and_plurals() -> None:
    from biblio.maps.build import topic_in_text

    assert topic_in_text("química", "Las grandes ramas de la Quimica moderna")
    assert topic_in_text("las redes neuronales", "Una red neuronal… y redes neuronales profundas")
    assert not topic_in_text("salsa bechamel", "Bubble sort y cocktail sort ordenan listas")
    assert not topic_in_text("grafo", "La fotografía digital")  # not inside another word


def test_a_map_on_an_absent_topic_is_recorded(
    engine: Engine,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine, "_llm_factory", lambda _c: _NoConcepts())
    events = list(engine.concept_map("salsa bechamel"))
    gap = events[-1][1]
    assert events[-1][0] == "gap" and gap["absent"]
    assert engine.gaps.gaps()[0]["events"][0]["detail"] == "el tema no aparece en tus notas"
