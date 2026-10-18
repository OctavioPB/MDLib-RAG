from __future__ import annotations

import json
import random
import threading
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from biblio.config import Config
from biblio.generate.llm import Message
from biblio.index.indexer import index_library
from biblio.practice.quiz import (
    Candidate,
    assign_kinds,
    candidate_sections,
    choose_sections,
    parse_grade,
    parse_question,
)
from biblio.practice.store import PracticeStore, QuestionContent, Section
from biblio.retrieve.bm25 import BM25Retriever
from biblio.retrieve.rerank import RerankPlan
from biblio.retrieve.store import open_index
from biblio.web.engine import Engine
from biblio.web.server import make_server

from fakes import FakeEmbedder, WordCounter

BODY = "El teorema de Bayes invierte probabilidades condicionadas usando la marginal. " * 3
GUIDES = {
    "est/bayes.md": f"# Bayes\n\n## 0. Tabla de Contenidos\n\n{BODY}\n\n## 1. Teorema\n\n{BODY}"
    f"\n\n## 2. Aplicaciones\n\n{BODY.replace('Bayes', 'clasificador')}",
    "alg/svd.md": "# SVD\n\n## Definición\n\n"
    + "La descomposición en valores singulares factoriza cualquier matriz real. " * 3,
}


@pytest.fixture
def cfg(config: Config, tmp_path: Path) -> Config:
    library = tmp_path / "library"
    for rel, text in GUIDES.items():
        (library / rel).parent.mkdir(parents=True, exist_ok=True)
        (library / rel).write_text(text, encoding="utf-8")
    cfg = replace(
        config,
        chunking=replace(config.chunking, min_tokens=5, target_tokens=40, max_tokens=60),
        paths=replace(config.paths, library=library, index_db=tmp_path / "index.db"),
        practice=replace(
            config.practice, db=tmp_path / "practice.db", min_section_tokens=5, max_questions=5
        ),
    )
    index_library(
        cfg,
        library,
        full=True,
        dtype="float32",
        embedder_factory=FakeEmbedder,
        counter_factory=WordCounter,
    )
    return cfg


# --- parsing the model's replies ----------------------------------------------------------------


def test_multiple_choice_reply_is_validated() -> None:
    good = (
        'Aquí va:\n```json\n{"pregunta": "¿Qué invierte Bayes?", "opciones": '
        '["Probabilidades condicionadas", "Matrices", "Grafos", "Integrales"], '
        '"correcta": "A", "explicacion": "Lo dice el fragmento."}\n```'
    )
    content = parse_question(good, "mc")
    assert content is not None and content.correct == 0 and len(content.options) == 4
    assert parse_question(good.replace('"Grafos"', '"matrices"'), "mc") is None  # duplicate
    assert parse_question(good.replace(', "Integrales"', ""), "mc") is None  # three options
    assert parse_question(good.replace('"A"', "7"), "mc") is None
    assert parse_question("no es JSON", "mc") is None


def test_true_false_must_match_what_was_asked() -> None:
    reply = (
        '{"afirmacion": "Bayes invierte probabilidades.", "verdadera": true, "explicacion": "x"}'
    )
    assert parse_question(reply, "tf", truth=True) == QuestionContent(
        "Bayes invierte probabilidades.", "x", truth=True
    )
    assert parse_question(reply, "tf", truth=False) is None
    assert parse_question(reply.replace("true", '"falsa"'), "tf", truth=False) is not None


def test_open_question_needs_a_reference() -> None:
    reply = '{"pregunta": "¿Qué hace Bayes?", "respuesta": "Invierte probabilidades."}'
    content = parse_question(reply, "open_model")
    assert content is not None and content.reference == "Invierte probabilidades."
    assert parse_question('{"pregunta": "¿Qué hace Bayes?"}', "open_self") is None


def test_option_letters_written_by_the_model_are_removed() -> None:
    reply = (
        '{"pregunta": "¿Qué?", "opciones": ["A) uno", "B. dos", "c: tres", "D - cuatro"], '
        '"correcta": "c", "explicacion": "La C, porque la B no."}'
    )
    content = parse_question(reply, "mc")
    assert content is not None and content.options == ("uno", "dos", "tres", "cuatro")
    assert content.correct == 2


def test_the_right_answer_goes_where_the_prompt_asks(cfg: Config) -> None:
    from biblio.practice.quiz import generate_question

    asked: list[str] = []

    class _Echo:
        def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
            letter = messages[1].content.split("Pon la respuesta correcta en la opción ")[1][0]
            asked.append(letter)
            yield (
                f'{{"pregunta": "¿Qué?", "opciones": ["a", "b", "c", "d"], "correcta": "{letter}",'
                ' "explicacion": "x"}'
            )

        def close(self) -> None:
            pass

    section = Section("g.md", ("G", "S"), 1, 2, "texto")
    rng = random.Random(3)
    results = [generate_question(_Echo(), cfg, section, "mc", rng) for _ in range(12)]
    assert {r.correct for r in results if r} == {"ABCD".index(a) for a in asked}
    assert len(set(asked)) > 1  # the position varies


def test_grade_reply() -> None:
    grade = parse_grade('{"veredicto": "Parcial", "razon": "Le falta la marginal."}')
    assert grade is not None and (grade.result, grade.reason) == (
        "partial",
        "Le falta la marginal.",
    )
    assert parse_grade('{"veredicto": "quizás"}') is None


# --- choosing sections --------------------------------------------------------------------------


def _section(n: int) -> Candidate:
    heading = json.dumps(["G", f"S{n}"], ensure_ascii=False)
    return Candidate(Section("g.md", ("G", f"S{n}"), n, n, "t"), ("g.md", heading))


def test_failed_sections_come_first_then_fresh_then_oldest() -> None:
    c = [_section(i) for i in range(5)]
    latest = {
        c[0].key: ("correct", "2026-10-02"),
        c[1].key: ("wrong", "2026-10-01"),
        c[2].key: ("correct", "2026-10-01"),
    }
    chosen = choose_sections([*c, c[3]], latest, {c[4].key}, 10, random.Random(0))
    assert [s.title for s in chosen] == ["S1", "S3", "S2", "S0"]  # S4 flagged, S3 once


def test_kinds_are_spread_evenly() -> None:
    kinds = assign_kinds(["mc", "tf"], 5, random.Random(1))
    assert sorted(kinds) == ["mc", "mc", "mc", "tf", "tf"]
    with pytest.raises(ValueError):
        assign_kinds(["essay"], 2, random.Random(1))


def test_candidate_sections_per_scope(cfg: Config) -> None:
    conn = open_index(cfg)
    guide = candidate_sections(conn, cfg, "guide", "est/bayes.md")
    assert [c.section.title for c in guide] == ["1. Teorema", "2. Aplicaciones"]  # no TOC
    assert {c.section.doc_path for c in candidate_sections(conn, cfg, "folder", "alg")} == {
        "alg/svd.md"
    }
    topic = candidate_sections(conn, cfg, "topic", "valores singulares", BM25Retriever(conn))
    assert topic[0].section.doc_path == "alg/svd.md"
    review = candidate_sections(conn, cfg, "review", "", review=[guide[1].key])
    assert [c.section.title for c in review] == ["2. Aplicaciones"]


# --- the history --------------------------------------------------------------------------------


def test_store_progress_review_and_flags(tmp_path: Path) -> None:
    store = PracticeStore(tmp_path / "p.db")
    s1 = Section("a.md", ("Guía A", "Uno"), 1, 5, "texto")
    s2 = Section("a.md", ("Guía A", "Dos"), 6, 9, "texto")
    quiz = store.new_quiz("guide", "a.md", 2)
    q1 = store.add_question(quiz, "mc", s1, QuestionContent("q", "e", ("a", "b", "c", "d"), 1))
    q2 = store.add_question(quiz, "open_model", s2, QuestionContent("q", "e", reference="r"))
    store.record_attempt(q1, "1", "correct", "code")
    attempt = store.record_attempt(q2, "mi respuesta", "correct", "model", "bien")
    assert store.review_model_grade(attempt, "wrong")  # the user disagrees
    assert not store.review_model_grade(9999, "wrong")
    assert store.flag(q1, "ambigua") and store.question(q1) is not None
    progress = store.progress()
    (guide,) = progress.guides
    assert (guide.title, guide.answered, guide.score, guide.to_review) == (
        "Guía A",
        2,
        1.0,
        ("Dos",),
    )
    assert (progress.grader_reviewed, progress.grader_agreed) == (1, 0)
    assert (progress.questions, progress.flagged, progress.to_review) == (2, 1, 1)
    assert store.flagged_sections() == {("a.md", json.dumps(["Guía A", "Uno"], ensure_ascii=False))}


# --- the web API --------------------------------------------------------------------------------


class _Teacher:
    """Answers each prompt with the JSON its kind asks for."""

    def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
        user = messages[1].content
        if "Respuesta del estudiante" in user:
            yield '{"veredicto": "correcta", "razon": "Recoge la idea."}'
        elif "opción múltiple" in user:
            yield (
                '{"pregunta": "¿Qué invierte?", "opciones": ["probabilidades", "matrices", '
                '"grafos", "series"], "correcta": 0, "explicacion": "Según el fragmento."}'
            )
        elif "afirmación" in user:
            truth = "true" if "debe ser verdadera" in user else "false"
            yield f'{{"afirmacion": "Algo.", "verdadera": {truth}, "explicacion": "Porque sí."}}'
        else:
            yield '{"pregunta": "¿Qué hace?", "respuesta": "Invierte probabilidades."}'

    def close(self) -> None:
        pass


@pytest.fixture
def server(cfg: Config) -> Iterator[tuple[str, int, Engine]]:
    engine = Engine(
        cfg,
        open_index(cfg, shared_across_threads=True),
        FakeEmbedder(),
        plan_resolver=lambda _c: RerankPlan(None, 0, "test"),
        llm_factory=lambda _c: _Teacher(),
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


def _call(server: tuple[str, int, Engine], method: str, path: str, body: Any = None) -> Any:
    import http.client

    conn = http.client.HTTPConnection(server[0], server[1], timeout=10)
    data = json.dumps(body).encode() if body is not None else None
    conn.request(method, path, body=data, headers={"Content-Type": "application/json"})
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


def test_a_whole_quiz_through_the_api(server: tuple[str, int, Engine]) -> None:
    start = {
        "scope": {"kind": "guide", "value": "est/bayes.md"},
        "count": 2,
        "kinds": ["mc", "open_model"],
    }
    status, events = _call(server, "POST", "/api/practice/start", start)
    assert status == 200 and [n for n, _ in events] == ["quiz", "question", "question", "done"]
    questions = {q["kind"]: q for n, q in events if n == "question"}
    mc = questions["mc"]
    assert "correct" not in mc and "reference" not in mc  # no answer key before answering

    # multiple choice: graded by code, with its source shown afterwards
    right = mc["options"].index("probabilidades")
    _, wrong = _call(
        server, "POST", "/api/practice/check", {"question_id": mc["id"], "choice": (right + 1) % 4}
    )
    assert wrong["result"] == "wrong" and wrong["correct"] == right
    assert wrong["source"]["citation"].startswith("est/bayes.md › ") and wrong["source"]["text"]

    # open answer graded by the model, then reviewed by the user
    open_q = questions["open_model"]
    _, graded = _call(
        server, "POST", "/api/practice/grade", {"question_id": open_q["id"], "answer": "Invierte"}
    )
    assert graded["graded"] and graded["result"] == "correct" and graded["reference"]
    _, ok = _call(
        server,
        "POST",
        "/api/practice/review",
        {"attempt_id": graded["attempt_id"], "result": "partial"},
    )
    assert ok == {"ok": True}

    _, progress = _call(server, "GET", "/api/practice/progress")
    assert progress["guides"][0]["answered"] == 2 and progress["grader"]["reviewed"] == 1
    assert progress["guides"][0]["to_review"]  # both sections need review now

    # review mode asks again about what was failed
    status, events = _call(
        server,
        "POST",
        "/api/practice/start",
        {"scope": {"kind": "review", "value": ""}, "count": 5, "kinds": ["tf"]},
    )
    assert status == 200 and events[0][1]["planned"] == 2


def test_self_graded_and_flagged_questions(server: tuple[str, int, Engine]) -> None:
    start = {"scope": {"kind": "folder", "value": "alg"}, "count": 1, "kinds": ["open_self"]}
    _, events = _call(server, "POST", "/api/practice/start", start)
    qid = events[1][1]["id"]
    _, revealed = _call(server, "POST", "/api/practice/reveal", {"question_id": qid})
    assert revealed["reference"] == "Invierte probabilidades."
    _, saved = _call(
        server,
        "POST",
        "/api/practice/self",
        {"question_id": qid, "answer": "x", "result": "partial"},
    )
    assert saved["attempt_id"] > 0
    assert _call(server, "POST", "/api/practice/flag", {"question_id": qid, "note": "ambigua"})[
        1
    ] == {"ok": True}
    status, body = _call(
        server, "POST", "/api/practice/start", start
    )  # its only section is flagged
    assert status == 400 and "secciones" in body["error"]


def test_practice_requests_are_validated(server: tuple[str, int, Engine]) -> None:
    bad = [
        {"scope": {"kind": "everything", "value": "x"}, "count": 2, "kinds": ["mc"]},
        {"scope": {"kind": "guide", "value": ""}, "count": 2, "kinds": ["mc"]},
        {"scope": {"kind": "guide", "value": "est/bayes.md"}, "count": 99, "kinds": ["mc"]},
        {"scope": {"kind": "guide", "value": "est/bayes.md"}, "count": 2, "kinds": ["essay"]},
        {"scope": {"kind": "folder", "value": "../x"}, "count": 2, "kinds": ["mc"]},
    ]
    for body in bad:
        assert _call(server, "POST", "/api/practice/start", body)[0] == 400, body
    assert _call(server, "POST", "/api/practice/check", {"question_id": 404, "choice": 1})[0] == 404
    assert _call(server, "POST", "/api/practice/check", {"question_id": "1", "choice": 1})[0] == 400
    _, guides = _call(server, "GET", "/api/practice/guides")
    assert {g["path"] for g in guides["guides"]} == {"est/bayes.md", "alg/svd.md"}
