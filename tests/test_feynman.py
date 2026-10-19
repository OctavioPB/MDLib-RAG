from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from test_practice import _call, cfg  # noqa: F401  (the fixture is used by name)

from biblio.config import Config
from biblio.generate.context import Fragment
from biblio.generate.llm import Message
from biblio.practice.feynman import (
    Claim,
    Coverage,
    Idea,
    contains,
    parse_claims,
    parse_coverage,
    parse_ideas,
    proposed_grade,
)
from biblio.retrieve.rerank import RerankPlan
from biblio.retrieve.store import open_index
from biblio.web.engine import Engine
from biblio.web.server import make_server

from fakes import FakeEmbedder, WordCounter

TEXT = "El teorema de Bayes invierte probabilidades condicionadas usando la marginal."
FRAGMENTS = [
    Fragment(1, "est/bayes.md", ("Bayes", "Teorema"), TEXT, 1, 3),
    Fragment(2, "alg/svd.md", ("SVD", "Definición"), "La SVD factoriza cualquier matriz.", 1, 2),
]
EXPLANATION = "Bayes sirve para invertir probabilidades condicionadas. Lo inventó Newton en 1990."


def test_quotes_are_matched_without_accents_case_or_punctuation() -> None:
    assert contains(TEXT, "INVIERTE probabilidades, condicionadas")
    assert contains("La definición", "la definicion")
    assert not contains(TEXT, "invierte")  # one word proves nothing
    assert not contains(TEXT, "probabilidades marginales")


def test_ideas_need_a_quote_from_the_fragment_they_cite() -> None:
    reply = json.dumps(
        {
            "ideas": [
                {
                    "idea": "Invierte condicionadas.",
                    "fragmento": 1,
                    "cita": "invierte probabilidades",
                },
                {"idea": "Repetida.", "fragmento": "[1]", "cita": "usando la marginal"},
                {"idea": "Invierte condicionadas.", "fragmento": 1, "cita": "usando la marginal"},
                {"idea": "Cita de otro fragmento.", "fragmento": 2, "cita": "usando la marginal"},
                {"idea": "Fragmento inexistente.", "fragmento": 9, "cita": "factoriza cualquier"},
                {
                    "idea": "Factoriza matrices.",
                    "fragmento": 2,
                    "cita": "factoriza cualquier matriz",
                },
            ]
        }
    )
    ideas = parse_ideas(f"Aquí van: {reply}", FRAGMENTS, max_ideas=5)
    assert ideas is not None
    assert [(i.text, i.fragment) for i in ideas] == [
        ("Invierte condicionadas.", 1),
        ("Repetida.", 1),
        ("Factoriza matrices.", 2),
    ]
    assert parse_ideas("no es JSON", FRAGMENTS, 5) is None
    assert parse_ideas(reply, FRAGMENTS, 1) == ideas[:1]


def test_coverage_without_a_phrase_from_the_explanation_counts_as_missing() -> None:
    ideas = [Idea("a", 1, "x"), Idea("b", 1, "y"), Idea("c", 2, "z"), Idea("d", 2, "w")]
    reply = json.dumps(
        {
            "ideas": [
                {"numero": 1, "veredicto": "cubierta", "frase": "invertir probabilidades"},
                {"numero": 2, "veredicto": "Parcial", "frase": "algo que no escribió"},
                {"numero": 3, "veredicto": "falta", "frase": "invertir probabilidades"},
                # idea 4 skipped by the model
            ]
        }
    )
    coverage = parse_coverage(reply, ideas, EXPLANATION)
    assert coverage == [
        Coverage("covered", "invertir probabilidades"),
        Coverage("missing", ""),
        Coverage("missing", ""),
        Coverage("missing", ""),
    ]
    assert parse_coverage('{"ideas": []}', ideas, EXPLANATION) is None


def test_claims_must_be_written_by_the_user_and_quoted_from_the_notes() -> None:
    reply = json.dumps(
        {
            "afirmaciones": [
                {
                    "afirmacion": "Bayes sirve para invertir probabilidades condicionadas",
                    "veredicto": "respaldada",
                    "fragmento": 1,
                    "cita": "invierte probabilidades condicionadas",
                },
                {"afirmacion": "Lo inventó Newton en 1990", "veredicto": "sin respaldo"},
                {
                    "afirmacion": "Lo inventó Newton",
                    "veredicto": "contradice",
                    "fragmento": 1,
                    "cita": "una cita inventada por el modelo",
                },
                {"afirmacion": "Bayes es un algoritmo", "veredicto": "sin respaldo"},
            ]
        }
    )
    claims = parse_claims(reply, FRAGMENTS, EXPLANATION, max_claims=6)
    assert claims == [
        Claim(
            "Bayes sirve para invertir probabilidades condicionadas",
            "supported",
            1,
            "invierte probabilidades condicionadas",
        ),
        Claim("Lo inventó Newton en 1990", "unsupported", 0, ""),
    ]


def test_proposed_grade(config: Config) -> None:
    fy = config.feynman  # correct_at 0.8, partial_at 0.4
    covered, partial, missing = (
        Coverage("covered", "p"),
        Coverage("partial", "p"),
        Coverage("missing", ""),
    )
    assert proposed_grade([covered] * 4 + [partial], [], fy) == (0.9, "correct")
    against = [Claim("x", "contradicts", 1, "q")]
    assert proposed_grade([covered] * 5, against, fy) == (1.0, "partial")
    assert proposed_grade([covered, missing], [], fy) == (0.5, "partial")
    assert proposed_grade([partial, missing, missing], [], fy)[1] == "wrong"
    assert proposed_grade([], [], fy) == (0.0, "wrong")


# --- through the API -----------------------------------------------------------------------------


class _Checker:
    """Answers each Feynman step from the fragments in the prompt."""

    def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
        system, user = messages[0].content, messages[1].content
        numbers = dict(re.findall(r"(?m)^\[(\d+)\] (\S+)", user))
        bayes = next((int(n) for n, path in numbers.items() if path == "est/bayes.md"), 1)
        if "extraes las ideas clave" in system:
            ideas = [
                {
                    "idea": "Invierte condicionadas.",
                    "fragmento": bayes,
                    "cita": "invierte probabilidades",
                },
                {"idea": "Usa la marginal.", "fragmento": bayes, "cita": "usando la marginal"},
            ]
            yield json.dumps({"ideas": ideas})
        elif "qué ideas clave" in system:
            yield json.dumps(
                {
                    "ideas": [
                        {"numero": 1, "veredicto": "cubierta", "frase": "invierte probabilidades"},
                        {"numero": 2, "veredicto": "falta", "frase": ""},
                    ]
                }
            )
        else:
            claims = [
                {
                    "afirmacion": "Bayes invierte probabilidades",
                    "veredicto": "respaldada",
                    "fragmento": bayes,
                    "cita": "invierte probabilidades condicionadas",
                },
                {"afirmacion": "lo usan los astrónomos", "veredicto": "sin respaldo"},
            ]
            yield json.dumps({"afirmaciones": claims})

    def close(self) -> None:
        pass


@pytest.fixture
def server(cfg: Config) -> Iterator[tuple[str, int, Engine]]:  # noqa: F811
    engine = Engine(
        cfg,
        open_index(cfg, shared_across_threads=True),
        FakeEmbedder(),
        plan_resolver=lambda _c: RerankPlan(None, 0, "test"),
        llm_factory=lambda _c: _Checker(),
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


def test_explain_check_confirm_and_vote(server: tuple[str, int, Engine]) -> None:
    engine = server[2]
    status, events = _call(server, "POST", "/api/feynman/start", {"concept": "teorema de  Bayes"})
    assert status == 200 and [n for n, _ in events] == ["ready"]
    ready = events[0][1]
    assert ready["ideas"] == 2 and "text" not in json.dumps(ready)  # no ideas before explaining
    sid = ready["session_id"]

    too_short = {"session_id": sid, "explanation": "Bayes."}
    assert _call(server, "POST", "/api/feynman/check", too_short)[0] == 400
    assert (
        _call(server, "POST", "/api/feynman/confirm", {"session_id": sid, "result": "correct"})[0]
        == 404
    )  # nothing checked yet

    explanation = (
        "Bayes invierte probabilidades cuando se conoce la otra condicionada, "
        "y lo usan los astrónomos."
    )
    status, events = _call(
        server, "POST", "/api/feynman/check", {"session_id": sid, "explanation": explanation}
    )
    assert status == 200 and [n for n, _ in events] == ["coverage", "claims", "done"]
    ideas = events[0][1]["ideas"]
    assert [i["verdict"] for i in ideas] == ["covered", "missing"]
    claims = events[1][1]["claims"]
    assert [c["verdict"] for c in claims] == ["supported", "unsupported"]
    done = events[2][1]
    assert done["proposed"] == "partial" and done["score"] == 0.5
    assert any(s["citation"].startswith("est/bayes.md › ") for s in done["sources"])

    vote = {"session_id": sid, "point": "idea:1", "fair": False}
    assert _call(server, "POST", "/api/feynman/vote", vote)[0] == 200
    assert _call(server, "POST", "/api/feynman/vote", {**vote, "fair": True})[0] == 200  # changed
    assert _call(server, "POST", "/api/feynman/vote", {**vote, "point": "claim:5"})[0] == 404
    assert engine.practice.feynman_agreement() == (1, 1)

    status, body = _call(
        server, "POST", "/api/feynman/confirm", {"session_id": sid, "result": "partial"}
    )
    assert status == 200 and body == {"recorded": 1}  # both ideas came from one section
    histories = engine.practice.section_histories()
    (key,) = [k for k in histories if k[0] == "est/bayes.md"]
    assert [r for _, r in histories[key]] == ["partial"]  # counts for reviews and mastery
    assert (
        _call(server, "POST", "/api/feynman/confirm", {"session_id": sid, "result": "correct"})[0]
        == 404
    )
    again = {"session_id": sid, "explanation": explanation}
    assert _call(server, "POST", "/api/feynman/check", again)[0] == 409

    _, progress = _call(server, "GET", "/api/practice/progress")
    assert progress["feynman"] == {"votes": 1, "fair": 1, "unreliable": False}


def test_too_little_in_the_notes_is_said_plainly(server: tuple[str, int, Engine]) -> None:
    engine = server[2]
    engine._llm_factory = lambda _c: _Nothing()  # type: ignore[method-assign]
    status, body = _call(server, "POST", "/api/feynman/start", {"concept": "Bayes"})
    assert status == 400 and "ideas clave" in body["error"]
    # the generation slot is free again
    engine._llm_factory = lambda _c: _Checker()  # type: ignore[method-assign]
    assert _call(server, "POST", "/api/feynman/start", {"concept": "Bayes"})[0] == 200


class _Nothing:
    def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
        yield '{"ideas": []}'

    def close(self) -> None:
        pass


# --- offline evaluation (`biblio eval-feynman`) -----------------------------------------------


def test_eval_set_loading(tmp_path: Path) -> None:
    from biblio.eval.feynman import FeynmanSetError, load_feynman_set

    path = tmp_path / "feynman.jsonl"
    with pytest.raises(FeynmanSetError, match="Write it first"):
        load_feynman_set(path)
    line = {"id": "f1", "concept": "Bayes", "explanation": "…", "missing": ["a"]}
    path.write_text(json.dumps(line) + "\n\n" + json.dumps(line) + "\n", encoding="utf-8")
    with pytest.raises(FeynmanSetError, match="repeated id"):
        load_feynman_set(path)
    path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    (item,) = load_feynman_set(path)
    assert item.missing == ("a",) and item.unsupported == ()
    example = Path(__file__).parents[1] / "eval" / "feynman.example.jsonl"
    assert load_feynman_set(example)  # the example stays valid


def test_points_are_matched_once_by_similarity_or_quotation() -> None:
    from biblio.eval.feynman import match_points

    embed = FakeEmbedder().embed
    expected = ["la marginal normaliza", "lo publicó Laplace en 1950"]
    reported = ["la marginal normaliza", "la marginal normaliza", "otra cosa distinta"]
    assert match_points(expected, reported, embed, 0.99) == [(0, 0)]
    claims = ["Además, lo publicó Laplace en 1950 según creo"]
    assert match_points(expected, claims, embed, 0.99) == []
    assert match_points(expected, claims, embed, 0.99, literal=True) == [(1, 0)]
    assert match_points([], reported, embed, 0.5) == []


def test_eval_run_counts_found_and_extra_points(cfg: Config) -> None:  # noqa: F811
    from biblio.eval.feynman import Item, run_feynman_eval
    from biblio.retrieve.pipeline import build_retriever

    embedder = FakeEmbedder()
    retriever = build_retriever(cfg, "bm25", embedder_factory=lambda _c: embedder)
    item = Item(
        "f1",
        "Bayes",
        "Bayes invierte probabilidades cuando se conoce la otra condicionada, "
        "y lo usan los astrónomos.",
        missing=("Usa la marginal.", "Algo que el modelo no señala."),
        unsupported=("lo usan los astrónomos",),
    )
    try:
        record = run_feynman_eval(cfg, retriever, _Checker(), embedder.embed, [item], "t")
    finally:
        retriever.close()
    assert record["tag"] == "t" and record["n_items"] == 1
    m = record["metrics"]
    assert m["missing_recall"] == 0.5 and m["missing_extra"] == 0.0
    assert m["unsupported_recall"] == 1.0 and m["unsupported_extra"] == 0.0
    assert record["items"][0]["not_found"] == ["Algo que el modelo no señala."]
