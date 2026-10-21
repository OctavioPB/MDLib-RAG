"""«Gap Spotter» (ADR-047): candidates from two marked sources, judged against the notes
with checked quotes, grouped into guide cards; through the API, kept and noted as pending."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator, Sequence
from dataclasses import replace

import pytest
from test_practice import _call, cfg  # noqa: F401  (the fixture is used by name)

from biblio.config import Config
from biblio.generate.context import Fragment
from biblio.generate.llm import Message
from biblio.retrieve.rerank import RerankPlan
from biblio.retrieve.store import open_index
from biblio.study.spotter import (
    SYLLABUS,
    UNEXPLAINED,
    Candidate,
    Coverage,
    complete_cards,
    parse_guides,
    parse_judgement,
    parse_unexplained,
    spot_gaps,
)
from biblio.web.engine import Engine
from biblio.web.server import make_server

from fakes import FakeEmbedder, WordCounter

TEXT = (
    "El teorema de Bayes invierte probabilidades condicionadas usando la marginal. "
    "En la práctica se combina con un prior conjugado para obtener la posterior."
)
FRAG = Fragment(1, "stats/bayes.md", ("Bayes", "Teorema"), TEXT, 1, 4)


def test_unexplained_terms_need_their_name_and_quote_in_the_fragment(cfg: Config) -> None:  # noqa: F811
    reply = json.dumps(
        {
            "terminos": [
                {
                    "termino": "prior conjugado",
                    "cita": "se combina con un prior conjugado",
                    "fragmento": 1,
                },
                {
                    "termino": "MCMC",
                    "cita": "se combina con un prior conjugado",
                    "fragmento": 1,
                },  # not named
                {
                    "termino": "posterior",
                    "cita": "una cita inventada que no está",
                    "fragmento": 1,
                },  # bad quote
                {
                    "termino": "prior conjugado",
                    "cita": "un prior conjugado para obtener",
                    "fragmento": 1,
                },  # twice
            ]
        }
    )
    found = parse_unexplained(reply, [FRAG], cfg.spotter)
    assert found is not None and [c.name for c in found] == ["prior conjugado"]
    assert found[0].source == UNEXPLAINED and found[0].mention == (
        1,
        "se combina con un prior conjugado",
    )
    assert parse_unexplained("no es JSON", [FRAG], cfg.spotter) is None


def test_the_notes_are_credited_only_with_a_checked_quote() -> None:
    candidate = Candidate("teorema de Bayes", "invertir probabilidades", SYLLABUS)
    good = json.dumps(
        {
            "veredicto": "cubierto",
            "fragmento": 1,
            "cita": "invierte probabilidades condicionadas usando",
        }
    )
    covered = parse_judgement(good, candidate, [FRAG], 6)
    assert covered is not None and covered.verdict == "covered" and covered.evidence == FRAG
    invented = json.dumps(
        {"veredicto": "cubierto", "fragmento": 1, "cita": "explica la regla de Bayes en detalle"}
    )
    found = parse_judgement(invented, candidate, [FRAG], 6)
    assert found is not None and found.verdict == "missing" and found.evidence is None
    absent = parse_judgement(json.dumps({"veredicto": "no"}), candidate, [FRAG], 6)
    assert absent is not None and absent.verdict == "missing"
    assert parse_judgement(json.dumps({"veredicto": "quizá"}), candidate, [FRAG], 6) is None
    # near-verbatim: six consecutive words of the quote are written in the note, the rest drifted
    drifted = json.dumps(
        {
            "veredicto": "cubierto",
            "fragmento": 3,
            "cita": "El teorema de Bayes invierte probabilidades condicionadas mediante marginal",
        }
    )
    near = parse_judgement(drifted, candidate, [FRAG], 6)
    assert (
        near is not None and near.verdict == "covered" and near.evidence == FRAG
    )  # wrong number, found anyway


def _gaps(n: int) -> list[Coverage]:
    return [
        Coverage(Candidate(f"subtema {i}", "", SYLLABUS), "missing", None, "")
        for i in range(1, n + 1)
    ]


def test_guides_use_each_gap_once_and_reach_the_minimum(cfg: Config) -> None:  # noqa: F811
    gaps = _gaps(7)
    reply = json.dumps(
        {
            "guias": [
                {
                    "titulo": "Guía A",
                    "por_que": "x",
                    "subtemas": [1, 2, 9],
                    "preguntas": ["¿Qué?"],
                    "notas": [1, 5],
                },
                {"titulo": "Guía B", "subtemas": [2, 3]},  # 2 already used
                {"titulo": "", "subtemas": [4]},  # no title: dropped
            ]
        }
    )
    cards = parse_guides(reply, gaps, [FRAG], cfg.spotter)
    assert cards is not None
    assert [(c.title, c.gaps) for c in cards] == [("Guía A", (0, 1)), ("Guía B", (2,))]
    assert cards[0].notes == (1,) and cards[0].questions == ("¿Qué?",)  # note 5 does not exist
    filled = complete_cards(list(cards), gaps, cfg.spotter)
    assert len(filled) == cfg.spotter.guides_min  # gaps left out get a card of their own
    assert len({g for c in filled for g in c.gaps}) == sum(len(c.gaps) for c in filled)  # each once
    assert len(complete_cards([], _gaps(2), cfg.spotter)) == 2  # fewer gaps, fewer guides


def _first_words(user: str, number: int, count: int) -> str:
    """The first words of fragment [number] as rendered in a prompt: a quote that exists."""
    body = user.split(f"[{number}] ", 1)[1].split("\n", 1)[1]
    return " ".join(re.findall(r"\w+", body)[:count])


class _Spotter:
    """A scripted model: one term mentioned without explaining, a three-subtopic syllabus, the
    first subtopic covered by the notes, and one guide for the rest."""

    def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
        system, user = messages[0].content, messages[1].content
        if system.startswith("Lees fragmentos"):
            terms = []
            if "usando la marginal" in user:  # a term the fixtures mention without explaining
                terms.append(
                    {
                        "termino": "marginal",
                        "cita": "condicionadas usando la marginal",
                        "fragmento": 1,
                    }
                )
            yield json.dumps({"terminos": terms})
        elif system.startswith("Propones el temario"):
            names = ["teorema de Bayes", "prior conjugado", "inferencia variacional"]
            yield json.dumps(
                {"subtemas": [{"nombre": n, "descripcion": f"qué es {n}"} for n in names]}
            )
        elif system.startswith("Decides si"):
            if "Subtema: teorema de Bayes" in user and "[1] " in user:
                yield json.dumps(
                    {"veredicto": "cubierto", "fragmento": 1, "cita": _first_words(user, 1, 5)}
                )
            else:
                yield json.dumps({"veredicto": "no"})
        elif system.startswith("Propones guías"):
            yield json.dumps(
                {
                    "guias": [
                        {
                            "titulo": "Inferencia bayesiana avanzada",
                            "por_que": "completa el tema",
                            "subtemas": [1, 2],
                            "preguntas": ["¿Qué es un prior conjugado?"],
                            "notas": [1],
                        }
                    ]
                }
            )
        else:  # the gap detector's own calls (ADR-030)
            yield json.dumps({"tema": "Inferencia bayesiana avanzada", "mismo": 0})

    def close(self) -> None:
        pass


def test_a_run_judges_both_sources_and_suggests_guides(cfg: Config) -> None:  # noqa: F811
    spot = replace(cfg.spotter, guides_min=2, batch=2)
    frags = [
        FRAG,
        Fragment(
            2,
            "stats/mcmc.md",
            ("MCMC", "Idea"),
            "Las cadenas de Markov muestrean la posterior.",
            1,
            2,
        ),
    ]
    steps = list(spot_gaps(_Spotter(), spot, "inferencia bayesiana", frags, lambda name: [FRAG]))
    assert [s.stage for s in steps][:2] == ["unexplained", "syllabus"] and steps[
        -1
    ].stage == "guides"
    last = steps[-1]
    names = {c.candidate.name: c for c in last.coverage}
    assert names["teorema de Bayes"].verdict == "covered" and names["teorema de Bayes"].quote
    assert (
        names["prior conjugado"].verdict == "missing"
        and names["inferencia variacional"].verdict == "missing"
    )
    # the unexplained term was judged too; it stays only if the notes do not explain it elsewhere
    assert any(c.candidate.source == UNEXPLAINED for c in last.coverage)
    assert len(last.guides) >= 2 and last.guides[0].title == "Inferencia bayesiana avanzada"


# --- through the API -----------------------------------------------------------------------------


@pytest.fixture
def server(cfg: Config) -> Iterator[tuple[str, int, Engine]]:  # noqa: F811
    engine = Engine(
        cfg,
        open_index(cfg, shared_across_threads=True),
        FakeEmbedder(),
        plan_resolver=lambda _c: RerankPlan(None, 0, "test"),
        llm_factory=lambda _c: _Spotter(),
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


def test_a_run_is_streamed_kept_and_noted_as_pending(server: tuple[str, int, Engine]) -> None:
    status, events = _call(server, "POST", "/api/spotter", {"topic": "teorema de Bayes"})
    assert status == 200 and events[0][0] == "start" and events[-1][0] == "done"
    done = events[-1][1]
    coverage = done["coverage"]
    assert coverage and all(c["verdict"] in ("covered", "partial", "missing") for c in coverage)
    for c in coverage:  # a covered subtopic is shown with its evidence
        assert c["verdict"] != "covered" or c["evidence"]["quote"]
    assert done["guides"] and done["guides"][0]["gaps"]
    run_id = done["id"]

    status, data = _call(server, "GET", "/api/spotter/history")
    assert status == 200 and data["runs"][0]["id"] == run_id
    assert data["runs"][0]["topic"] == "teorema de Bayes"
    status, data = _call(server, "GET", f"/api/spotter/{run_id}")
    assert status == 200 and data["guides"] == done["guides"]

    assert _call(server, "POST", "/api/spotter/pending", {"id": run_id, "guide": 0})[0] == 200
    gaps = _call(server, "GET", "/api/gaps")[1]["gaps"]
    assert any(e["source"] == "spotter" for g in gaps for e in g["events"])
    assert _call(server, "POST", "/api/spotter/pending", {"id": run_id, "guide": 99})[0] == 404
    assert _call(server, "POST", "/api/spotter/delete", {"id": run_id})[0] == 200
    assert _call(server, "GET", f"/api/spotter/{run_id}")[0] == 404


def test_a_reply_cut_by_the_budget_keeps_its_complete_guides(cfg: Config) -> None:  # noqa: F811
    cut = (
        '{"guias": [{"titulo": "IRLS", "por_que": "x", "subtemas": [1], "preguntas": ["¿Qué?"]},'
        ' {"titulo": "Frontera", "subtemas": [2], "preguntas": ["¿Es lineal'
    )
    cards = parse_guides(cut, _gaps(3), [FRAG], cfg.spotter)
    assert cards is not None and [c.title for c in cards] == ["IRLS"]


def test_latex_in_a_question_does_not_invalidate_the_reply(cfg: Config) -> None:  # noqa: F811
    """Regression: a question with a LaTeX command made the whole JSON unreadable."""
    reply = (
        '{"guias": [{"titulo": "A", "subtemas": [1], "preguntas": ["¿Es convexa en \\mathbf{w}?"]},'
        ' {"titulo": "B", "subtemas": [2], "preguntas": ["¿Qué es \\alpha?"]}]}'
    )
    cards = parse_guides(reply, _gaps(2), [FRAG], cfg.spotter)
    assert cards is not None and [c.title for c in cards] == ["A", "B"]
    assert cards[0].questions == ("¿Es convexa en \\mathbf{w}?",)
