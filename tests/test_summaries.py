"""Summaries of a chapter, note or book (ADR-056 stage E): windows, checked quotes, the store, and
the `/api/understand` route on a small library with a book."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from biblio.config import Config
from biblio.generate.context import Fragment
from biblio.generate.llm import Message
from biblio.index.embedder import DeviceChoice
from biblio.index.indexer import index_library
from biblio.retrieve.rerank import RerankPlan
from biblio.retrieve.store import open_index
from biblio.study.summaries import (
    BOOK,
    CHAPTER,
    ChapterLine,
    Point,
    Summary,
    SummaryStore,
    parse_part,
    summarize_book,
    summarize_document,
    windows,
)
from biblio.web.engine import Engine
from biblio.web.server import make_server

from fakes import FakeEmbedder, WordCounter

BAYES = "El teorema de Bayes invierte probabilidades condicionadas usando la marginal."


def frag(n: int, text: str = BAYES, heading: str = "Teorema") -> Fragment:
    return Fragment(n, "est/bayes.md", ("Bayes", heading), text, n * 10, n * 10 + 4)


class _LLM:
    """Answers each summary step by its system prompt; records the calls."""

    def __init__(self, part: str | None = None) -> None:
        self.part = part
        self.systems: list[str] = []

    def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
        system = messages[0].content
        self.systems.append(system)
        if "Resumes una parte" in system:
            yield self.part or json.dumps(
                {
                    "resumen": "Bayes invierte probabilidades.",
                    "puntos": [
                        {
                            "punto": "Invierte condicionadas.",
                            "fragmento": 1,
                            "cita": "invierte probabilidades condicionadas",
                        },
                        {
                            "punto": "Inventado.",
                            "fragmento": 1,
                            "cita": "esto no está en la sección",
                        },
                    ],
                }
            )
        elif "Unes los resúmenes" in system:
            yield "Un resumen unido de las partes."
        else:  # the book
            yield "El libro trata de Bayes [1] y de SVD [2], y de algo más [7]."

    def close(self) -> None:
        pass


def test_windows_keep_order_and_put_a_long_section_alone(config: Config) -> None:
    counter = WordCounter()
    short, long_ = frag(1, "uno dos"), frag(2, " ".join(["palabra"] * 50))
    groups = windows([short, frag(3, "tres cuatro"), long_, frag(4, "cinco")], counter, 30)
    assert [[f.number for f in g] for g in groups] == [[1, 3], [2], [4]]


def test_points_need_their_quote_in_their_section() -> None:
    fragments = [frag(7), frag(8, "Otro texto sobre SVD y matrices.", "SVD")]
    reply = json.dumps(
        {
            "resumen": "  Bayes   y SVD. ",
            "puntos": [
                {
                    "punto": "Invierte.",
                    "fragmento": 1,
                    "cita": "invierte probabilidades condicionadas",
                },
                {
                    "punto": "Mal citado.",
                    "fragmento": 2,
                    "cita": "invierte probabilidades condicionadas",
                },
                {"punto": "Sin sección.", "fragmento": 9, "cita": "Otro texto sobre SVD"},
                {"punto": "SVD.", "fragmento": 2, "cita": "Otro texto sobre SVD"},
            ],
        }
    )
    overview, points = parse_part(reply, fragments, 6)  # type: ignore[misc]
    assert overview == "Bayes y SVD."
    assert [(p.text, p.heading, p.start_line) for p in points] == [
        ("Invierte.", "Teorema", 70),  # local number 1 is the window's first section
        ("SVD.", "SVD", 80),
    ]
    assert parse_part("no es JSON", fragments, 6) is None
    assert parse_part('{"puntos": []}', fragments, 6) is None  # no summary


def test_a_long_document_is_read_in_windows_and_joined(config: Config) -> None:
    cfg = replace(config.understanding, window_tokens=12)
    llm = _LLM()
    seen: list[tuple[int, int]] = []
    made = summarize_document(
        llm,
        cfg,
        "Bayes",
        [frag(1), frag(2), frag(3)],
        WordCounter(),
        lambda d, t: seen.append((d, t)),
    )
    assert made is not None
    overview, points = made
    assert overview == "Un resumen unido de las partes." and seen == [(1, 3), (2, 3), (3, 3)]
    assert len(points) == 3 and all(
        p.quote == "invierte probabilidades condicionadas" for p in points
    )
    one = summarize_document(_LLM(), config.understanding, "Bayes", [frag(1)], WordCounter())
    assert one is not None and one[0] == "Bayes invierte probabilidades."  # no joining call
    assert summarize_document(_LLM("roto"), cfg, "Bayes", [frag(1)], WordCounter()) is None


def test_a_book_overview_flags_chapters_that_do_not_exist(config: Config) -> None:
    lines = [ChapterLine("a.md", "Bayes", "…"), ChapterLine("b.md", "SVD", "…")]
    text, invalid = summarize_book(_LLM(), config.understanding, "Libro", lines)
    assert text.startswith("El libro trata") and invalid == (7,)


def test_the_store_keeps_one_summary_per_document(tmp_path: Path) -> None:
    store = SummaryStore(tmp_path / "u.db")
    point = Point("Invierte.", "Teorema", 1, 4, "invierte probabilidades")
    first = store.put(Summary(CHAPTER, "a.md", "A", "h1", "m", "uno", (point,)))
    assert first.created and store.get(CHAPTER, "a.md") == first
    store.put(Summary(CHAPTER, "a.md", "A", "h2", "m", "dos"))
    kept = store.get(CHAPTER, "a.md")
    assert kept is not None and (kept.source_hash, kept.overview, kept.points) == ("h2", "dos", ())
    assert store.get(BOOK, "a.md") is None
    store.close()


# --- through the API -----------------------------------------------------------------------


LIBRARY = {
    "est/bayes.md": "# Bayes\n\n## Teorema\n\n" + BAYES * 3,
    "Area/Books/Libro/01-uno.md": "# Uno\n\n## Teorema\n\n" + BAYES * 3,
    "Area/Books/Libro/02-dos.md": "# Dos\n\n## Más\n\n"
    + "La marginal normaliza el resultado del teorema. " * 3,
}


@pytest.fixture
def server(config: Config, tmp_path: Path) -> Iterator[tuple[str, int, Engine, list[str]]]:
    library = tmp_path / "library"
    for rel, text in LIBRARY.items():
        (library / rel).parent.mkdir(parents=True, exist_ok=True)
        (library / rel).write_text(text, encoding="utf-8")
    cfg = replace(
        config,
        chunking=replace(config.chunking, min_tokens=5, target_tokens=40, max_tokens=60),
        paths=replace(config.paths, library=library, index_db=tmp_path / "index.db"),
        understanding=replace(config.understanding, db=tmp_path / "understanding.db"),
    )
    index_library(
        cfg,
        library,
        full=True,
        dtype="float32",
        embedder_factory=FakeEmbedder,
        counter_factory=WordCounter,
    )
    models: list[str] = []

    def factory(c: Config) -> _LLM:
        models.append(c.llm.model)
        return _LLM()

    engine = Engine(
        cfg,
        open_index(cfg, shared_across_threads=True),
        FakeEmbedder(),
        plan_resolver=lambda _c: RerankPlan(None, 0, "test"),
        llm_factory=factory,
        counter_factory=lambda _c: WordCounter(),
        index_device=lambda _c: DeviceChoice("cpu", "float32"),
    )
    engine.llm_running = lambda: True  # type: ignore[method-assign]
    srv = make_server(engine, "127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield "127.0.0.1", srv.server_address[1], engine, models
    srv.shutdown()
    srv.server_close()
    engine.close()


def _events(
    server: tuple[str, int, Engine, list[str]], body: dict[str, object]
) -> tuple[int, list[tuple[str, dict]]]:  # type: ignore[type-arg]
    from test_web import _events as parse
    from test_web import request

    status, _, raw = request(server[:2], "POST", "/api/understand", body)
    return status, parse(raw) if status == 200 else []


def test_a_note_is_summarized_on_the_reasoning_model_once(
    server: tuple[str, int, Engine, list[str]],
) -> None:
    engine, models = server[2], server[3]
    status, events = _events(server, {"path": "est/bayes.md"})
    assert status == 200 and [n for n, _ in events][-1] == "summary"
    summary = events[-1][1]
    assert summary["kind"] == "chapter" and not summary["cached"]
    assert (
        summary["points"]
        and summary["points"][0]["quote"] == "invierte probabilidades condicionadas"
    )
    assert models == [engine.config.reason.model]
    status, events = _events(server, {"path": "est/bayes.md"})
    assert events == [("summary", {**summary, "cached": True})]  # kept: no LLM call
    assert models == [engine.config.reason.model]
    assert _events(server, {"path": "no/existe.md"})[0] == 400


def test_a_book_reuses_its_chapters_and_runs_on_the_tutors_model(
    server: tuple[str, int, Engine, list[str]],
) -> None:
    engine, models = server[2], server[3]
    _events(server, {"path": "Area/Books/Libro/01-uno.md"})  # chapter 1 already summarized
    models.clear()
    status, events = _events(server, {"book": "Area/Books/Libro"})
    assert status == 200
    summary = dict(events)["summary"]
    assert summary["kind"] == "book" and [c["title"] for c in summary["chapters"]] == ["Uno", "Dos"]
    assert summary["invalid"] == [7]
    # chapter 2 and the overview, both on the tutor's model; chapter 1 was reused
    assert models == [engine.config.llm.model, engine.config.llm.model]
    assert ("progress", {"chapter": 2, "chapters": 2}) in events
    assert _events(server, {"book": "Area/Books/Libro"})[1][0][1]["cached"]


def test_kept_summaries_are_listed_and_read_without_the_llm(
    server: tuple[str, int, Engine, list[str]],
) -> None:
    from test_web import request

    engine, models = server[2], server[3]
    _events(server, {"book": "Area/Books/Libro"})
    _events(server, {"path": "est/bayes.md"})
    models.clear()
    engine.llm_running = lambda: False  # type: ignore[method-assign]
    _, _, raw = request(server[:2], "GET", "/api/summaries")
    listed = json.loads(raw)["summaries"]
    assert {(s["kind"], s["key"]) for s in listed} == {
        ("chapter", "est/bayes.md"),
        ("chapter", "Area/Books/Libro/01-uno.md"),
        ("chapter", "Area/Books/Libro/02-dos.md"),
        ("book", "Area/Books/Libro"),
    }
    by_key = {s["key"]: s for s in listed}
    assert by_key["Area/Books/Libro/02-dos.md"]["book"] == {
        "id": "Area/Books/Libro",
        "title": "Libro",
    }
    assert by_key["est/bayes.md"]["area"] == "est" and not any(s["stale"] for s in listed)
    status, _, raw = request(server[:2], "GET", "/api/summaries/one?kind=book&key=Area/Books/Libro")
    assert status == 200 and json.loads(raw)["chapters"][0]["title"] == "Uno"
    assert request(server[:2], "GET", "/api/summaries/one?kind=book&key=nada")[0] == 404
    assert models == []  # no LLM to read what is kept
    with engine.lock:  # the chapter's text changes: it and its book are out of date
        engine._conn.execute(
            "UPDATE documents SET sha256 = 'x' WHERE path = 'Area/Books/Libro/02-dos.md'"
        )
    _, _, raw = request(server[:2], "GET", "/api/summaries")
    stale = {s["key"] for s in json.loads(raw)["summaries"] if s["stale"]}
    assert stale == {"Area/Books/Libro/02-dos.md", "Area/Books/Libro"}
