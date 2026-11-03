"""Choosing where to practice: a guide, a book, a chapter, a folder or all of it (ADR-063)."""

from __future__ import annotations

import http.client
import json
import threading
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from biblio.config import Config
from biblio.generate.llm import Message
from biblio.index.indexer import index_library
from biblio.practice.quiz import candidate_sections
from biblio.retrieve.rerank import RerankPlan
from biblio.retrieve.store import open_index
from biblio.web.engine import Engine
from biblio.web.server import make_server

from fakes import FakeEmbedder, WordCounter

BODY = "Un grupo de consumidores reparte las particiones de un tema entre sus lectores. " * 4

# two books on the same subject, one note of the user's and a chapter whose text says «redes»
LIBRARY = {
    "Cloud/Books/Libro A/02-intro.md": (
        f"# Intro A\n\n## Kafka en breve\n\n{BODY}\n\n## Brokers\n\n{BODY}"
    ),
    "Cloud/Books/Libro A/03-redes.md": f"# Redes A\n\n## Subredes\n\n{BODY}",
    "Cloud/Books/Libro B/02-intro.md": f"# Intro B\n\n## Kafka en breve\n\n{BODY}",
    "Cloud/Books/Solo.md": f"# Solo\n\n## Una sección\n\n{BODY}",
    "mis/kafka.md": f"# Mi Kafka\n\n## Notas\n\n{BODY}",
}


class _Llm:
    """Answers with the JSON each kind of request asks for."""

    def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
        user = messages[1].content
        if "opción múltiple" in user:
            yield (
                '{"pregunta": "¿Qué reparte?", "opciones": ["particiones", "a", "b", "c"], '
                '"correcta": 0, "explicacion": "Según el fragmento."}'
            )
        else:
            yield '{"pregunta": "¿Qué hace?", "respuesta": "Reparte particiones."}'

    def close(self) -> None:
        pass


@pytest.fixture
def cfg(config: Config, tmp_path: Path) -> Config:
    library = tmp_path / "library"
    for rel, text in LIBRARY.items():
        (library / rel).parent.mkdir(parents=True, exist_ok=True)
        (library / rel).write_text(text, encoding="utf-8")
    cfg = replace(
        config,
        chunking=replace(config.chunking, min_tokens=5, target_tokens=60, max_tokens=80),
        paths=replace(config.paths, library=library, index_db=tmp_path / "index.db"),
        practice=replace(
            config.practice, db=tmp_path / "practice.db", min_section_tokens=5, max_questions=5
        ),
        dictionary=replace(config.dictionary, guide_max_fragments=3),
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


@pytest.fixture
def server(cfg: Config) -> Iterator[tuple[str, int, Engine]]:
    engine = Engine(
        cfg,
        open_index(cfg, shared_across_threads=True),
        FakeEmbedder(),
        plan_resolver=lambda _c: RerankPlan(None, 0, "test"),
        llm_factory=lambda _c: _Llm(),
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
    conn = http.client.HTTPConnection(server[0], server[1], timeout=20)
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


def _docs(cands: Sequence[Any]) -> set[str]:
    return {c.section.doc_path for c in cands}


# --- what each source reads --------------------------------------------------------------------


def test_each_source_reads_its_own_part_of_the_library(cfg: Config) -> None:
    conn = open_index(cfg)
    book_a = candidate_sections(conn, cfg, "book", "Cloud/Books/Libro A")
    assert _docs(book_a) == {"Cloud/Books/Libro A/02-intro.md", "Cloud/Books/Libro A/03-redes.md"}
    chapter = candidate_sections(conn, cfg, "chapter", "Cloud/Books/Libro B/02-intro.md")
    assert _docs(chapter) == {"Cloud/Books/Libro B/02-intro.md"}
    one_note_book = candidate_sections(conn, cfg, "book", "Cloud/Books/Solo.md")
    assert _docs(one_note_book) == {"Cloud/Books/Solo.md"}
    whole = candidate_sections(conn, cfg, "library", "")
    assert _docs(whole) == {*LIBRARY_PATHS}
    # «Libro A» is not a prefix of «Libro AB»: a book never reads its neighbour
    assert not _docs(book_a) & {"Cloud/Books/Libro B/02-intro.md"}


LIBRARY_PATHS = tuple(LIBRARY)


def test_the_sources_are_checked_against_the_library(server: tuple[str, int, Engine]) -> None:
    engine = server[2]
    assert engine.valid_source("library", "")
    assert engine.valid_source("book", "Cloud/Books/Libro A")
    assert engine.valid_source("chapter", "Cloud/Books/Libro A/03-redes.md")
    assert engine.valid_source("guide", "mis/kafka.md")
    assert engine.valid_source("folder", "mis")
    assert not engine.valid_source("book", "Cloud/Books/Libro Z")
    assert not engine.valid_source("chapter", "Cloud/Books/Libro A/nada.md")
    assert not engine.valid_source("folder", "../fuera")
    assert engine.source_scope("book", "Cloud/Books/Libro A") == "Cloud/Books/Libro A"
    assert engine.source_scope("chapter", "a/b.md") == "a/b.md"
    assert engine.source_scope("library", "") is None


def test_the_guide_picker_lists_only_the_users_own_notes(server: tuple[str, int, Engine]) -> None:
    status, body = _call(server, "GET", "/api/practice/guides")
    assert status == 200
    assert body["guides"] == [{"path": "mis/kafka.md", "title": "Mi Kafka", "folder": "mis"}]


# --- the quiz ----------------------------------------------------------------------------------


def _start(server: tuple[str, int, Engine], scope: dict[str, Any], **extra: Any) -> tuple[int, Any]:
    body = {"scope": scope, "count": 3, "kinds": ["open_self"], **extra}
    return _call(server, "POST", "/api/practice/start", body)


def test_a_quiz_by_book_asks_only_about_that_book(server: tuple[str, int, Engine]) -> None:
    status, events = _start(server, {"kind": "book", "value": "Cloud/Books/Libro B"})
    assert status == 200
    questions = [q for n, q in events if n == "question"]
    assert questions and all(q["guide"] == "Intro B" for q in questions)
    status, events = _start(server, {"kind": "chapter", "value": "Cloud/Books/Libro A/03-redes.md"})
    questions = [q for n, q in events if n == "question"]
    assert status == 200 and all(q["guide"] == "Redes A" for q in questions)
    status, events = _start(server, {"kind": "library", "value": ""})
    assert status == 200 and [n for n, _ in events][-1] == "done"


def test_a_topic_can_be_looked_for_inside_one_book(server: tuple[str, int, Engine]) -> None:
    within = {"kind": "book", "value": "Cloud/Books/Libro A"}
    status, events = _start(server, {"kind": "topic", "value": "kafka"}, within=within)
    assert status == 200
    questions = [q for n, q in events if n == "question"]
    assert questions and all(q["guide"] in ("Intro A", "Redes A") for q in questions)


@pytest.mark.parametrize(
    "scope, extra",
    [
        ({"kind": "book", "value": "Cloud/Books/Libro Z"}, {}),
        ({"kind": "chapter", "value": "Cloud/Books/Libro A/nada.md"}, {}),
        ({"kind": "topic", "value": "kafka"}, {"within": {"kind": "book", "value": "no"}}),
        ({"kind": "topic", "value": "kafka"}, {"within": {"kind": "mars", "value": "x"}}),
        ({"kind": "topic", "value": "kafka"}, {"within": "Cloud"}),
    ],
)
def test_a_source_that_does_not_exist_is_refused(
    server: tuple[str, int, Engine], scope: dict[str, Any], extra: dict[str, Any]
) -> None:
    status, body = _start(server, scope, **extra)
    assert status == 400 and "error" in body


# --- «Explícamelo tú» and the dictionary ----------------------------------------------------------


def test_feynman_reads_only_the_chosen_book(server: tuple[str, int, Engine]) -> None:
    engine = server[2]
    engine.config = replace(engine.config, feynman=replace(engine.config.feynman, reason=False))
    start = {
        "concept": "grupo de consumidores",
        "scope": {"kind": "book", "value": "Cloud/Books/Libro B"},
    }

    class _Ideas:
        def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
            idea = {
                "idea": "Reparte particiones",
                "fragmento": 1,
                "cita": "reparte las particiones",
            }
            yield json.dumps({"ideas": [idea]})

        def close(self) -> None:
            pass

    engine._llm_factory = lambda _c: _Ideas()  # type: ignore[assignment,return-value]
    status, events = _call(server, "POST", "/api/feynman/start", start)
    assert status == 200 and events[0][0] == "ready"
    session = engine.practice.feynman(events[0][1]["session_id"])
    assert session is not None
    assert {f["doc_path"] for f in session.fragments} == {"Cloud/Books/Libro B/02-intro.md"}
    bad = {**start, "scope": {"kind": "book", "value": "nope"}}
    assert _call(server, "POST", "/api/feynman/start", bad)[0] == 400


def test_a_dictionary_of_a_book_is_taken_across_the_whole_book(
    server: tuple[str, int, Engine],
) -> None:
    status, events = _call(
        server, "POST", "/api/dictionary", {"kind": "book", "value": "Cloud/Books/Libro A"}
    )
    assert status == 200
    start = dict(events)["start"]
    assert start["title"] == "Libro A" and start["available"] == 3
    docs = {f["citation"].split(" › ")[0] for f in start["fragments"]}
    assert docs == {"Cloud/Books/Libro A/02-intro.md", "Cloud/Books/Libro A/03-redes.md"}
    bad = _call(server, "POST", "/api/dictionary", {"kind": "book", "value": "Cloud/Books/Zzz"})
    assert bad[0] == 400
    topic = {
        "kind": "topic",
        "value": "kafka",
        "within": {"kind": "book", "value": "Cloud/Books/Libro B"},
    }
    status, events = _call(server, "POST", "/api/dictionary", topic)
    assert status == 200
    cites = {f["citation"].split(" › ")[0] for f in dict(events)["start"]["fragments"]}
    assert cites == {"Cloud/Books/Libro B/02-intro.md"}
