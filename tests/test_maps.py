from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Sequence
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from biblio.config import Config
from biblio.generate.context import Fragment
from biblio.generate.llm import Message
from biblio.index.indexer import index_library
from biblio.maps.build import (
    CachedEmbed,
    Extraction,
    build_map,
    fold,
    map_outline,
    parse_extraction,
)
from biblio.maps.export import map_fragments, to_canvas, to_markdown
from biblio.retrieve.rerank import RerankPlan
from biblio.retrieve.store import open_index
from biblio.web.engine import Engine
from biblio.web.server import make_server

from fakes import FakeEmbedder, WordCounter

TEXT = (
    "Naive Bayes asume independencia condicional entre las variables dada la clase. "
    "Se basa en el teorema de Bayes y se usa en clasificación de texto, "
    "por ejemplo para filtrar spam."
)


def fragment(n: int = 1, text: str = TEXT, heading: str = "Teorema") -> Fragment:
    return Fragment(n, f"g{n}.md", ("Guía", heading), text, 1, 5)


# --- extraction is checked against the fragment -------------------------------------------------


def test_only_concepts_written_in_the_fragment_are_kept() -> None:
    reply = json.dumps(
        {
            "conceptos": [
                "Independencia Condicional",  # case differs: kept
                "teorema de Bayes",
                "clasificacion de texto",  # accent missing: kept
                "red neuronal",  # not in the text: dropped
                "ejemplo",  # generic: dropped
                "Bayes",
                "independencia condicional",  # duplicate: dropped
            ],
            "relaciones": [
                {"de": "Naive Bayes", "a": "teorema de Bayes", "tipo": "Se basa en."},
                {"de": "independencia condicional", "a": "teorema de Bayes", "tipo": "se usa en"},
                {"de": "red neuronal", "a": "Bayes", "tipo": "usa"},
            ],
        }
    )
    ex = parse_extraction(f"```json\n{reply}\n```", fragment(), 8)
    assert ex is not None
    assert ex.concepts == (
        "Independencia Condicional",
        "teorema de Bayes",
        "clasificacion de texto",
        "Bayes",
    )
    # "Naive Bayes" was not in the concept list, "red neuronal" not in the text
    assert ex.relations == (("Independencia Condicional", "teorema de Bayes", "se usa en"),)
    assert parse_extraction("no es JSON", fragment(), 8) is None
    assert parse_extraction('{"relaciones": []}', fragment(), 8) is None


def test_fold_ignores_accents_case_and_punctuation() -> None:
    assert fold("  Clasificación   de TEXTO.") == "clasificacion de texto"


# --- merging, hierarchy and layout --------------------------------------------------------------


def _cfg(config: Config, **kw: Any) -> Any:
    return replace(config.maps, **kw)


def test_map_merges_concepts_and_puts_the_topic_in_the_centre(config: Config) -> None:
    extractions = [
        Extraction(
            1,
            ("Naive Bayes", "independencia condicional", "teorema de Bayes", "spam"),
            (
                ("Naive Bayes", "independencia condicional", "asume"),
                ("teorema de Bayes", "spam", "se usa para"),
            ),
        ),
        Extraction(
            2,
            ("Independencia condicional", "Teorema de Bayes", "verosimilitud"),
            (("teorema de Bayes", "verosimilitud", "se calcula con"),),
        ),
    ]
    cmap = build_map("Naive Bayes", extractions, None, _cfg(config, level1_max=2))
    nodes = {n.label: n for n in cmap.nodes}
    root = cmap.nodes[0]
    assert (root.id, root.label, root.level, root.sources) == ("root", "Naive Bayes", 0, (1,))
    assert nodes["independencia condicional"].sources == (1, 2)  # merged across fragments
    assert {n.label for n in cmap.nodes if n.level == 1} == {
        "independencia condicional",
        "teorema de Bayes",
    }
    assert nodes["spam"].parent == nodes["teorema de Bayes"].id  # attached by its relation
    assert nodes["verosimilitud"].level == 2
    tree = {(e.source, e.target): e.label for e in cmap.edges if e.kind == "tree"}
    assert tree[("root", nodes["independencia condicional"].id)] == "asume"
    lines = map_outline(cmap)
    assert (
        "- **independencia condicional** [1][2] — _Naive Bayes asume independencia condicional_"
        in lines
    )
    for n in cmap.nodes[1:]:  # laid out around the centre, children further out
        assert (n.x, n.y) != (0, 0)
    r = {n.label: (n.x**2 + n.y**2) ** 0.5 for n in cmap.nodes}
    assert r["spam"] > r["teorema de Bayes"]


def test_concepts_that_contain_the_topic_stay_separate(config: Config) -> None:
    """Regression: with the topic «quimica», «química orgánica» and «bioquímica» were merged into
    the centre (they contain the word), so the map lost its main branches and their relations
    were attributed to the topic («quimica estudia carbono»)."""
    extractions = [
        Extraction(
            1,
            ("química orgánica", "química inorgánica", "bioquímica", "La Química", "carbono"),
            (("química orgánica", "carbono", "estudia"),),
        ),
        Extraction(2, ("química orgánica", "Químicas", "bioquímica"), ()),
    ]
    cmap = build_map("quimica", extractions, None, config.maps)
    nodes = {n.label: n for n in cmap.nodes}
    assert {"química orgánica", "química inorgánica", "bioquímica"} <= set(nodes)
    assert cmap.nodes[0].sources == (1, 2)  # «La Química» and «Químicas» are the topic itself
    assert "La Química" not in nodes and "Químicas" not in nodes
    edges = {(e.source, e.target, e.label) for e in cmap.edges}
    assert (nodes["química orgánica"].id, nodes["carbono"].id, "estudia") in edges
    assert not any(e.source == "root" and e.label == "estudia" for e in cmap.edges)


def test_children_need_a_relation_and_weak_words_are_dropped(config: Config) -> None:
    extractions = [
        Extraction(1, ("teorema de Bayes", "predecir", "datos dispersos"), ()),
        Extraction(2, ("teorema de Bayes", "probabilidad a priori"), ()),
        Extraction(
            3,
            ("teorema de Bayes", "probabilidad a priori", "verosimilitud"),
            (("teorema de Bayes", "verosimilitud", "se calcula con"),),
        ),
    ]
    cmap = build_map("clasificación", extractions, None, _cfg(config, level1_max=1))
    labels = {n.label: n for n in cmap.nodes}
    assert "predecir" not in labels  # one word, one fragment, no relation
    assert "datos dispersos" not in labels  # no stated relation to any parent
    assert "probabilidad a priori" not in labels  # two fragments in common are not a relation
    assert labels["verosimilitud"].parent == labels["teorema de Bayes"].id  # stated relation
    assert cmap.omitted == 3


def test_near_synonyms_merge_by_embedding(config: Config) -> None:
    vectors = {"red bayesiana": [1.0, 0.0], "redes bayesianas": [0.99, 0.05], "spam": [0.0, 1.0]}
    calls: list[list[str]] = []

    def embed(names: Sequence[str]) -> np.ndarray:
        calls.append(list(names))
        return np.array([vectors[n] for n in names], dtype=np.float32)

    cached = CachedEmbed(embed)
    extractions = [
        Extraction(1, ("red bayesiana", "spam"), ()),
        Extraction(2, ("redes bayesianas",), ()),
    ]
    cmap = build_map("tema", extractions, cached, _cfg(config, merge_similarity=0.9))
    merged = [n for n in cmap.nodes if "bayesian" in n.label]
    assert len(merged) == 1 and merged[0].sources == (1, 2)
    build_map("tema", extractions, cached, _cfg(config, merge_similarity=0.9))
    assert sum(len(c) for c in calls) == 3  # each name embedded once


# --- fragments and export -----------------------------------------------------------------------


def test_navigation_sections_are_not_read() -> None:
    from biblio.retrieve.types import Hit

    def hit(rank: int, heading: tuple[str, ...]) -> Hit:
        return Hit(rank, f"d{rank}.md", heading, "texto " * 10, 1, 2, rank, 1.0, "t")

    hits = [hit(1, ("G", "0. Tabla de Contenidos")), hit(2, ("G",)), hit(3, ("G", "Teorema"))]
    cfg_like = type("C", (), {"fragments": 5, "max_fragment_chars": 20})()
    fragments = map_fragments(hits, cfg_like)  # type: ignore[arg-type]
    assert [(f.number, f.heading_path[-1], len(f.text)) for f in fragments] == [(1, "Teorema", 20)]


def test_markdown_and_canvas_export(config: Config) -> None:
    extractions = [
        Extraction(
            1,
            ("Naive Bayes", "teorema de Bayes"),
            (("Naive Bayes", "teorema de Bayes", "se basa en"),),
        )
    ]
    cmap = build_map("Naive Bayes", extractions, None, config.maps)
    md = to_markdown(cmap, [fragment()], date(2026, 10, 18))
    assert md.startswith("# Mapa: Naive Bayes\n") and "## Fuentes\n\n1. g1.md › Teorema" in md
    canvas = to_canvas(cmap, [fragment()])
    texts = {n["id"]: n["text"] for n in canvas["nodes"]}
    assert "[[g1#Teorema|1]]" in texts["root"]
    assert canvas["edges"][0] | {"id": None} == {
        "id": None,
        "fromNode": "root",
        "toNode": canvas["nodes"][1]["id"],
        "label": "se basa en",
    }
    assert all(isinstance(n["x"], int) and n["width"] >= 140 for n in canvas["nodes"])


# --- the web API --------------------------------------------------------------------------------

GUIDES = {
    "est/bayes.md": "# Bayes\n\n## Teorema\n\n" + TEXT,
    "est/spam.md": "# Spam\n\n## Filtros\n\n"
    + "Los filtros de spam usan el teorema de Bayes y la clasificación de texto. " * 2,
}


class _Extractor:
    def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
        yield json.dumps(
            {
                "conceptos": ["teorema de Bayes", "clasificación de texto", "spam"],
                "relaciones": [{"de": "spam", "a": "teorema de Bayes", "tipo": "se filtra con"}],
            }
        )

    def close(self) -> None:
        pass


@pytest.fixture
def server(config: Config, tmp_path: Path) -> Iterator[tuple[str, int]]:
    library = tmp_path / "library"
    for rel, text in GUIDES.items():
        (library / rel).parent.mkdir(parents=True, exist_ok=True)
        (library / rel).write_text(text, encoding="utf-8")
    cfg = replace(
        config,
        chunking=replace(config.chunking, min_tokens=5, target_tokens=60, max_tokens=80),
        paths=replace(
            config.paths,
            library=library,
            index_db=tmp_path / "index.db",
            drafts_dir=tmp_path / "drafts",
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
    engine = Engine(
        cfg,
        open_index(cfg, shared_across_threads=True),
        FakeEmbedder(),
        plan_resolver=lambda _c: RerankPlan(None, 0, "test"),
        llm_factory=lambda _c: _Extractor(),
        counter_factory=lambda _c: WordCounter(),
    )
    engine.llm_running = lambda: True  # type: ignore[method-assign]
    srv = make_server(engine, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield "127.0.0.1", srv.server_address[1]
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


def _done(events: list[tuple[str, Any]]) -> Any:
    return next(d for n, d in events if n == "done")


def test_a_map_is_streamed_and_saved(server: tuple[str, int], tmp_path: Path) -> None:
    status, events = _post(server, "/api/maps", {"topic": "spam"})
    names = [n for n, _ in events]
    assert status == 200 and names[0] == "start" and names[-2:] == ["done", "gap"]  # a thin map
    assert names.count("map") == len(events[0][1]["fragments"]) == 2  # one update per fragment
    final = [d for n, d in events if n == "map"][-1]["map"]
    labels = {n["label"]: n for n in final["nodes"]}
    assert labels["spam"]["level"] == 0  # the topic is the centre
    assert labels["teorema de Bayes"]["sources"] == [1, 2]
    map_id = _done(events)["map_id"]
    status, saved = _post(server, "/api/maps/save", {"id": map_id})
    assert status == 200 and Path(saved["canvas"]).is_file()
    canvas = json.loads(Path(saved["canvas"]).read_text(encoding="utf-8"))
    assert {n["id"] for n in canvas["nodes"]} >= {"root"}
    assert Path(saved["markdown"]).parent == tmp_path / "drafts" / "maps"
    assert "teorema de Bayes" in saved["outline"]  # rebuilt from the stored map
    assert _post(server, "/api/maps/save", {"id": 999})[0] == 404
    assert _post(server, "/api/maps/save", {"id": "../x"})[0] == 400
    assert _post(server, "/api/maps", {"topic": "  "})[0] == 400


def _get(server: tuple[str, int], path: str) -> tuple[int, Any]:
    import http.client

    conn = http.client.HTTPConnection(*server, timeout=10)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = json.loads(resp.read().decode("utf-8"))
    conn.close()
    return resp.status, body


def test_maps_are_kept_searched_reopened_and_deleted(server: tuple[str, int]) -> None:
    _, events = _post(server, "/api/maps", {"topic": "spam"})
    first = _done(events)["map_id"]
    _, events = _post(server, "/api/maps", {"topic": "filtros"})
    second = _done(events)["map_id"]
    _, listed = _get(server, "/api/maps/history")
    assert [(m["id"], m["topic"]) for m in listed["maps"]] == [(second, "filtros"), (first, "spam")]
    assert listed["maps"][1]["concepts"] == 2  # «spam» is the centre, not a concept
    # search by a concept's name, accents ignored
    _, found = _get(server, "/api/maps/history?q=clasificacion")
    assert {m["id"] for m in found["maps"]} == {first, second} and "«" in found["maps"][0][
        "snippet"
    ]
    status, stored = _get(server, f"/api/maps/history/{first}")
    assert status == 200 and stored["topic"] == "spam"
    assert stored["map"]["nodes"][0]["id"] == "root" and stored["fragments"][0]["text"]
    assert _post(server, "/api/maps/delete", {"id": first})[1] == {"ok": True}
    assert _get(server, f"/api/maps/history/{first}")[0] == 404
    assert _post(server, "/api/maps/delete", {"id": first})[0] == 404
    _, listed = _get(server, "/api/maps/history?q=spam")
    assert [m["id"] for m in listed["maps"]] == [second]  # «spam» is a concept of «filtros»


def test_moved_nodes_are_kept_and_used_when_saving(server: tuple[str, int], tmp_path: Path) -> None:
    _, events = _post(server, "/api/maps", {"topic": "spam"})
    map_id = _done(events)["map_id"]
    _, stored = _get(server, f"/api/maps/history/{map_id}")
    assert stored["positions"] == {}
    child = stored["map"]["nodes"][1]["id"]
    moved = {"positions": {child: [1234.4, -55], "no-such-node": [1, 2]}, "id": map_id}
    assert _post(server, "/api/maps/positions", moved)[1] == {"ok": True}
    _, stored = _get(server, f"/api/maps/history/{map_id}")
    assert stored["positions"] == {child: [1234, -55]}  # unknown ids are ignored
    assert stored["map"]["nodes"][1]["x"] != 1234  # the automatic layout is kept apart
    _, saved = _post(server, "/api/maps/save", {"id": map_id})
    canvas = json.loads(Path(saved["canvas"]).read_text(encoding="utf-8"))
    node = next(n for n in canvas["nodes"] if n["id"] == child)
    assert node["x"] + node["width"] // 2 == 1234  # the canvas uses where the user left it
    assert _post(server, "/api/maps/positions", {"id": map_id, "positions": {}})[1] == {"ok": True}
    assert _get(server, f"/api/maps/history/{map_id}")[1]["positions"] == {}
    for bad in ({"x": "1,2"}, {"x": [1]}, {"x": [True, 2]}, []):
        assert _post(server, "/api/maps/positions", {"id": map_id, "positions": bad})[0] == 400
    assert _post(server, "/api/maps/positions", {"id": 999, "positions": {}})[0] == 404


def test_an_older_maps_db_gains_the_positions_column(tmp_path: Path) -> None:
    import sqlite3

    from biblio.maps.history import MapStore

    path = tmp_path / "maps.db"
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE maps (id INTEGER PRIMARY KEY, topic TEXT NOT NULL, created TEXT NOT NULL, "
        "concepts INTEGER NOT NULL, names TEXT NOT NULL, map TEXT NOT NULL, "
        "fragments TEXT NOT NULL)"
    )
    page_map = {"topic": "t", "nodes": [{"id": "root", "label": "t", "level": 0}], "edges": []}
    old.execute(
        "INSERT INTO maps VALUES (1, 't', '2026-10-19', 0, '', ?, '[]')", (json.dumps(page_map),)
    )
    old.commit()
    old.close()
    store = MapStore(path)
    stored = store.get(1)
    assert stored is not None and stored.positions == {}
    assert stored.summary.kind == "concepts"  # and the kind column (ADR-036)
    assert store.set_positions(1, {"root": (5, 6)}) and store.get(1).positions == {"root": (5, 6)}  # type: ignore[union-attr]


# --- mastery map (ADR-026) ----------------------------------------------------------------------


def test_mastery_follows_the_practice_of_each_concepts_sections(
    server: tuple[str, int], config: Config
) -> None:
    from datetime import datetime, timedelta

    from biblio.practice.store import PracticeStore, QuestionContent, Section

    _, events = _post(server, "/api/maps", {"topic": "spam"})
    map_id = _done(events)["map_id"]
    _, stored = _get(server, f"/api/maps/history/{map_id}")
    _, mastery = _get(server, f"/api/maps/history/{map_id}/mastery")
    assert {n["state"] for n in mastery["nodes"].values()} == {"new"}  # nothing practised yet

    store = PracticeStore(config.practice.db)  # the same file the server uses
    first = stored["fragments"][0]
    section = Section(
        first["doc_path"], (first["guide"], *first["sections"]), *first["lines"], first["text"]
    )
    quiz = store.new_quiz("guide", first["doc_path"], 1)
    qid = store.add_question(
        quiz, "mc", section, QuestionContent("q", "e", ("a", "b", "c", "d"), 0)
    )
    store.record_attempt(qid, "0", "correct", "code")
    node = next(n for n in stored["map"]["nodes"] if n["sources"] == [1, 2])  # in both fragments
    _, mastery = _get(server, f"/api/maps/history/{map_id}/mastery")
    info = mastery["nodes"][node["id"]]
    assert info["state"] == "mastered" and info["practiced"] == 1 and info["recall"] > 0.95
    assert len(info["sections"]) == 2  # «Practicar esto» gets both of its sections

    # two months later, without reviews, the memory has faded
    old = (datetime.now().astimezone() - timedelta(days=60)).isoformat(timespec="seconds")
    store._conn.execute("UPDATE attempts SET answered = ?", (old,))
    store._conn.commit()
    _, mastery = _get(server, f"/api/maps/history/{map_id}/mastery")
    assert mastery["nodes"][node["id"]]["state"] in ("partial", "weak")
    assert _get(server, "/api/maps/history/999/mastery")[0] == 404
    store.close()
