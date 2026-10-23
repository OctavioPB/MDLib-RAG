from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Sequence
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any, ClassVar

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
from biblio.maps.select import map_candidates, order_sections
from biblio.retrieve.rerank import RerankPlan
from biblio.retrieve.store import open_index
from biblio.retrieve.types import Hit
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


# --- notes in English (ADR-042) -----------------------------------------------------------------

BURNOUT = (
    "The 2024 Gallup report estimates that employee burnout costs the global economy USD 8.8 "
    "trillion. Direct replacement cost is between 50% and 200% of the salary. Team disruption "
    "follows when a high-performer leaves."
)


def test_english_notes_keep_their_names_with_a_spanish_one_beside_them() -> None:
    """Regression: told to write names «as in the text», the 4B model translated a whole English
    section («costo de reemplazo»), every name failed the check and the map came out empty."""
    reply = json.dumps(
        {
            "tema": "burnout",
            "conceptos": [
                {"nombre": "replacement cost", "es": "costo de reemplazo"},
                {"nombre": "team disruption", "es": "Team disruption"},  # no translation: none
                {"nombre": "costo de reemplazo", "es": ""},  # translated: not in the text
                "burnout",  # a plain name is still read
            ],
            "relaciones": [{"de": "burnout", "a": "replacement cost", "tipo": "causa"}],
        }
    )
    ex = parse_extraction(reply, fragment(text=BURNOUT), 8)
    assert ex is not None
    assert ex.concepts == ("replacement cost", "team disruption", "burnout")
    assert ex.translations == (("replacement cost", "costo de reemplazo"),)
    assert ex.topic_names == ("burnout",)
    # a topic name the fragment does not write is ignored
    other = parse_extraction(json.dumps({"tema": "agotamiento", "conceptos": []}), fragment(), 8)
    assert other is not None and other.topic_names == ()


def test_the_topic_as_an_english_note_names_it_is_the_centre(config: Config) -> None:
    extractions = [
        Extraction(
            1,
            ("burnout", "replacement cost", "team disruption"),
            (("burnout", "replacement cost", "causa"), ("burnout", "team disruption", "causa")),
            (("replacement cost", "costo de reemplazo"),),
            ("burnout",),
        ),
        # a Spanish note writing the Spanish name is the same concept
        Extraction(2, ("costo de reemplazo", "rotación"), (), (), ()),
    ]
    cmap = build_map("agotamiento laboral", extractions, None, _cfg(config))
    root = cmap.nodes[0]
    assert root.label == "agotamiento laboral" and root.sources == (1,)  # «burnout» is the centre
    nodes = {n.label: n for n in cmap.nodes}
    assert "burnout" not in nodes
    cost = nodes["replacement cost"]
    assert (cost.translation, cost.sources, cost.parent) == ("costo de reemplazo", (1, 2), "root")
    assert any("replacement cost (costo de reemplazo)" in line for line in map_outline(cmap))


def test_english_text_is_told_apart_from_spanish() -> None:
    from biblio.maps.build import in_english

    assert in_english(BURNOUT)
    assert not in_english(TEXT)


# --- the sections a map reads (ADR-051, stage 1) ------------------------------------------------


def _hit(chunk: int, heading_path: tuple[str, ...], rank: int) -> Hit:
    return Hit(
        chunk, f"n{chunk}.md", heading_path, "texto sobre la api", chunk, chunk, rank, 0.0, "t"
    )


def test_a_map_reads_the_sections_that_explain_the_topic_first(config: Config) -> None:
    cfg = config.maps
    folder = _hit(1, ("Proyecto", "Instalación"), 1)  # mentions the topic, does not explain it
    product = _hit(
        2, ("API de Anthropic", "Gobernanza"), 2
    )  # the guide's title names it: not enough
    what = _hit(3, ("Guía", "¿Qué es una API?"), 1)
    asked: list[tuple[str, int]] = []

    def search(query: str, k: int) -> list[Hit]:
        asked.append((query, k))
        return [folder, product] if query == "api" else [what, folder]

    hits = map_candidates(search, "api", cfg, rrf_k=60)
    assert asked == [
        ("api", cfg.candidates),
        (cfg.define_query.format(topic="api"), cfg.candidates),
    ]
    assert [h.chunk_id for h in hits] == [1, 3, 2]  # fused: found by both queries first
    # without the reranker, a heading that names the topic goes first; the rest keep their order
    assert [h.chunk_id for h in order_sections("api", hits, None)] == [3, 1, 2]
    # with it, its scores order each group
    assert [h.chunk_id for h in order_sections("api", hits, [0.1, 0.2, 0.9])] == [3, 2, 1]


class _CpuReranker:
    calls: ClassVar[list[tuple[str, str, int]]] = []
    closed: ClassVar[int] = 0

    def __init__(self, device: str) -> None:
        self.device = device

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        _CpuReranker.calls.append((self.device, query, len(passages)))
        return [float("Filtros" in p) for p in passages]

    def close(self) -> None:
        _CpuReranker.closed += 1


def test_the_map_reranks_its_candidates_on_cpu_and_releases_it(
    config: Config, tmp_path: Path
) -> None:
    library = tmp_path / "library"
    for rel, text in GUIDES.items():
        (library / rel).parent.mkdir(parents=True, exist_ok=True)
        (library / rel).write_text(text, encoding="utf-8")
    cfg = replace(
        config,
        chunking=replace(config.chunking, min_tokens=5, target_tokens=60, max_tokens=80),
        paths=replace(config.paths, library=library, index_db=tmp_path / "index.db"),
        maps=replace(config.maps, rerank=True, history_db=tmp_path / "maps.db"),
    )
    index_library(
        cfg,
        library,
        full=True,
        dtype="float32",
        embedder_factory=FakeEmbedder,
        counter_factory=WordCounter,
    )
    _CpuReranker.calls.clear()
    _CpuReranker.closed = 0
    engine = Engine(
        cfg,
        open_index(cfg, shared_across_threads=True),
        FakeEmbedder(),
        reranker_factory=lambda _c, device: _CpuReranker(device),
        plan_resolver=lambda _c: RerankPlan(None, 0, "test"),
        llm_factory=lambda _c: _Extractor(),
        counter_factory=lambda _c: WordCounter(),
    )
    try:
        events = list(engine.concept_map("spam"))
    finally:
        engine.close()
    names = [n for n, _ in events]
    assert names[:2] == ["selecting", "start"]  # the page says what the wait is for
    assert _CpuReranker.calls == [("cpu", cfg.maps.define_query.format(topic="spam"), 2)]
    assert _CpuReranker.closed == 1  # released before the LLM reads: +2.5 GiB only meanwhile
    first = events[1][1]["fragments"][0]
    assert first["doc_path"] == "est/spam.md"  # the one the reranker preferred
    # one of the two sections explains «spam» in general: the map says so (ADR-051 stage 3)
    gap = next(d for n, d in events if n == "gap")
    assert (gap["general"], gap["read"]) == (1, 2) and not gap["absent"]


# --- the topic's acronym and the LLM's organisation (ADR-051, stage 2) ---------------------------


def test_the_topic_spelled_out_where_the_notes_write_it_is_the_topic(config: Config) -> None:
    from biblio.maps.build import spelled_out_in

    assert spelled_out_in("api", "Una **API (Application Programming Interface)** es") == [
        "Application Programming Interface"
    ]
    assert spelled_out_in("API", "la interfaz de programación de aplicaciones (API) define") == [
        "interfaz de programación de aplicaciones"  # Spanish word order
    ]
    assert (
        spelled_out_in("api", "El aprendizaje por imitación no es una API.") == []
    )  # initials only
    assert spelled_out_in("costo de oportunidad", "costo de oportunidad (CO)") == []
    # in a map, the spelled-out name joins the topic instead of becoming a branch
    text = "Una API (Application Programming Interface) es un contrato con un endpoint."
    ex = Extraction(
        1,
        ("application programming interface", "contrato", "endpoint"),
        (("application programming interface", "contrato", "es un"),),
        topic_names=tuple(spelled_out_in("api", text)),
    )
    cmap = build_map("api", [ex], None, config.maps)
    labels = {n.label: n.level for n in cmap.nodes}
    assert "application programming interface" not in labels and labels["contrato"] == 1


def test_the_llm_only_chooses_branches_by_name_and_notes_place_the_children(
    config: Config,
) -> None:
    from biblio.maps.build import group_under, inventory, parse_selection

    by_name = {"endpoint": "c1", "rate limiting": "c2", "json": "c3"}
    reply = (
        'Listo: {"clave": ["API", "Endpoint", "endpoint", "inventado", "rate limiting", "json"]}'
    )
    # the topic, a name not listed and a repeat are dropped; the cap holds
    assert parse_selection(reply, by_name, 2) == ["c1", "c2"]
    assert parse_selection("no es json", by_name, 9) == []
    exs = [
        Extraction(1, ("endpoint", "solicitud", "json"), (("solicitud", "endpoint", "llega a"),)),
        Extraction(2, ("rate limiting", "cliente"), ()),
        Extraction(3, ("kong",), ()),  # tied to no branch: left out, not guessed
    ]
    inv = inventory("api", exs, None, config.maps)
    ids = {inv.others[n].label: n for n in inv.others}
    children = group_under(inv, [ids["endpoint"], ids["rate limiting"]], config.maps)
    named = {inv.others[p].label: [inv.others[c].label for c in v] for p, v in children.items()}
    assert named == {"endpoint": ["solicitud", "json"], "rate limiting": ["cliente"]}


class _Organizer:
    """Extracts like `_Extractor`; asked for the key concepts, names «clasificación de texto»."""

    def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
        if "Eliges los conceptos clave" in messages[0].content:
            yield json.dumps({"clave": ["spam", "clasificación de texto"]})  # the topic dropped
        else:
            yield from _Extractor().stream(messages, max_tokens)

    def close(self) -> None:
        pass


def test_the_last_map_is_organized_by_the_llm(config: Config) -> None:
    from biblio.maps.build import generate_map

    cfg = replace(config.maps, organize=True, organize_min_branches=1)
    progress = list(generate_map(_Organizer(), cfg, "spam", [fragment(1), fragment(2)], None))
    assert [p.stage for p in progress] == ["", "organizing", ""]  # one more map, organized
    final = progress[-1].map
    level1 = [n.label for n in final.nodes if n.level == 1]
    assert level1 == ["clasificación de texto"]
    child = next(n for n in final.nodes if n.level == 2)
    assert child.label == "teorema de Bayes" and child.parent == next(
        n.id for n in final.nodes if n.level == 1
    )
    # a grouping the notes do not state gets no label: the relation is never the model's
    edge = next(e for e in final.edges if child.id in (e.source, e.target) and e.kind == "tree")
    assert edge.label == ""


class _ChildPicker:
    """Asked about «endpoint», names one tied concept and one that is not tied; about
    «rate limiting», answers nothing usable."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
        user = messages[1].content
        self.asked.append(user)
        if "Rama: endpoint" in user:
            yield json.dumps(
                {"hijos": ["json", "cliente"]}
            )  # «cliente» is tied to the other branch
        else:
            yield "no sé"

    def close(self) -> None:
        pass


def test_the_llm_confirms_children_only_among_those_tied_to_the_branch(config: Config) -> None:
    from biblio.maps.build import ask_children, inventory

    exs = [
        Extraction(1, ("endpoint", "solicitud", "json"), (("solicitud", "endpoint", "llega a"),)),
        Extraction(2, ("rate limiting", "cliente"), ()),
    ]
    inv = inventory("api", exs, None, config.maps)
    ids = {inv.others[n].label: n for n in inv.others}
    llm = _ChildPicker()
    children = ask_children(llm, config.maps, inv, [ids["endpoint"], ids["rate limiting"]])
    named = {inv.others[p].label: [inv.others[c].label for c in v] for p, v in children.items()}
    # «solicitud» was offered and not chosen; «cliente» was never offered to «endpoint»
    assert named == {"endpoint": ["json"], "rate limiting": ["cliente"]}  # unusable: ties decide
    assert "- json\n- solicitud" in llm.asked[0] and "cliente" not in llm.asked[0]


def test_a_fragment_explains_the_topic_by_score_or_by_heading_and_the_llm() -> None:
    from biblio.maps.select import count_general

    scored = Fragment(1, "a.md", ("Guía", "Definición"), "texto", 1, 2)  # high score
    named = Fragment(2, "b.md", ("Guía", "¿Qué es un webhook?"), "texto", 1, 2)  # heading names it
    other = Fragment(3, "c.md", ("Proyecto", "Decisiones"), "texto", 1, 2)  # neither
    asked: list[int] = []

    def focus(f: Fragment) -> str:
        asked.append(f.number)
        return "general"

    scores = {("a.md", 1): 0.9, ("b.md", 1): 0.1, ("c.md", 1): 0.1}
    assert count_general("webhook", [scored, named, other], scores, 0.4, focus) == 2
    assert asked == [2]  # the LLM is asked only about a low score whose heading names the topic
