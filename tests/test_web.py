from __future__ import annotations

import http.client
import json
import re
import threading
from collections.abc import Iterator, Sequence
from dataclasses import replace
from importlib import resources
from pathlib import Path
from typing import Any, ClassVar

import pytest

from biblio.config import Config
from biblio.generate.llm import Message
from biblio.index.embedder import DeviceChoice
from biblio.index.indexer import index_library
from biblio.retrieve.rerank import RerankPlan
from biblio.retrieve.store import open_index
from biblio.web.engine import Engine
from biblio.web.server import LoopbackOnlyError, make_server

from fakes import FakeEmbedder, WordCounter

GUIDES = {
    "est/bayes.md": "# Bayes\n\n## Teorema\n\n"
    + "El teorema de Bayes invierte probabilidades condicionadas usando la marginal. " * 3,
    "alg/svd.md": "# SVD\n\n## Definición\n\n"
    + "La descomposición en valores singulares factoriza cualquier matriz real. " * 3,
}


# every file the page loads from /static/ (each must load nothing from outside)
ASSETS = (
    "app.js",
    "app.css",
    "theme.js",
    "guias.js",
    "practicar.js",
    "feynman.js",
    "plan.js",
    "reader.js",
    "library.js",
    "dictionary.js",
    "devices.js",
    "settings.js",
    "mapas.js",
    "i18n.js",
    "i18n-en.js",
    "spotter.js",
)


class _Reranker:
    instances: ClassVar[list[_Reranker]] = []

    def __init__(self, device: str) -> None:
        self.device = device
        self.closed = False
        _Reranker.instances.append(self)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [float("bayes" in p.lower()) for p in passages]

    def close(self) -> None:
        self.closed = True


class _LLM:
    def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
        yield from ["El teorema ", "invierte probabilidades [1]."]

    def close(self) -> None:
        pass


@pytest.fixture
def engine(config: Config, tmp_path: Path) -> Iterator[Engine]:
    library = tmp_path / "library"
    for rel, text in GUIDES.items():
        (library / rel).parent.mkdir(parents=True, exist_ok=True)
        (library / rel).write_text(text, encoding="utf-8")
    cfg = replace(
        config,
        chunking=replace(config.chunking, min_tokens=5, target_tokens=40, max_tokens=60),
        paths=replace(config.paths, library=library, index_db=tmp_path / "index.db"),
    )
    index_library(
        cfg,
        library,
        full=True,
        dtype="float32",
        embedder_factory=FakeEmbedder,
        counter_factory=WordCounter,
    )
    plan = {"value": RerankPlan("cuda", 40)}
    _Reranker.instances.clear()
    engine = Engine(
        cfg,
        open_index(cfg, shared_across_threads=True),
        FakeEmbedder(),
        reranker_factory=lambda _c, device: _Reranker(device),
        plan_resolver=lambda _c: plan["value"],
        llm_factory=lambda _c: _LLM(),
        counter_factory=lambda _c: WordCounter(),
        index_embedder_factory=lambda _c, _d: FakeEmbedder(),
        index_counter_factory=lambda _c: WordCounter(),
        index_device=lambda _c: DeviceChoice("cpu", "float32"),
    )
    engine.plan = plan  # type: ignore[attr-defined]  # lets tests flip the GPU plan
    yield engine
    engine.close()


@pytest.fixture
def server(engine: Engine) -> Iterator[tuple[str, int]]:
    srv = make_server(engine, "127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield "127.0.0.1", srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def request(
    server: tuple[str, int],
    method: str,
    path: str,
    body: Any = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    host, port = server
    conn = http.client.HTTPConnection(host, port, timeout=10)
    data = json.dumps(body).encode() if body is not None else None
    hdrs = {"Content-Type": "application/json"} if data is not None else {}
    hdrs |= headers or {}
    conn.request(method, path, body=data, headers=hdrs)
    resp = conn.getresponse()
    payload = resp.read()
    conn.close()
    return resp.status, {k.lower(): v for k, v in resp.getheaders()}, payload


# --- static files and security ------------------------------------------------------------------


def test_page_and_assets_are_served_with_a_strict_policy(server: tuple[str, int]) -> None:
    status, headers, body = request(server, "GET", "/")
    assert status == 200 and b"<title>Biblioteca</title>" in body
    assert "script-src 'self'" in headers["content-security-policy"]
    assert headers["x-content-type-options"] == "nosniff"
    assert "access-control-allow-origin" not in headers
    for name in ASSETS:
        assert request(server, "GET", f"/static/{name}")[0] == 200
    assert request(server, "GET", "/static/../server.py")[0] == 404


def test_static_assets_reference_no_external_url() -> None:
    """ADR-002: the UI must work with the network off, so nothing may load from outside."""
    static = resources.files("biblio.web") / "static"
    for name in ("index.html", *ASSETS):
        text = (static / name).read_text(encoding="utf-8")
        assert not re.search(r"https?://|//cdn|@import", text), name


def test_foreign_host_header_is_rejected(server: tuple[str, int]) -> None:
    status, _, _ = request(server, "GET", "/api/status", headers={"Host": "evil.example:80"})
    assert status == 403


def test_api_requires_json_and_valid_fields(server: tuple[str, int]) -> None:
    host, port = server
    conn = http.client.HTTPConnection(host, port, timeout=10)
    conn.request("POST", "/api/search", body=b"query=x", headers={"Content-Type": "text/plain"})
    assert conn.getresponse().status == 415
    conn.close()
    assert request(server, "POST", "/api/search", {"query": "  "})[0] == 400
    assert request(server, "POST", "/api/search", {"query": "x" * 5000})[0] == 400
    assert request(server, "POST", "/api/nope", {"query": "x"})[0] == 404


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "example.com"])
def test_refuses_to_listen_outside_loopback(engine: Engine, host: str) -> None:
    with pytest.raises(LoopbackOnlyError):
        make_server(engine, host, 0)


# --- search and the reranker's GPU rule ---------------------------------------------------------


def test_search_returns_cards_with_obsidian_links(server: tuple[str, int]) -> None:
    status, _, body = request(server, "POST", "/api/search", {"query": "teorema de bayes"})
    data = json.loads(body)
    assert status == 200 and data["retriever"] == "hybrid-rerank"
    top = data["hits"][0]
    assert top["doc_path"] == "est/bayes.md" and top["citation"] == "est/bayes.md › Teorema"
    assert top["open_url"].startswith("obsidian://open?path=")
    assert "%2F" in top["open_url"] or "%5C" in top["open_url"]  # the path is percent-encoded


def test_search_hits_carry_the_full_text_and_its_place(server: tuple[str, int]) -> None:
    _, _, body = request(server, "POST", "/api/search", {"query": "teorema de bayes"})
    top = json.loads(body)["hits"][0]
    assert top["guide"] == "Bayes" and top["sections"] == ["Teorema"]
    assert top["text"].count("El teorema de Bayes") == 3  # whole passage for the sources panel


def test_folders_and_scoped_search(server: tuple[str, int]) -> None:
    status, _, body = request(server, "GET", "/api/folders")
    assert status == 200
    assert json.loads(body)["folders"] == [
        {"path": "alg", "documents": 1},
        {"path": "est", "documents": 1},
    ]
    scoped = {"query": "teorema de bayes", "scope": "alg"}
    _, _, body = request(server, "POST", "/api/search", scoped)
    assert {h["doc_path"] for h in json.loads(body)["hits"]} == {"alg/svd.md"}
    status, _, _ = request(server, "POST", "/api/search", {"query": "x", "scope": "../etc"})
    assert status == 400


def test_fast_precision_skips_the_reranker(server: tuple[str, int]) -> None:
    _, _, body = request(server, "POST", "/api/search", {"query": "bayes", "precision": "fast"})
    assert json.loads(body)["retriever"] == "hybrid" and not _Reranker.instances
    status, _, _ = request(server, "POST", "/api/search", {"query": "x", "precision": "max"})
    assert status == 400


def test_reranker_follows_the_gpu_plan_per_request(engine: Engine) -> None:
    engine.search("bayes", 3)
    engine.search("bayes", 3)
    assert len(_Reranker.instances) == 1  # loaded once, kept warm
    engine.plan["value"] = RerankPlan(None, 0, "LLM running")  # type: ignore[attr-defined]
    result = engine.search("bayes", 3)
    assert result.retriever == "hybrid" and _Reranker.instances[0].closed


# --- ask ----------------------------------------------------------------------------------------


def _events(body: bytes) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in body.decode("utf-8").strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_ask_streams_meta_tokens_and_validation(
    server: tuple[str, int], engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine.search("bayes", 3)  # loads the reranker on the "GPU"
    monkeypatch.setattr(engine, "llm_running", lambda: True)
    status, headers, body = request(server, "POST", "/api/ask", {"question": "¿Qué hace Bayes?"})
    assert status == 200 and headers["content-type"].startswith("text/event-stream")
    events = _events(body)
    names = [name for name, _ in events]
    assert names[0] == "meta" and names[-1] == "done" and names.count("token") == 2
    meta, done = events[0][1], events[-1][1]
    assert meta["fragments"][0]["number"] == 1
    assert done["grounded"] and done["cited"] == [1] and not done["abstained"]
    assert _Reranker.instances[0].closed  # released before the LLM ran (ADR-014)


def test_ask_honours_scope_and_strategy(
    server: tuple[str, int], engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(engine, "llm_running", lambda: True)
    body = {"question": "¿Qué hace Bayes?", "scope": "alg"}
    _, _, raw = request(server, "POST", "/api/ask", body)
    meta = _events(raw)[0][1]
    assert {f["doc_path"] for f in meta["fragments"]} == {"alg/svd.md"}
    assert meta["fragments"][0]["text"] and meta["fragments"][0]["guide"] == "SVD"
    status, _, _ = request(server, "POST", "/api/ask", {**body, "strategy": "magic"})
    assert status == 400


def test_ask_without_llm_server_is_unavailable(
    server: tuple[str, int], engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(engine, "llm_running", lambda: False)
    status, _, body = request(server, "POST", "/api/ask", {"question": "¿algo?"})
    assert status == 503 and "serve-llm" in json.loads(body)["error"]


def test_status(server: tuple[str, int], engine: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "llm_running", lambda: False)
    status, _, body = request(server, "GET", "/api/status")
    data = json.loads(body)
    assert status == 200 and data["documents"] == 2 and data["llm_running"] is False


def test_malformed_bodies_get_a_400_not_a_dropped_connection(server: tuple[str, int]) -> None:
    host, port = server
    latin1 = '{"query": "¿Qué?"}'.encode("latin-1")  # not valid UTF-8
    for raw in (latin1, b"{not json", b"[1, 2]"):
        conn = http.client.HTTPConnection(host, port, timeout=10)
        conn.request("POST", "/api/search", body=raw, headers={"Content-Type": "application/json"})
        assert conn.getresponse().status == 400, raw
        conn.close()


# --- Guías: drafts (ADR-019) --------------------------------------------------------------------


class _ScriptedLLM:
    """Replies in order, one reply per call, streamed in two tokens."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)

    def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        half = len(reply) // 2
        yield from (reply[:half], reply[half:])

    def close(self) -> None:
        pass


def _llm_on(engine: Engine, monkeypatch: pytest.MonkeyPatch, *replies: str) -> None:
    monkeypatch.setattr(engine, "llm_running", lambda: True)
    monkeypatch.setattr(engine, "_llm_factory", lambda _c: _ScriptedLLM(*replies))


def test_drafts_folder_listing(tmp_path: Path) -> None:
    from biblio.web.drafts import list_drafts, read_draft

    (tmp_path / "solo-indice.outline.md").write_text(
        "# Esquema: Solo índice\n\n## A\nnota\n\n## B\n", encoding="utf-8"
    )
    (tmp_path / "escrita.md").write_text(
        "# Escrita\n\n> aviso\n\n## Uno\n\nTexto [1].\n\n## Dos\n\nMás [2].\n\n"
        "## Fuentes\n\n1. a.md › X (líneas 1-2)\n2. b.md › Y (líneas 3-4)\n",
        encoding="utf-8",
    )
    (tmp_path / "Mal Nombre.md").write_text("# x\n", encoding="utf-8")
    drafts = {d.slug: d for d in list_drafts(tmp_path)}
    assert set(drafts) == {"solo-indice", "escrita"}
    assert drafts["solo-indice"].topic == "Solo índice" and drafts["solo-indice"].sections == 2
    assert not drafts["solo-indice"].has_draft
    assert (drafts["escrita"].sections, drafts["escrita"].sources) == (2, 2)
    detail = read_draft(tmp_path, "solo-indice")
    assert detail is not None and detail.outline is not None
    assert [s.notes for s in detail.outline.sections] == ["nota", ""]
    assert read_draft(tmp_path, "../solo-indice") is None
    assert read_draft(tmp_path, "no-existe") is None


def test_outline_is_streamed_and_saved(
    server: tuple[str, int], engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    _llm_on(engine, monkeypatch, "## Teorema\nla fórmula\n## Índice\n## Aplicaciones\n")
    status, _, raw = request(server, "POST", "/api/drafts/outline", {"topic": "Bayes"})
    events = _events(raw)
    assert status == 200 and [n for n, _ in events].count("token") == 2
    outline = events[-1][1]
    assert events[-1][0] == "outline" and outline["slug"] == "bayes"
    assert outline["sections"] == [
        {"title": "Teorema", "notes": "la fórmula"},
        {"title": "Aplicaciones", "notes": ""},
    ]  # the navigation section was dropped
    assert "est/bayes.md" in outline["siblings"]
    _, _, body = request(server, "GET", "/api/drafts")
    listed = json.loads(body)["drafts"]
    assert [(d["slug"], d["has_outline"], d["has_draft"]) for d in listed] == [
        ("bayes", True, False)
    ]


def test_draft_is_written_section_by_section(
    server: tuple[str, int], engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    _llm_on(engine, monkeypatch, "Invierte probabilidades condicionadas [1].")
    sections = [{"title": "Teorema", "notes": "con la fórmula"}, {"title": "  Usos \n "}]
    status, _, raw = request(
        server, "POST", "/api/drafts/write", {"topic": "Bayes", "sections": sections}
    )
    events = _events(raw)
    names = [n for n, _ in events]
    assert status == 200 and names[0] == "start" and names[-1] == "done"
    done_sections = [d for n, d in events if n == "section_done"]
    assert [d["title"] for d in done_sections] == ["Teorema", "Usos"]
    assert done_sections[0]["cited"] == [1] and not done_sections[0]["uncited"]
    done = events[-1][1]
    assert done["markdown"].startswith("# Bayes\n") and "## Usos" in done["markdown"]
    assert done["sources"][0]["number"] == 1 and done["sources"][0]["text"]
    _, _, body = request(server, "GET", "/api/drafts/bayes")
    detail = json.loads(body)
    assert detail["markdown"] == done["markdown"]
    assert detail["outline"]["sections"][1] == {"title": "Usos", "notes": ""}


def test_draft_requests_are_validated(
    server: tuple[str, int], engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    _llm_on(engine, monkeypatch, "x")
    for sections in ([], [{"title": ""}], [{"notes": "x"}], "A", [{"title": "x" * 300}]):
        body = {"topic": "Bayes", "sections": sections}
        assert request(server, "POST", "/api/drafts/write", body)[0] == 400, sections
    assert request(server, "GET", "/api/drafts/..%2Fsecret")[0] == 404
    engine._generating.acquire()  # another generation in progress
    try:
        status, _, body = request(server, "POST", "/api/drafts/outline", {"topic": "Bayes"})
        assert status == 409 and "espera" in json.loads(body)["error"]
    finally:
        engine._generating.release()


def test_closing_a_stream_stops_the_generation() -> None:
    from biblio.web.engine import _streamed

    closed, finished = threading.Event(), threading.Event()
    seen: list[str] = []

    def work(on_token: Any) -> str:
        try:
            on_token("a")
            seen.append("a")
            closed.wait(5)  # the page goes away while the model is still generating
            on_token("b")
            seen.append("b")
            return "done"
        finally:
            finished.set()

    def consume() -> Iterator[tuple[str, dict[str, Any]]]:
        yield from _streamed(work, lambda t: ("token", {"text": t}))

    stream = consume()
    assert next(stream) == ("token", {"text": "a"})
    stream.close()
    closed.set()
    assert finished.wait(5)
    assert seen == ["a"]  # the next token raised inside the worker instead of being generated


# --- conversation history (ADR-023) -------------------------------------------------------------


def test_messages_are_stored_and_listed(
    server: tuple[str, int], engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(engine, "llm_running", lambda: True)
    _, _, raw = request(server, "POST", "/api/ask", {"question": "¿Qué hace Bayes?"})
    done = _events(raw)[-1][1]
    cid = done["conversation_id"]
    _, _, body = request(
        server, "POST", "/api/search", {"query": "valores singulares", "conversation_id": cid}
    )
    assert json.loads(body)["conversation_id"] == cid
    _, _, body = request(server, "GET", "/api/conversations")
    (listed,) = json.loads(body)["conversations"]
    assert (listed["id"], listed["title"], listed["turns"]) == (cid, "¿Qué hace Bayes?", 2)
    _, _, body = request(server, "GET", f"/api/conversations/{cid}")
    detail = json.loads(body)
    first, second = detail["turns"]
    assert first["answer"] == "El teorema invierte probabilidades [1]."
    assert first["payload"]["fragments"][0]["text"] and first["payload"]["done"]["grounded"]
    assert second["mode"] == "find" and second["payload"]["hits"][0]["text"]  # shown as it was
    _, _, body = request(server, "GET", "/api/conversations?q=singulares")
    assert [c["id"] for c in json.loads(body)["conversations"]] == [cid]
    assert request(server, "GET", "/api/conversations/999")[0] == 404
    assert request(server, "POST", "/api/search", {"query": "x", "conversation_id": "1"})[0] == 400


def test_a_failed_answer_is_stored_with_its_conversation(
    server: tuple[str, int], engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Broken:
        def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
            raise RuntimeError("el modelo se cayó")
            yield ""

        def close(self) -> None:
            pass

    monkeypatch.setattr(engine, "llm_running", lambda: True)
    monkeypatch.setattr(engine, "_llm_factory", lambda _c: _Broken())
    _, _, raw = request(server, "POST", "/api/ask", {"question": "¿Y esto?"})
    name, data = _events(raw)[-1]
    assert name == "error" and "se cayó" in data["error"]
    stored = engine.conversations.get(data["conversation_id"])
    assert stored is not None and stored[1][0].status == "error"


def test_rename_export_and_delete_a_conversation(
    server: tuple[str, int], engine: Engine, tmp_path: Path
) -> None:
    _, _, body = request(server, "POST", "/api/search", {"query": "teorema de bayes"})
    cid = json.loads(body)["conversation_id"]
    assert (
        request(server, "POST", "/api/conversations/rename", {"id": cid, "title": "Repaso"})[0]
        == 200
    )
    status, _, body = request(server, "POST", "/api/conversations/export", {"id": cid})
    exported = json.loads(body)
    assert status == 200 and exported["markdown"].startswith("# Repaso\n")
    assert Path(exported["path"]).parent == engine.config.paths.drafts_dir / "conversations"
    assert request(server, "POST", "/api/conversations/delete", {"id": cid})[0] == 200
    assert request(server, "POST", "/api/conversations/delete", {"id": cid})[0] == 404
    assert request(server, "POST", "/api/conversations/rename", {"id": cid, "title": " "})[0] == 400


# --- «Actualizar»: reindexing from the page -----------------------------------------------------


def test_reindex_from_the_page(server: tuple[str, int], engine: Engine) -> None:
    library = engine.config.paths.library
    assert library is not None
    (library / "est" / "nueva.md").write_text(
        "# Nueva\n\n## Idea\n\n" + "Una nota nueva sobre el teorema central del límite. " * 3,
        encoding="utf-8",
    )
    status, _, raw = request(server, "POST", "/api/reindex", {})
    events = _events(raw)
    names = [n for n, _ in events]
    assert status == 200 and names[0] == "start" and names[-1] == "done"
    assert events[0][1] == {"files": 3, "device": "cpu", "full": False}
    done = events[-1][1]
    assert (done["added"], done["unchanged"], done["documents"]) == (1, 2, 3)
    assert json.loads(request(server, "GET", "/api/status")[2])["documents"] == 3
    query = {"query": "teorema central del límite", "precision": "fast"}  # the fake reranker
    _, _, body = request(server, "POST", "/api/search", query)  # always prefers «bayes»
    assert json.loads(body)["hits"][0]["doc_path"] == "est/nueva.md"  # searchable at once


def test_reindex_refuses_while_another_index_runs(server: tuple[str, int], engine: Engine) -> None:
    import os

    lock = engine.config.paths.index_db.with_name(engine.config.paths.index_db.name + ".lock")
    lock.write_text(str(os.getpid()), encoding="utf-8")  # a live process holds the index
    try:
        status, _, body = request(server, "POST", "/api/reindex", {})
        assert status == 409 and "actualizando" in json.loads(body)["error"]
    finally:
        lock.unlink()


def test_reindex_does_not_need_the_llm(
    server: tuple[str, int], engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(engine, "llm_running", lambda: False)
    status, _, raw = request(server, "POST", "/api/reindex", {})
    assert status == 200 and _events(raw)[-1][1]["unchanged"] == 2


def test_page_scripts_do_not_redefine_each_others_globals() -> None:
    # classic scripts share one global scope: a later file's `function x` silently replaces an
    # earlier one (reader.js once replaced app.js's inlineNodes, mapas.js guias.js's renderOutline)
    import collections

    static = Path(__file__).parents[1] / "src" / "biblio" / "web" / "static"
    owners: dict[str, list[str]] = collections.defaultdict(list)
    for name in ASSETS:
        if not name.endswith(".js"):
            continue
        text = (static / name).read_text(encoding="utf-8")
        for found in re.findall(r"^(?:async )?function (\w+)|^(?:const|let) (\w+)", text, re.M):
            owners[found[0] or found[1]].append(name)
    clashes = {k: v for k, v in owners.items() if len(v) > 1}
    assert clashes == {}


# --- where the library is (ADR-039) -------------------------------------------------------------


def test_a_moved_library_reuses_the_index(
    server: tuple[str, int], engine: Engine, tmp_path: Path
) -> None:
    import shutil

    old = engine.config.paths.library
    assert old is not None
    moved = tmp_path / "moved vault"
    shutil.copytree(old, moved)
    status, _, raw = request(server, "POST", "/api/library/check", {"path": str(moved)})
    check = json.loads(raw)
    assert status == 200 and check["plan"] == "relocate" and check["matching"] == check["notes"]
    status, _, raw = request(server, "POST", "/api/library/set", {"path": str(moved)})
    events = _events(raw)
    names = [n for n, _ in events]
    assert names[0] == "check" and names[-1] == "library" and "done" in names
    done = dict(events)["done"]
    assert done["unchanged"] == check["notes"] and done["added"] == 0  # nothing re-embedded
    assert engine.config.paths.library == moved.resolve()
    assert json.loads(request(server, "GET", "/api/library")[2])["path"] == str(moved.resolve())
    local = (engine.config.root / "config" / "local.toml").read_text(encoding="utf-8")
    assert moved.resolve().as_posix() in local
    query = {"query": "teorema de Bayes", "precision": "fast"}
    assert json.loads(request(server, "POST", "/api/search", query)[2])["hits"]  # still searchable


def test_another_library_gets_its_own_index(
    server: tuple[str, int], engine: Engine, tmp_path: Path
) -> None:
    other = tmp_path / "other"
    (other / "bio").mkdir(parents=True)
    (other / "bio" / "celula.md").write_text(
        "# Célula\n\n## Mitocondria\n\n" + "La mitocondria produce energía para la célula. " * 3,
        encoding="utf-8",
    )
    check = json.loads(request(server, "POST", "/api/library/check", {"path": str(other)})[2])
    assert check["plan"] == "rebuild" and check["notes"] == 1 and check["matching"] == 0
    events = _events(request(server, "POST", "/api/library/set", {"path": str(other)})[2])
    assert dict(events)["start"]["full"] is True and dict(events)["done"]["documents"] == 1
    assert json.loads(request(server, "GET", "/api/status")[2])["documents"] == 1  # swapped
    hits = json.loads(
        request(server, "POST", "/api/search", {"query": "mitocondria", "precision": "fast"})[2]
    )["hits"]
    assert hits[0]["doc_path"] == "bio/celula.md"
    assert not engine.config.paths.index_db.with_name("index.db.tmp").exists()

    empty = tmp_path / "empty"
    empty.mkdir()
    status, _, raw = request(server, "POST", "/api/library/set", {"path": str(empty)})
    assert status == 400 and "Markdown" in json.loads(raw)["error"]
    assert engine.config.paths.library == other.resolve()  # unchanged


def test_folders_can_be_browsed(server: tuple[str, int], tmp_path: Path) -> None:
    from urllib.parse import quote

    (tmp_path / "Notas").mkdir()
    (tmp_path / ".oculta").mkdir()
    status, _, raw = request(server, "GET", f"/api/library/browse?path={quote(str(tmp_path))}")
    listing = json.loads(raw)
    names = [f["name"] for f in listing["folders"]]
    assert status == 200 and "Notas" in names and ".oculta" not in names
    assert listing["parent"] == str(tmp_path.resolve().parent)
    top = json.loads(request(server, "GET", "/api/library/browse")[2])
    assert top["path"] == "" and top["folders"]  # the drives
    missing = quote(str(tmp_path / "no existe"))
    assert request(server, "GET", f"/api/library/browse?path={missing}")[0] == 400


# --- «Reset» (ADR-040) ---------------------------------------------------------------------------


def test_reset_moves_the_app_memory_and_keeps_the_library(
    server: tuple[str, int], engine: Engine
) -> None:
    cfg = engine.config
    library = cfg.paths.library
    assert library is not None and cfg.conversations.db.is_relative_to(cfg.root)  # a test root
    notes = sorted(p.relative_to(library) for p in library.rglob("*.md"))
    request(server, "POST", "/api/search", {"query": "teorema de Bayes", "precision": "fast"})
    cfg.paths.drafts_dir.mkdir(parents=True, exist_ok=True)
    (cfg.paths.drafts_dir / "guia.md").write_text("# Guía\n", encoding="utf-8")
    items = {
        i["key"]: i["count"] for i in json.loads(request(server, "GET", "/api/reset")[2])["items"]
    }
    assert items["conversations"] == 1 and items["drafts"] == 1

    assert request(server, "POST", "/api/reset", {})[0] == 400  # the page's «¿Estás seguro?»
    status, _, raw = request(server, "POST", "/api/reset", {"confirm": True})
    result = json.loads(raw)
    backup = Path(result["backup"])
    assert (
        status == 200 and (backup / "conversations.db").is_file() and (backup / "drafts").is_dir()
    )
    assert not cfg.paths.drafts_dir.exists()
    assert json.loads(request(server, "GET", "/api/conversations")[2])["conversations"] == []
    request(server, "POST", "/api/search", {"query": "Bayes", "precision": "fast"})  # stores reopen
    assert len(json.loads(request(server, "GET", "/api/conversations")[2])["conversations"]) == 1
    assert sorted(p.relative_to(library) for p in library.rglob("*.md")) == notes  # untouched
    assert cfg.paths.index_db.is_file()
    assert json.loads(request(server, "GET", "/api/status")[2])["documents"] == len(notes)


def test_reset_never_touches_the_library(config: Config, tmp_path: Path) -> None:
    from biblio.reset import check_safe

    library = tmp_path / "vault"
    library.mkdir()
    cfg = replace(config, paths=replace(config.paths, library=library))
    with pytest.raises(ValueError):
        check_safe(cfg, [library / "drafts"])  # drafts configured inside the library
    with pytest.raises(ValueError):
        check_safe(cfg, [tmp_path])  # a folder that contains the library
    with pytest.raises(ValueError):
        check_safe(cfg, [cfg.paths.index_db])
    check_safe(cfg, [cfg.conversations.db, cfg.paths.drafts_dir])


def test_the_vendored_fonts_are_served_and_nothing_else(server: tuple[str, int]) -> None:
    import re

    css = (Path(__file__).parents[1] / "src/biblio/web/static/app.css").read_text(encoding="utf-8")
    fonts = re.findall(r"url\(/static/vendor/fonts/([^)]+)\)", css)
    assert len(fonts) == 8  # Fraunces 300/400 normal and italic, Plus Jakarta Sans 400-700
    for name in fonts:
        status, headers, _ = request(server, "GET", f"/static/vendor/fonts/{name}")
        assert status == 200 and headers.get("content-type") == "font/woff2", name
    for bad in ("LICENSE-Fraunces.txt", "../app.css", "..%2Fapp.css", "x/y.woff2", "nope.woff2"):
        assert request(server, "GET", f"/static/vendor/fonts/{bad}")[0] == 404, bad


def test_the_llm_is_switched_off_and_on_from_this_pc_only(
    server: tuple[str, int], engine: Engine
) -> None:
    """ADR-043: «Ajustes › Modelo de lenguaje». The page learns what it can do from /api/status."""
    calls: list[str] = []

    class FakeSwitch:
        owned = True

        def can_start(self) -> bool:
            return True

        def stop(self) -> str:
            calls.append("stop")
            return "stopped"

        def start(self) -> None:
            calls.append("start")

    status = json.loads(request(server, "GET", "/api/status")[2])
    assert status["llm_control"]["available"] is False
    assert request(server, "POST", "/api/llm/stop", {})[0] == 503  # nothing to control
    engine.llm_switch = FakeSwitch()  # type: ignore[assignment]
    code, _, raw = request(server, "POST", "/api/llm/stop", {})
    assert code == 200 and json.loads(raw)["result"] == "stopped" and calls == ["stop"]
    assert json.loads(request(server, "GET", "/api/status")[2])["llm_control"]["owned"] is True
    engine._generating.acquire()  # something is being generated: refused
    try:
        assert request(server, "POST", "/api/llm/stop", {})[0] == 409
    finally:
        engine._generating.release()
    assert calls == ["stop"]


def test_search_grouped_by_note_and_inside_one_note(server: tuple[str, int]) -> None:
    """ADR-045: «Solo ubicar» asks for one line per note; the reader searches inside one note."""
    query = {"query": "teorema de Bayes", "precision": "fast", "k": 2}
    data = json.loads(request(server, "POST", "/api/search", {**query, "group": True})[2])
    groups = [h["group"] for h in data["hits"]]
    assert groups == sorted(groups) and set(groups) <= {0, 1}  # at most k notes, in order
    for number in set(groups):
        paths = {h["doc_path"] for h in data["hits"] if h["group"] == number}
        assert len(paths) == 1  # one note per group
    flat = json.loads(request(server, "POST", "/api/search", query)[2])
    assert "group" not in flat["hits"][0] and len(flat["hits"]) <= 2

    before = len(json.loads(request(server, "GET", "/api/conversations")[2])["conversations"])
    note = data["hits"][0]["doc_path"]
    code, _, raw = request(server, "POST", "/api/notes/search", {"path": note, "query": "Bayes"})
    inside = json.loads(raw)
    assert code == 200 and inside["hits"] and {h["doc_path"] for h in inside["hits"]} == {note}
    after = len(json.loads(request(server, "GET", "/api/conversations")[2])["conversations"])
    assert after == before  # a search inside a note is not kept in the history
    bad = {"path": "no/existe.md", "query": "Bayes"}
    assert request(server, "POST", "/api/notes/search", bad)[0] == 400


# --- the interface in English (ADR-046) ---------------------------------------------------------

_STATIC_DIR = Path(__file__).parents[1] / "src" / "biblio" / "web" / "static"
_NOT_TRANSLATED = {"Español", "English", "OPB", "PB", "O"}  # each language names itself; the brand


def _english() -> dict[str, str]:
    text = (_STATIC_DIR / "i18n-en.js").read_text(encoding="utf-8")
    body = text[text.index("const I18N_EN = ") + len("const I18N_EN = ") : text.rindex(";")]
    data = json.loads(body)
    assert isinstance(data, dict)
    return data


def _html_texts() -> list[str]:
    from html.parser import HTMLParser

    class Texts(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.found: list[str] = []
            self.skip = 0

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            if tag in ("script", "style", "svg", "code", "pre"):
                self.skip += 1
            for name, value in attrs:
                if name in ("placeholder", "title", "aria-label") and value:
                    self.found.append(" ".join(value.split()))

        def handle_endtag(self, tag: str) -> None:
            if tag in ("script", "style", "svg", "code", "pre"):
                self.skip -= 1

        def handle_data(self, data: str) -> None:
            text = " ".join(data.split())
            if not self.skip and text:
                self.found.append(text)

    parser = Texts()
    parser.feed((_STATIC_DIR / "index.html").read_text(encoding="utf-8"))
    letters = re.compile(r"[A-Za-zÁÉÍÓÚáéíóúñÑ]")
    return [t for t in parser.found if letters.search(t) and t not in _NOT_TRANSLATED]


def test_every_interface_text_has_an_english_entry() -> None:
    english = _english()
    missing = [t for t in _html_texts() if t not in english]
    for name in ASSETS:
        if name.endswith(".js") and not name.startswith("i18n"):
            src = (_STATIC_DIR / name).read_text(encoding="utf-8")
            for raw in re.findall(r'\btx\("((?:[^"\\]|\\.)*)"', src):
                key = json.loads(f'"{raw}"')
                if key not in english:
                    missing.append(f"{name}: {key}")
    assert missing == []
    for spanish, text in english.items():  # the same placeholders on both sides
        assert sorted(re.findall(r"\{\d+\}", spanish)) == sorted(re.findall(r"\{\d+\}", text)), (
            spanish
        )


def test_page_scripts_keep_no_spanish_text_outside_tx() -> None:
    """A Spanish literal with accents or ¿¡ outside tx() would stay Spanish in English mode."""
    allowed = {"## Fuentes"}  # draft content the page parses, not interface text
    found = []
    for name in ASSETS:
        if not name.endswith(".js") or name.startswith("i18n"):
            continue
        text_of = (_STATIC_DIR / name).read_text(encoding="utf-8")
        src = re.sub(r"//[^\n]*|/\*.*?\*/", "", text_of, flags=re.S)
        for match in re.finditer(r'"((?:[^"\\\n]|\\.)*)"', src):
            text = match.group(1)
            wrapped = src[max(0, match.start() - 3) : match.start()].endswith("tx(")
            if re.search(r"[áéíóúñÁÉÍÓÚÑ¿¡]", text) and text not in allowed and not wrapped:
                found.append(f"{name}: {text}")
    assert found == []


def test_the_english_files_are_served(server: tuple[str, int]) -> None:
    for path in ("/static/i18n.js", "/static/i18n-en.js"):
        status, headers, _ = request(server, "GET", path)
        assert status == 200 and "javascript" in headers.get("content-type", "")
