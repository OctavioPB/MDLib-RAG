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
    for path in (
        "/static/app.js",
        "/static/app.css",
        "/static/guias.js",
        "/static/practicar.js",
        "/static/theme.js",
    ):
        assert request(server, "GET", path)[0] == 200
    assert request(server, "GET", "/static/../server.py")[0] == 404


def test_static_assets_reference_no_external_url() -> None:
    """ADR-002: the UI must work with the network off, so nothing may load from outside."""
    static = resources.files("biblio.web") / "static"
    for name in ("index.html", "app.js", "app.css", "guias.js", "practicar.js", "theme.js"):
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
