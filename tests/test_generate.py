from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Sequence
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, ClassVar

import pytest

from biblio.config import Config
from biblio.generate.ask import Tutor, load_prompt
from biblio.generate.citations import extract_citations, is_abstention, validate
from biblio.generate.context import assemble, merge_contiguous
from biblio.generate.llm import LLMError, Message, OpenAICompatibleClient
from biblio.offline import NetworkPolicyError
from biblio.retrieve.types import Hit

from fakes import WordCounter


def hit(rank: int, path: str, start: int, end: int, text: str = "uno dos tres") -> Hit:
    return Hit(rank, path, ("Guía", f"S{rank}"), text, start, end, rank, 1.0 / rank, "test")


# --- context ------------------------------------------------------------------------------------


def test_contiguous_hits_of_a_document_are_joined() -> None:
    merged = merge_contiguous(
        [hit(1, "a.md", 10, 20, "A"), hit(2, "b.md", 1, 5), hit(3, "a.md", 22, 30, "B")]
    )
    assert [(h.doc_path, h.start_line, h.end_line) for h in merged] == [
        ("a.md", 10, 30),
        ("b.md", 1, 5),
    ]
    assert merged[0].text == "A\n\nB"


def test_budget_drops_whole_fragments_and_numbers_from_one() -> None:
    long = " ".join(["palabra"] * 50)
    hits = [hit(1, "a.md", 1, 5), hit(2, "b.md", 1, 5, long), hit(3, "c.md", 1, 5)]
    context = assemble(hits, WordCounter(), budget_tokens=40, max_fragments=6, fixed_tokens=10)
    assert [f.doc_path for f in context.fragments] == ["a.md", "c.md"]  # b.md never cut in half
    assert [f.number for f in context.fragments] == [1, 2]
    assert context.dropped == 1
    assert context.tokens <= 40
    assert context.fragments[0].render().startswith("[1] a.md › S1 (líneas 1-5)\n")


def test_fragment_limit() -> None:
    hits = [hit(i, f"{i}.md", 1, 2) for i in range(1, 10)]
    context = assemble(hits, WordCounter(), budget_tokens=10_000, max_fragments=4, fixed_tokens=0)
    assert len(context.fragments) == 4


# --- citations ----------------------------------------------------------------------------------


def test_citation_formats() -> None:
    assert extract_citations("A [2]. B [1][3]. C [1, 4]; D [5;6]. [2]") == [2, 1, 3, 4, 5, 6]


def test_validation_flags() -> None:
    abstention = load_prompt("abstention")
    good = validate("El DW es central [1] y el mart departamental [2].", 3, abstention)
    assert good.grounded and good.cited == (1, 2) and not good.invalid
    invented = validate("Según [7], sí.", 3, abstention)
    assert invented.invalid == (7,) and not invented.grounded
    uncited = validate("Es así porque sí.", 3, abstention)
    assert not uncited.grounded and not uncited.abstained
    refused = validate("No encuentro esto en tu biblioteca.", 3, abstention)
    assert refused.abstained and not refused.grounded


def test_abstention_tolerates_punctuation_but_not_extra_content() -> None:
    abstention = "No encuentro esto en tu biblioteca."
    assert is_abstention("«No encuentro esto en tu biblioteca»", abstention)
    assert not is_abstention(
        "No encuentro esto en tu biblioteca, pero en general Kubernetes se instala con kubeadm.",
        abstention,
    )


# --- LLM client against a fake OpenAI-compatible server on loopback ----------------------------


class _Handler(BaseHTTPRequestHandler):
    received: ClassVar[dict[str, Any]] = {}

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        _Handler.received = json.loads(self.rfile.read(length))
        if _Handler.received["model"] == "missing":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error": "model not found"}')
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for piece in ["Hola ", "[1]", "."]:
            chunk = {"choices": [{"delta": {"content": piece}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def fake_server() -> Iterator[str]:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def test_client_streams_and_sends_the_configured_model(config: Config, fake_server: str) -> None:
    client = OpenAICompatibleClient(replace(config.llm, base_url=fake_server))
    tokens = list(client.stream([Message("user", "hola")], max_tokens=16))
    assert "".join(tokens) == "Hola [1]."
    assert _Handler.received["model"] == config.llm.model
    assert _Handler.received["max_tokens"] == 16 and _Handler.received["stream"] is True
    client.close()


def test_client_errors(config: Config, fake_server: str) -> None:
    missing = OpenAICompatibleClient(replace(config.llm, base_url=fake_server, model="missing"))
    with pytest.raises(LLMError, match="404"):
        list(missing.stream([Message("user", "x")], 8))
    with pytest.raises(NetworkPolicyError):
        OpenAICompatibleClient(replace(config.llm, base_url="https://api.example.com/v1"))
    closed = OpenAICompatibleClient(replace(config.llm, base_url="http://127.0.0.1:9/v1"))
    with pytest.raises(LLMError, match="serve-llm"):
        list(closed.stream([Message("user", "x")], 8))


# --- Tutor --------------------------------------------------------------------------------------


class _Retriever:
    name = "fake"

    def __init__(self, hits: list[Hit]) -> None:
        self._hits = hits

    def search(self, query: str, k: int) -> list[Hit]:
        return self._hits[:k]

    def close(self) -> None:
        pass


class _LLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.messages: Sequence[Message] = ()

    def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
        self.messages = messages
        yield from self.reply.split(" ")

    def close(self) -> None:
        pass


def test_tutor_builds_a_numbered_prompt_and_validates(config: Config) -> None:
    llm = _LLM("El DW es central [1] y el mart, departamental [2].")
    hits = [hit(1, "a.md", 1, 5, "Data Warehouse central"), hit(2, "b.md", 1, 5, "Data Mart")]
    tutor = Tutor(config, _Retriever(hits), llm, WordCounter())
    answer = tutor.ask("¿DW vs mart?")
    system, user = llm.messages
    assert "No encuentro esto en tu biblioteca." in system.content
    assert "[1] a.md › S1 (líneas 1-5)\nData Warehouse central" in user.content
    assert user.content.rstrip().endswith("citando cada afirmación con [n].")
    assert answer.validation.grounded and answer.validation.cited == (1, 2)


def test_tutor_abstains_without_calling_the_llm_when_nothing_is_retrieved(config: Config) -> None:
    llm = _LLM("should not be used")
    answer = Tutor(config, _Retriever([]), llm, WordCounter()).ask("¿algo?")
    assert answer.validation.abstained
    assert llm.messages == ()


def test_prompt_braces_in_evidence_are_safe(config: Config) -> None:
    latex = hit(1, "m.md", 1, 2, r"$$\frac{a}{b} = {x}$$")
    llm = _LLM("Es una fracción [1].")
    Tutor(config, _Retriever([latex]), llm, WordCounter()).ask("¿qué es {x}?")
    assert r"\frac{a}{b} = {x}" in llm.messages[1].content
    assert "¿qué es {x}?" in llm.messages[1].content


def test_placeholder_citations_are_not_grounded() -> None:
    abstention = load_prompt("abstention")
    v = validate("A es así [1]. B también [n]. C [X].", 3, abstention)
    assert v.cited == (1,) and v.placeholders == 2 and not v.grounded


def test_map_and_reduce_prompts_show_a_number_not_a_placeholder() -> None:
    for name in ("map_system", "reduce_system", "reduce_user"):
        assert "[n]" not in load_prompt(name)
