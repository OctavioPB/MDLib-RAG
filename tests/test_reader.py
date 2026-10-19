from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from test_practice import _Teacher
from test_web import _events, engine, request, server  # noqa: F401  (fixtures used by name)

from biblio.generate.llm import Message
from biblio.reader import note_tree
from biblio.web.engine import Engine

NOTE = """---
title: Nota de prueba
---
# Nota de prueba

## Fórmulas

La media es $\\mu = \\frac{1}{n}\\sum x_i$ y se ve [[Bayes|el teorema]] y [[Nada]].

$$
\\sigma^2 = E[(X-\\mu)^2]
$$

| A | B |
|:--|--:|
| **1** | `x` |

```python
print("hola")
```

<div>crudo</div>

![figura](img.png) [web](https://example.org) [otra](../est/bayes.md)
"""


def _walk(blocks: Sequence[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for b in blocks:
        yield b
        yield from _walk(b.get("children", []))


def test_a_note_becomes_typed_blocks_with_their_source_lines() -> None:
    resolve = {"Bayes.md": "est/bayes.md", "est/bayes.md": "est/bayes.md"}.get
    tree = note_tree(NOTE, "doc/nota.md", resolve)
    assert tree["title"] == "Nota de prueba"
    blocks = tree["blocks"]
    assert [b["type"] for b in blocks] == [
        "heading",
        "heading",
        "p",
        "math",
        "table",
        "code",
        "code",
        "p",
    ]
    assert blocks[0]["lines"] == [4, 4]  # after the frontmatter: same numbers as the chunks
    para = blocks[2]["inlines"]
    math = next(n for n in para if n["t"] == "math")
    assert math["tex"] == "\\mu = \\frac{1}{n}\\sum x_i" and not math["display"]
    links = [n for n in para if n["t"] == "wikilink"]
    assert links == [
        {"t": "wikilink", "text": "el teorema", "note": "est/bayes.md"},
        {"t": "wikilink", "text": "Nada", "note": None},
    ]
    assert blocks[3]["tex"].strip() == "\\sigma^2 = E[(X-\\mu)^2]" and blocks[3]["lines"] == [
        10,
        12,
    ]
    cells = [b for b in _walk([blocks[4]]) if b["type"] in ("th", "td")]
    assert [c.get("align") for c in cells[:2]] == ["left", "right"]
    assert cells[2]["inlines"] == [{"t": "strong", "children": [{"t": "text", "text": "1"}]}]
    assert blocks[5] == {**blocks[5], "lang": "python", "text": 'print("hola")\n'}
    assert blocks[6]["lang"] == "html"  # raw HTML is shown as code, never rendered
    last = blocks[7]["inlines"]
    assert last[0] == {"t": "image", "alt": "figura"}
    web = next(n for n in last if n["t"] == "link" and "href" in n)
    note = next(n for n in last if n["t"] == "link" and "note" in n)
    assert web["href"] == "https://example.org" and note["note"] == "est/bayes.md"
    assert "<" not in json.dumps([b for b in blocks if b["type"] != "code"])


def test_the_reader_serves_only_indexed_notes(server: tuple[str, int]) -> None:  # noqa: F811
    status, _, raw = request(server, "GET", "/api/notes/view?path=est/bayes.md")
    view = json.loads(raw)
    assert status == 200 and view["title"] == "Bayes" and not view["read"]
    assert view["blocks"][1]["type"] == "heading" and view["blocks"][2]["lines"] == [5, 5]
    for bad in ("nada.md", "../est/bayes.md", "est/../est/bayes.md", ""):
        assert request(server, "GET", f"/api/notes/view?path={bad}")[0] == 404, bad


def test_katex_is_served_from_its_folder_only(server: tuple[str, int]) -> None:  # noqa: F811
    status, headers, body = request(server, "GET", "/static/vendor/katex/katex.min.js")
    assert status == 200 and headers["content-type"].startswith("text/javascript") and body
    status, headers, _ = request(
        server, "GET", "/static/vendor/katex/fonts/KaTeX_Main-Regular.woff2"
    )
    assert status == 200 and headers["content-type"] == "font/woff2"
    for bad in (
        "../app.js",
        "fonts/../../app.js",
        "VERSION.txt",
        "LICENSE",
        "fonts/x/y.woff2",
        "nada.js",
    ):
        assert request(server, "GET", f"/static/vendor/katex/{bad}")[0] == 404, bad


def test_vendored_katex_loads_nothing_from_outside() -> None:
    folder = Path(__file__).parents[1] / "src" / "biblio" / "web" / "static" / "vendor" / "katex"
    script = (folder / "katex.min.js").read_text(encoding="utf-8")
    # only the SVG and MathML namespace names, which are identifiers, never requested
    assert set(re.findall(r"https?://[^\"' )]+", script)) == {
        "http://www.w3.org/2000/svg",
        "http://www.w3.org/1998/Math/MathML",
    }
    assert not re.search(r"XMLHttpRequest|sendBeacon|WebSocket|importScripts|window\.fetch", script)
    css = (folder / "katex.min.css").read_text(encoding="utf-8")
    fonts = re.findall(r"url\(([^)]+)\)", css)
    assert fonts and all(f.startswith("fonts/") and f.endswith(".woff2") for f in fonts)
    assert all((folder / f).is_file() for f in fonts)
    assert not re.search(r"https?://|//cdn|@import", css)


class _Explainer(_Teacher):
    """Writes questions as the practice tests' teacher, and answers as the tutor."""

    def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
        if "tutor" in messages[0].content:
            yield "Invierte probabilidades condicionadas [1]."
        else:
            yield from super().stream(messages, max_tokens)


def test_explain_a_passage_cites_it_first_and_is_kept_in_the_history(
    server: tuple[str, int],  # noqa: F811
    engine: Engine,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine, "llm_running", lambda: True)
    monkeypatch.setattr(engine, "_llm_factory", lambda _c: _Explainer())
    body = {
        "path": "est/bayes.md",
        "lines": [5, 5],
        "text": "El teorema de Bayes invierte probabilidades",
    }
    _, _, raw = request(server, "POST", "/api/notes/explain", body)
    events = _events(raw)
    meta, done = events[0][1], next(d for n, d in events if n == "done")
    first = meta["fragments"][0]
    assert first["number"] == 1 and first["text"] == body["text"]
    assert first["citation"] == "est/bayes.md › Teorema" and first["lines"] == [5, 5]
    second = meta["fragments"][1]  # the section the passage sits in, whole
    assert second["citation"] == "est/bayes.md › Teorema"
    assert second["text"].count("invierte probabilidades") == 3
    others = meta["fragments"][2:]
    assert all(f["doc_path"] != "est/bayes.md" for f in others)  # nothing overlapping again
    assert done["cited"] == [1] and done["conversation_id"]
    stored = engine.conversations.get(done["conversation_id"])
    assert stored is not None
    assert stored[1][0].question.startswith("Explícame este pasaje de «est/bayes.md › Teorema»")

    too_short = {**body, "text": "Bayes"}
    assert request(server, "POST", "/api/notes/explain", too_short)[0] == 400
    assert request(server, "POST", "/api/notes/explain", {**body, "path": "nada.md"})[0] == 404


def test_questions_on_a_passage_are_practice_of_its_section(
    server: tuple[str, int],  # noqa: F811
    engine: Engine,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine, "llm_running", lambda: True)
    monkeypatch.setattr(engine, "_llm_factory", lambda _c: _Explainer())
    body = {
        "path": "est/bayes.md",
        "lines": [5, 5],
        "text": "El teorema de Bayes invierte probabilidades condicionadas",
        "kinds": ["mc", "tf", "open_self"],
    }
    status, _, raw = request(server, "POST", "/api/notes/quiz", body)
    events = _events(raw)
    assert status == 200 and [n for n, _ in events].count("question") == 3
    question = next(d for n, d in events if n == "question")
    stored = engine.practice.question(question["id"])
    assert stored is not None and stored.section.heading_path == ("Bayes", "Teorema")
    assert stored.section.text == body["text"]  # written from the passage, not the whole section
    assert request(server, "POST", "/api/notes/quiz", {**body, "kinds": ["open_model"]})[0] == 400


def test_the_library_lists_every_note_with_its_read_mark(
    server: tuple[str, int],  # noqa: F811
    engine: Engine,  # noqa: F811
) -> None:
    engine.practice.set_read("est/bayes.md", True)
    status, _, raw = request(server, "GET", "/api/notes")
    notes = json.loads(raw)["notes"]
    assert status == 200 and [n["path"] for n in notes] == ["alg/svd.md", "est/bayes.md"]
    assert {n["path"]: (n["title"], n["read"]) for n in notes} == {
        "alg/svd.md": ("SVD", False),
        "est/bayes.md": ("Bayes", True),
    }
