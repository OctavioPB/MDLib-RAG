from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from datetime import datetime, timedelta
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
    assert last[0] == {
        "t": "image",
        "alt": "figura",
        "src": "img.png",
    }  # drawn by the reader (ADR-048)
    web = next(n for n in last if n["t"] == "link" and "href" in n)
    note = next(n for n in last if n["t"] == "link" and "note" in n)
    assert web["href"] == "https://example.org" and note["note"] == "est/bayes.md"
    assert "<" not in json.dumps([b for b in blocks if b["type"] != "code"])


LONG_NOTE = """# 6. Principal Component Analysis

## Chapter 6. Principal Component Analysis

### Understanding Principal Component Analysis

Texto.

#### import matplotlib.pyplot as plt

#### x_train = x_train.reshape(-1, 1)

#### EBITDA = EBIT + D&A

#### `pandas.read_csv`

##### Demasiado hondo

### Filtering Noise
"""


def test_the_outline_leaves_out_headings_that_are_code() -> None:  # ADR-049
    tree = note_tree(LONG_NOTE, "doc/pca.md", lambda target: None, depth=3)
    assert [(h["level"], h["text"]) for h in tree["outline"]] == [
        (2, "Chapter 6. Principal Component Analysis"),
        (3, "Understanding Principal Component Analysis"),
        (4, "EBITDA = EBIT + D&A"),  # a formula, not code
        (4, "pandas.read_csv"),  # a code span the author made a title
        (3, "Filtering Noise"),
    ]  # the note's H1 is the page title; H5 is below the three levels
    assert tree["outline"][1]["line"] == 5  # where the reader finds the heading
    assert tree["words"] == len(LONG_NOTE.split())
    assert [
        h["level"] for h in note_tree(LONG_NOTE, "doc/pca.md", lambda t: None, 1)["outline"]
    ] == [2]
    # text before the title's H1 (a web clipping) does not make the title a level of the index
    preamble = "Texto previo.\n\n# T\n\n## A\n\n### B\n\n#### C\n"
    outline = note_tree(preamble, "doc/t.md", lambda t: None)["outline"]
    assert [h["text"] for h in outline] == ["A", "B", "C"]


PRACTICED_NOTE = (
    "# PCA\n\n## Componentes\n\n"
    + "Las componentes principales ordenan las direcciones de mayor varianza del conjunto. " * 2
    + "\n\n#### import numpy as np\n\n"
    + "El código calcula la matriz de covarianza y sus vectores propios para cada caso. " * 2
    + "\n\n## Ruido\n\n"
    + "Reducir y restaurar las dimensiones filtra el ruido que no sigue a las componentes. " * 2
    + "\n\n## Usos\n\n### Anomalías\n\n"  # a parent heading with no text of its own
    + "Una muestra anómala pierde más información al reducirla y restaurarla con el modelo. " * 2
    + "\n"
)


def test_the_index_shows_how_each_section_is_remembered(
    server: tuple[str, int],  # noqa: F811
    engine: Engine,  # noqa: F811
) -> None:  # ADR-050
    from biblio.index.indexer import index_library
    from biblio.practice.store import QuestionContent, Section

    from fakes import FakeEmbedder, WordCounter

    library = engine.config.paths.library
    assert library is not None
    (library / "est" / "pca.md").write_text(PRACTICED_NOTE, encoding="utf-8")
    index_library(
        engine.config,
        library,
        full=False,
        dtype="float32",
        embedder_factory=FakeEmbedder,
        counter_factory=WordCounter,
    )

    def answer(heading_path: tuple[str, ...], result: str) -> None:
        section = Section("est/pca.md", heading_path, 1, 2, "texto")
        quiz = engine.practice.new_quiz("guide", "est/pca.md", 1)
        q = engine.practice.add_question(
            quiz, "tf", section, QuestionContent("¿Sí?", "", truth=True)
        )
        engine.practice.record_attempt(q, "true", result, "code")

    status, _, raw = request(server, "GET", "/api/notes/mastery?path=est/pca.md")
    before = json.loads(raw)
    assert status == 200 and [e["state"] for e in before["entries"]] == ["new"] * 4
    # practice under the code heading belongs to the entry the index shows: «Componentes»
    answer(("PCA", "Componentes", "import numpy as np"), "correct")
    answer(("PCA", "Ruido"), "wrong")
    answer(("PCA", "Usos", "Anomalías"), "correct")  # its chunk starts at «## Usos»
    answer(("PCA", "Ya no existe"), "wrong")  # a section the note no longer has
    with engine.practice._conn:  # answered a month ago: due today
        engine.practice._conn.execute(
            "UPDATE attempts SET answered = ? WHERE question_id IN "
            "(SELECT id FROM questions WHERE heading_path NOT IN (?, ?))",
            (
                (datetime.now().astimezone() - timedelta(days=30)).isoformat(timespec="seconds"),
                json.dumps(["PCA", "Componentes", "import numpy as np"], ensure_ascii=False),
                json.dumps(["PCA", "Usos", "Anomalías"], ensure_ascii=False),
            ),
        )
    data = json.loads(request(server, "GET", "/api/notes/mastery?path=est/pca.md")[2])
    first, second, uses, anomalies = data["entries"]
    assert first["line"] == 3 and first["state"] != "new"
    assert (first["practiced"], first["sections"]) == (1, 2)
    assert second["practiced"] == 1 and second["sections"] == 1
    assert uses["state"] == "new" and anomalies["state"] != "new" and anomalies["practiced"] == 1
    assert data["due"] == [["est/pca.md", json.dumps(["PCA", "Ruido"], ensure_ascii=False)]]
    assert request(server, "GET", "/api/notes/mastery?path=nada.md")[0] == 404


def test_the_reader_serves_only_indexed_notes(server: tuple[str, int]) -> None:  # noqa: F811
    status, _, raw = request(server, "GET", "/api/notes/view?path=est/bayes.md")
    view = json.loads(raw)
    assert status == 200 and view["title"] == "Bayes" and not view["read"]
    assert view["words"] > 0 and view["words_per_minute"] > 0 and isinstance(view["outline"], list)
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
