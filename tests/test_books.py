"""Books in the library and the reader's chapter context (ADR-053)."""

from __future__ import annotations

import json

from test_web import engine, request, server  # noqa: F401  (fixtures used by name)

from biblio.books import book_id, find_books
from biblio.index.indexer import index_library
from biblio.reader import note_tree
from biblio.web.engine import Engine

from fakes import FakeEmbedder, WordCounter

NOTES = [
    ("AI/Books/Applied ML/10-8-deep-learning.md", "8. Deep Learning"),
    ("AI/Books/Applied ML/02-preface.md", "Preface"),
    ("AI/Books/Applied ML/03-1-machine-learning.md", "1. Machine Learning"),
    ("AI/Books/Applied ML/Applied ML.md", "A"),  # the cover: a broken heading
    ("AI/Books/hackers-delight/01-intro.md", "Intro"),
    ("AI/Books/go-ml/go-ml.md", "Machine Learning with Go"),  # a slug folder with a cover
    ("AI/Books/go-ml/02-basics.md", "Basics"),
    ("Data/Books/causality.md", "Causality"),  # one note directly in Books/
    ("AI/agentes.md", "Agentes"),  # the user's own note
]


def test_a_book_is_a_folder_in_books_with_its_chapters_in_natural_order() -> None:
    books = {b.id: b for b in find_books(NOTES, "Books")}
    assert set(books) == {
        "AI/Books/Applied ML",
        "AI/Books/hackers-delight",
        "AI/Books/go-ml",
        "Data/Books/causality.md",
    }
    ml = books["AI/Books/Applied ML"]
    assert ml.title == "Applied ML" and ml.area == "AI"  # the folder, not the cover's «A»
    assert ml.cover == "AI/Books/Applied ML/Applied ML.md"
    assert [c.title for c in ml.chapters] == ["Preface", "1. Machine Learning", "8. Deep Learning"]
    assert books["AI/Books/go-ml"].title == "Machine Learning with Go"  # a slug: the cover's title
    assert books["AI/Books/hackers-delight"].title == "Hackers delight"  # a slug, no cover
    assert books["Data/Books/causality.md"].title == "Causality"
    assert book_id("AI/agentes.md", "Books") is None


def test_a_books_boxes_stay_out_of_the_index_and_are_counted() -> None:
    text = (
        "# Cap\n\n## R-CNNs\n\nTexto.\n\n### NOTE\n\nUn recuadro.\n\n"
        "### Tip\n\nOtro.\n\n## YOLO\n\nMás.\n"
    )
    tree = note_tree(text, "c.md", lambda t: None, callouts=("note", "tip"))
    assert [h["text"] for h in tree["outline"]] == ["R-CNNs", "YOLO"]
    assert tree["callouts"] == 2


def test_the_shelf_a_book_and_a_chapter_in_the_reader(
    server: tuple[str, int],  # noqa: F811
    engine: Engine,  # noqa: F811
) -> None:
    library = engine.config.paths.library
    assert library is not None
    book = library / "AI" / "Books" / "Applied ML"
    book.mkdir(parents=True)
    body = (
        "Machine learning lets a model learn patterns from data instead of rules written by hand. "
        * 3
    )
    for name, title in (
        ("02-preface.md", "Preface"),
        ("03-1-machine-learning.md", "1. Machine Learning"),
        ("10-8-deep-learning.md", "8. Deep Learning"),
    ):
        (book / name).write_text(f"# {title}\n\n## Section\n\n{body}\n", encoding="utf-8")
    index_library(
        engine.config,
        library,
        full=False,
        dtype="float32",
        embedder_factory=FakeEmbedder,
        counter_factory=WordCounter,
    )
    engine.practice.set_read("AI/Books/Applied ML/02-preface.md", True)

    shelf = json.loads(request(server, "GET", "/api/books")[2])
    found = next(b for b in shelf["books"] if b["id"] == "AI/Books/Applied ML")
    assert (found["chapters"], found["read"], found["english"]) == (3, 1, True)
    assert found["next"]["title"] == "1. Machine Learning"  # the first one not read
    assert {"name": "AI", "count": 1} in shelf["areas"]

    status, _, raw = request(server, "GET", "/api/books/one?id=AI%2FBooks%2FApplied%20ML")
    one = json.loads(raw)
    assert status == 200 and [c["read"] for c in one["chapters"]] == [True, False, False]
    assert request(server, "GET", "/api/books/one?id=nada")[0] == 404

    notes = json.loads(request(server, "GET", "/api/notes")[2])["notes"]
    assert {n["path"]: n["book"] for n in notes}["est/bayes.md"] is None  # the user's own note

    path = "AI/Books/Applied ML/03-1-machine-learning.md"
    view = json.loads(request(server, "GET", f"/api/notes/view?path={path.replace(' ', '%20')}")[2])
    ctx = view["book"]
    assert (ctx["title"], ctx["position"], ctx["total"]) == ("Applied ML", 2, 3)
    assert ctx["prev"]["title"] == "Preface" and ctx["next"]["title"] == "8. Deep Learning"
    plain = json.loads(request(server, "GET", "/api/notes/view?path=est/bayes.md")[2])
    assert plain["book"] is None
