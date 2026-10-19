from __future__ import annotations

import http.client
import json
import threading
from collections.abc import Iterator, Sequence

import pytest
from test_practice import _call, cfg  # noqa: F401  (the fixture is used by name)

from biblio.config import Config
from biblio.generate.context import Fragment
from biblio.generate.llm import Message
from biblio.practice.dictionary import Entry, merge_entries, parse_entries, to_anki, to_markdown
from biblio.retrieve.rerank import RerankPlan
from biblio.retrieve.store import open_index
from biblio.web.engine import Engine
from biblio.web.server import make_server

from fakes import FakeEmbedder, WordCounter

CODE = """La regresión logística estima probabilidades con la función sigmoide.

```python
def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))
```

La pérdida es $L = -\\sum_i y_i \\log p_i$ sobre todos los ejemplos."""
FRAGMENT = Fragment(1, "ml/logit.md", ("Logit", "Modelo"), CODE, 10, 20)


def test_entries_are_checked_against_the_fragment() -> None:
    reply = json.dumps(
        {
            "conceptos": [
                {
                    "termino": "Regresión logística",
                    "definicion": "Un modelo que estima probabilidades.",
                    "cita": "estima probabilidades con la función sigmoide",
                },
                {"termino": "SVM", "definicion": "Otro modelo.", "cita": "estima probabilidades"},
                {
                    "termino": "sigmoide",
                    "definicion": "Una curva.",
                    "cita": "una cita inventada aquí",
                },
            ],
            "funciones": [
                {
                    "nombre": "sigmoid",
                    "firma": "def sigmoid(z: float) ->  float:",
                    "descripcion": "La curva.",
                },
                {
                    "nombre": "softmax",
                    "firma": "def sigmoid(z: float) -> float:",
                    "descripcion": "No.",
                },
                {"nombre": "relu", "firma": "def relu(x):", "descripcion": "No está en el código."},
            ],
            "formulas": [
                {
                    "formula": "$L = -\\sum_i y_i\\log p_i$",
                    "nombre": "Pérdida",
                    "descripcion": "Log loss.",
                },
                {"formula": "E = mc^2", "nombre": "Energía", "descripcion": "No está."},
            ],
        }
    )
    entries = parse_entries(f"Aquí: {reply}", FRAGMENT, per_list=5, definition_max=400)
    assert entries is not None
    assert [(e.kind, e.term) for e in entries] == [
        ("concept", "Regresión logística"),  # term absent (SVM) or quote invented: dropped
        ("function", "sigmoid"),  # signature found ignoring spaces; a name not in it: dropped
        ("formula", "Pérdida"),  # LaTeX found ignoring spaces and the $ signs
    ]
    assert entries[2].detail == "L = -\\sum_i y_i\\log p_i" and entries[0].sources == (1,)
    assert parse_entries("no JSON", FRAGMENT, 5, 400) is None
    capped = parse_entries(reply, FRAGMENT, per_list=1, definition_max=400)
    assert capped is not None and len(capped) == 3


def test_entries_found_again_are_merged_with_their_sources() -> None:
    a = Entry("concept", "Sigmoide", "Curva en S.", "q", (1,))
    b = Entry("concept", "sigmoide", "Otra redacción.", "q2", (3,))
    f1 = Entry("formula", "Pérdida", "x", "L = - y", (1,))
    f2 = Entry("formula", "Log loss", "y", "L=-y", (2,))  # the same formula, other spacing
    merged = merge_entries([a, f1], [b, f2, Entry("concept", "Nueva", "n", "q", (4,))], limit=3)
    assert merged == [
        Entry("concept", "Sigmoide", "Curva en S.", "q", (1, 3)),
        Entry("formula", "Pérdida", "x", "L = - y", (1, 2)),
        Entry("concept", "Nueva", "n", "q", (4,)),
    ]
    # the limit is per list: two concepts already, one more fits; a third formula would not
    more = merge_entries(merged, [Entry("concept", "Otra", "o", "q", (5,))], limit=2)
    assert len(more) == 3 and [e.term for e in more][-1] == "Nueva"  # concepts full at 2
    fns = [Entry("function", f"f{i}", "x", f"def f{i}():", (6,)) for i in range(3)]
    assert sum(e.kind == "function" for e in merge_entries(more, fns, limit=2)) == 2


def test_exports() -> None:
    entries = [
        Entry(
            "concept",
            "Regresión logística",
            "Estima probabilidades.",
            "estima probabilidades",
            (1,),
        ),
        Entry("function", "sigmoid", "La curva.", "def sigmoid(z: float) -> float:", (1,)),
        Entry("formula", "Pérdida", "Log loss.", "L = -\\sum_i y_i", (1,)),
    ]
    md = to_markdown("Logit", entries, [FRAGMENT], "2026-10-19")
    assert md.startswith("# Diccionario: Logit")
    assert "## Conceptos" in md and "## Funciones" in md and "## Fórmulas" in md
    assert "$$L = -\\sum_i y_i$$" in md and "1. ml/logit.md › Modelo (líneas 10-20)" in md
    anki = to_anki("Logit", entries, [FRAGMENT]).splitlines()
    assert anki[:2] == ["#separator:tab", "#html:true"] and anki[2] == "#tags:biblioteca Logit"
    assert all(line.count("\t") == 1 for line in anki[3:]) and len(anki) == 6
    assert "\\(L = -\\sum_i y_i\\)" in anki[5]


# --- through the API -----------------------------------------------------------------------------


class _Lexicographer:
    def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
        user = messages[1].content
        concepts = []
        if "teorema de Bayes" in user:
            concepts.append(
                {
                    "termino": "teorema de Bayes",
                    "definicion": "Una regla para invertir probabilidades.",
                    "cita": "invierte probabilidades condicionadas usando la marginal",
                }
            )
        yield json.dumps({"conceptos": concepts, "funciones": [], "formulas": []})

    def close(self) -> None:
        pass


@pytest.fixture
def server(cfg: Config) -> Iterator[tuple[str, int, Engine]]:  # noqa: F811
    engine = Engine(
        cfg,
        open_index(cfg, shared_across_threads=True),
        FakeEmbedder(),
        plan_resolver=lambda _c: RerankPlan(None, 0, "test"),
        llm_factory=lambda _c: _Lexicographer(),
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


def test_build_keep_review_and_export(server: tuple[str, int, Engine]) -> None:
    engine = server[2]
    status, events = _call(
        server, "POST", "/api/dictionary", {"kind": "guide", "value": "est/bayes.md"}
    )
    names = [n for n, _ in events]
    assert status == 200 and names[0] == "start" and names[-1] == "done"
    assert names.count("entries") == len(events[0][1]["fragments"])
    entries = [d for n, d in events if n == "entries"][-1]["entries"]
    assert [e["term"] for e in entries] == ["teorema de Bayes"]
    assert len(entries[0]["sources"]) >= 1  # found in more than one section, merged
    did = events[-1][1]["id"]

    assert _call(server, "POST", "/api/dictionary", {"kind": "guide", "value": "nada.md"})[0] == 400
    assert _call(server, "POST", "/api/dictionary", {"kind": "otro", "value": "x"})[0] == 400

    _, listed = _call(server, "GET", "/api/dictionary/history?q=bayes")
    assert [d["id"] for d in listed["dictionaries"]] == [did]
    assert listed["dictionaries"][0]["counts"] == {"concept": 1, "function": 0, "formula": 0}
    _, view = _call(server, "GET", f"/api/dictionary/{did}")
    assert view["title"] == "Bayes" and view["fragments"][0]["citation"].startswith("est/bayes.md")

    # a flashcard answered counts as practice of the entry's section
    status, first = _call(
        server, "POST", "/api/dictionary/review", {"id": did, "index": 0, "result": "partial"}
    )
    assert status == 200 and first["quiz_id"]
    body = {"id": did, "index": 0, "result": "correct", "quiz_id": first["quiz_id"]}
    assert _call(server, "POST", "/api/dictionary/review", body)[1]["quiz_id"] == first["quiz_id"]
    histories = engine.practice.section_histories()
    (key,) = [k for k in histories if k[0] == "est/bayes.md"]
    assert [r for _, r in histories[key]] == ["partial", "correct"]
    assert _call(server, "GET", f"/api/dictionary/{did}")[1]["marks"] == {"0": "correct"}
    assert _call(server, "POST", "/api/dictionary/review", {**body, "index": 9})[0] == 404

    conn = http.client.HTTPConnection(server[0], server[1], timeout=10)
    for fmt, kind, start in (
        ("md", "text/markdown", "# Diccionario: Bayes"),
        ("anki", "text/plain", "#separator:tab"),
    ):
        conn.request("GET", f"/api/dictionary/{did}/export?format={fmt}")
        resp = conn.getresponse()
        text = resp.read().decode("utf-8")
        assert resp.status == 200 and resp.getheader("Content-Type", "").startswith(kind)
        assert resp.getheader("Content-Disposition", "").startswith('attachment; filename="bayes')
        assert text.startswith(start)
    conn.close()

    assert _call(server, "POST", "/api/dictionary/delete", {"id": did})[0] == 200
    assert _call(server, "GET", f"/api/dictionary/{did}")[0] == 404
