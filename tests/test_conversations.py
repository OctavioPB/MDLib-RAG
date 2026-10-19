from __future__ import annotations

from datetime import date
from pathlib import Path

from biblio.conversations import ConversationStore, NewTurn, fts_all_words, to_markdown


def _explain(question: str, answer: str) -> NewTurn:
    fragment = {"number": 1, "citation": "est/bayes.md › Teorema", "lines": [3, 9]}
    return NewTurn(
        question,
        "explain",
        None,
        "direct",
        "done",
        answer=answer,
        payload={"fragments": [fragment]},
    )


def test_messages_start_and_extend_conversations(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "c.db")
    cid, _ = store.add_turn(None, _explain("¿Qué es la regresión?" + " larga" * 40, "Es…"))
    same, _ = store.add_turn(
        cid, NewTurn("regresión", "find", "Math", "direct", "done", payload={"hits": []})
    )
    assert same == cid
    other, _ = store.add_turn(9999, _explain("Otra", "x"))  # unknown id: a new conversation
    assert other != cid
    summary, turns = store.get(cid)  # type: ignore[misc]
    assert summary.turns == 2 and len(summary.title) <= 120 and summary.title.endswith("…")
    assert [t.mode for t in turns] == ["explain", "find"] and turns[1].scope == "Math"
    assert turns[0].payload["fragments"][0]["number"] == 1
    assert [c.id for c in store.conversations(10)] == [other, cid]  # newest first


def test_search_needs_every_word_and_ignores_accents(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "c.db")
    bayes, _ = store.add_turn(
        None, _explain("¿Qué es el teorema de Bayes?", "Invierte la probabilidad condicional.")
    )
    store.add_turn(None, _explain("¿Qué es la regresión?", "Ajusta una recta."))
    found = store.conversations(10, "probabilidad CONDICIONAL")
    assert [c.id for c in found] == [bayes] and "«" in found[0].snippet
    assert store.conversations(10, "regresion recta")[0].title == "¿Qué es la regresión?"
    assert store.conversations(10, "bayes recta") == []  # both words must appear
    assert fts_all_words("¿?") is None and len(store.conversations(10, "¿?")) == 2


def test_rename_delete_and_export(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "c.db")
    cid, _ = store.add_turn(None, _explain("¿Qué es Bayes?", "Una regla [1]."))
    store.add_turn(cid, NewTurn("bayes", "explain", None, "direct", "error", error="sin modelo"))
    assert store.rename(cid, "  Bayes   repaso ") and store.get(cid)[0].title == "Bayes repaso"  # type: ignore[index]
    summary, turns = store.get(cid)  # type: ignore[misc]
    md = to_markdown(summary, turns, date(2026, 10, 19))
    assert md.startswith("# Bayes repaso\n") and "## ¿Qué es Bayes?" in md
    assert "Una regla [1]." in md and "1. est/bayes.md › Teorema (líneas 3-9)" in md
    assert "> Error: sin modelo" in md
    assert store.delete(cid) and store.get(cid) is None and not store.delete(cid)
    assert store.conversations(10, "regla") == []  # its messages left the search index too
    assert not store.rename(cid, "x")
