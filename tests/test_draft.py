from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from biblio.config import Config
from biblio.generate.context import Fragment
from biblio.generate.draft import (
    Drafter,
    Outline,
    OutlineSection,
    SourceRegistry,
    drop_non_content_sections,
    parse_outline,
    render_draft,
    render_outline_file,
    renumber,
    sibling_outlines,
    slugify,
    strip_repeated_title,
)
from biblio.generate.llm import Message
from biblio.index.indexer import index_library
from biblio.retrieve.bm25 import BM25Retriever
from biblio.retrieve.store import open_index

from fakes import FakeEmbedder, WordCounter

GUIDES = {
    "ml/cv.md": "# Validación cruzada\n\n## 1. Marco conceptual\n\n"
    + "La validación cruzada estima el error de generalización con particiones. " * 3
    + "\n\n## 2. K-fold\n\n"
    + "K-fold divide los datos en k particiones y promedia el error de validación. " * 3,
    "ml/bootstrap.md": "# Bootstrap\n\n## 1. Marco conceptual\n\n"
    + "El bootstrap remuestrea con reemplazo para estimar la variabilidad. " * 3,
}


@pytest.fixture
def indexed(config: Config, tmp_path: Path) -> Config:
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
    return cfg


def frag(n: int, path: str, start: int = 1) -> Fragment:
    return Fragment(n, path, ("Guía", f"S{n}"), "texto", start, start + 5)


class _ScriptedLLM:
    """Replies in order; records what it was asked."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[Sequence[Message]] = []

    def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
        self.prompts.append(messages)
        yield self.replies.pop(0)

    def close(self) -> None:
        pass


# --- outline ------------------------------------------------------------------------------------


def test_parse_outline_reads_sections_notes_and_skips_comments() -> None:
    text = (
        "# Esquema: X\n<!-- ## no es sección\n   tampoco -->\n\n"
        "## 1. Marco conceptual\nqué es y por qué\n\n## 2. K-fold\n### detalle ignorado\n"
    )
    outline = parse_outline(text, "X")
    assert outline.sections == (
        OutlineSection("1. Marco conceptual", "qué es y por qué"),
        OutlineSection("2. K-fold", ""),
    )


def test_outline_file_round_trips() -> None:
    outline = Outline("CV", (OutlineSection("A", "nota"), OutlineSection("B")))
    assert (
        parse_outline(render_outline_file(outline, 'biblio draft "CV" --outline x'), "CV")
        == outline
    )


def test_navigation_and_title_sections_are_dropped() -> None:
    outline = Outline(
        "Validación cruzada",
        tuple(
            OutlineSection(t)
            for t in ["Guía de Validación Cruzada", "Índice", "0. Tabla de Contenidos", "1. K-fold"]
        ),
    )
    assert [s.title for s in drop_non_content_sections(outline).sections] == ["1. K-fold"]


def test_sibling_outlines_show_the_structure_of_related_guides(indexed: Config) -> None:
    text = sibling_outlines(open_index(indexed), ["ml/cv.md"], max_lines=10)
    assert "Guía «Validación cruzada» (ml/cv.md):" in text
    assert "## 1. Marco conceptual" in text and "## 2. K-fold" in text


# --- sections -----------------------------------------------------------------------------------


def test_renumber_maps_local_citations_to_global_sources() -> None:
    registry = SourceRegistry()
    registry.number(frag(9, "previa.md"))
    body, invalid = renumber(
        "A [1]. B [2][1]. C [1, 2]. D [5].", [frag(1, "a.md"), frag(2, "b.md")], registry
    )
    assert body == "A [2]. B [3][2]. C [2][3]. D [?]."
    assert invalid == (5,)
    assert [f.doc_path for f in registry.fragments] == ["previa.md", "a.md", "b.md"]


def test_repeated_title_is_stripped() -> None:
    assert strip_repeated_title("### 2. K-fold\n\nTexto [1].", "2. K-fold") == "Texto [1]."
    assert strip_repeated_title("### Otro\n\nTexto.", "2. K-fold") == "### Otro\n\nTexto."


def test_drafter_end_to_end(indexed: Config) -> None:
    conn = open_index(indexed)
    llm = _ScriptedLLM(
        "## 1. Marco conceptual\nqué es\n## Índice\n## 2. K-fold\ncómo",
        "### 1. Marco conceptual\nEstima el error de generalización [1].",
        "Falta evidencia en la biblioteca para esta sección.",
    )
    drafter = Drafter(indexed, BM25Retriever(conn), llm, WordCounter(), conn)

    outline = drafter.propose_outline("validación cruzada")
    assert [s.title for s in outline.sections] == ["1. Marco conceptual", "2. K-fold"]
    assert "Guía «Validación cruzada»" in llm.prompts[0][1].content  # sibling structure given

    registry = SourceRegistry()
    first = drafter.write_section(outline, outline.sections[0], registry)
    second = drafter.write_section(outline, outline.sections[1], registry)
    assert first.body == "Estima el error de generalización [1]." and not first.uncited
    assert second.missing_evidence

    draft = render_draft("Validación cruzada", [first, second], registry, date(2026, 10, 17))
    assert draft.startswith("# Validación cruzada\n\n> **Borrador generado por `biblio draft`")
    assert "esqueleto" in draft and "## Fuentes\n\n1. ml/" in draft
    assert "> Falta evidencia en la biblioteca para esta sección." in draft


def test_uncited_section_is_flagged() -> None:
    registry = SourceRegistry()
    from biblio.generate.draft import SectionResult

    draft = render_draft(
        "T", [SectionResult("A", "Texto sin citas.", False, True, ())], registry, date(2026, 1, 1)
    )
    assert "no cita ninguna fuente" in draft
    assert "## Fuentes\n\n(ninguna)" in draft


def test_slugify() -> None:
    assert slugify("Validación cruzada en ML!") == "validacion-cruzada-en-ml"
    assert slugify("¿?") == "borrador"
