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
    GuideMeta,
    Outline,
    OutlineSection,
    SourceRegistry,
    clean_draft,
    drop_non_content_sections,
    guide_meta,
    parse_outline,
    render_draft,
    render_outline_file,
    renumber,
    sibling_outlines,
    slugify,
    strip_repeated_title,
)
from biblio.generate.guide_style import check_structure, role_of
from biblio.generate.llm import Message
from biblio.index.indexer import index_library
from biblio.retrieve.bm25 import BM25Retriever
from biblio.retrieve.store import open_index
from biblio.retrieve.types import Hit

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
        '{"epigrafe": "Validar es desconfiar de lo que viste.", "conceptos": ["k-fold", "sesgo"]}',
    )
    cfg = replace(indexed, draft=replace(indexed.draft, evidence="library", author="A · B"))
    drafter = Drafter(cfg, BM25Retriever(conn), llm, WordCounter(), conn)

    outline = drafter.propose_outline("validación cruzada")
    # the two the model wrote, then the comparison, limitations and summary every guide has
    assert [s.title for s in outline.sections][:2] == ["1. Marco conceptual", "2. K-fold"]
    assert [role_of(s.title) for s in outline.sections[2:]] == [
        "comparison",
        "limitations",
        "summary",
    ]
    assert "Guía «Validación cruzada»" in llm.prompts[0][1].content  # sibling structure given

    registry = SourceRegistry()
    first = drafter.write_section(outline, outline.sections[0], registry)
    second = drafter.write_section(outline, outline.sections[1], registry)
    assert first.body == "Estima el error de generalización [1]." and not first.uncited
    assert second.missing_evidence
    assert "marco conceptual" in llm.prompts[1][0].content  # the role of the section is asked
    meta = drafter.frame(outline, [first, second], registry)
    assert meta.epigraph.startswith("Validar es desconfiar") and meta.concepts == (
        "k-fold",
        "sesgo",
    )

    draft = render_draft("Validación cruzada", [first, second], registry, date(2026, 10, 17), meta)
    assert draft.startswith("# Validación cruzada\n### OPB Reference Series — ")
    assert "**Autor:** A · B" in draft and "**Tema:** k-fold · sesgo" in draft
    assert '> *"Validar es desconfiar de lo que viste."*\n> — OPB' in draft
    assert "## 0. Tabla de Contenidos\n\n1. [1. Marco conceptual](#1-marco-conceptual)" in draft
    assert "## 1. Marco conceptual\n\nEstima" in draft and "## 2. K-fold" in draft
    assert "## Referencias\n\n1. ml/" in draft and "esqueleto" in draft
    assert "> Falta evidencia en la biblioteca para esta sección." in draft
    assert draft.rstrip().endswith("**OPB Reference Series** · A — B\n*From Pipeline to Decision.*")


def test_uncited_section_is_flagged() -> None:
    registry = SourceRegistry()
    from biblio.generate.draft import SectionResult

    draft = render_draft(
        "T", [SectionResult("A", "Texto sin citas.", False, True, ())], registry, date(2026, 1, 1)
    )
    assert "no cita ninguna fuente" in draft
    assert "## Referencias\n\n(ninguna)" in draft


def test_slugify() -> None:
    assert slugify("Validación cruzada en ML!") == "validacion-cruzada-en-ml"
    assert slugify("¿?") == "borrador"


# --- the series' shape (ADR-061) ----------------------------------------------------------------


def _full_guide() -> str:
    titles = [
        "Marco conceptual", "Fundamentos", "El problema base", "Garantías", "Mecánica",
        "Estructura de soporte", "Análisis de complejidad", "Comparativa con alternativas",
        "Aplicaciones en Data & AI Engineering", "Limitaciones y cuándo falla",
        "Resumen y framework de decisión",
    ]  # fmt: skip
    bodies = {
        "Comparativa con alternativas": "| A | B |\n|---|---|\n| x [1] | y |",
        "Resumen y framework de decisión": "- síntesis\n\n```\n¿Cabe? SÍ → usar\n NO → otra\n```",
    }
    from biblio.generate.draft import SectionResult

    sections = [SectionResult(t, bodies.get(t, "Texto [1]."), False, False, ()) for t in titles]
    registry = SourceRegistry()
    registry.number(frag(1, "AI/Books/Libro/01.md"))
    meta = GuideMeta("A · B", "AI", "AI → T", "Intermedio", ("c1", "c2"), "Una idea.")
    return render_draft("T", sections, registry, date(2026, 11, 2), meta)


def test_a_rendered_guide_has_every_block_of_the_series() -> None:
    report = check_structure(_full_guide())
    assert report.failed() == [] and report.sections == 11 and report.score == 1.0


def test_check_structure_names_what_a_plain_draft_lacks() -> None:
    plain = "# Tema\n\n> **Borrador**\n\n## Uno\n\nTexto.\n\n## Fuentes\n\n1. a.md\n"
    failed = check_structure(plain).failed()
    assert {"series line", "epigraph", "table of contents", "footer", "decision tree"} <= set(
        failed
    )
    assert check_structure(plain).score < 0.3


def test_sections_get_the_shape_their_role_asks_for() -> None:
    assert role_of("8. Comparativa con alternativas") == "comparison"
    assert role_of("10. Limitaciones y cuándo falla") == "limitations"
    assert role_of("11. Resumen ejecutivo y framework de decisión") == "summary"
    assert role_of("Cuándo NO usarlo") == "limitations"
    assert role_of("7. Análisis de complejidad") == "complexity"
    assert role_of("9. Aplicaciones en Data & AI Engineering") == "applications"
    assert role_of("1. Marco Conceptual") == "concept"
    assert role_of("5. Mecánica del algoritmo") == "body"


def test_clean_draft_takes_a_draft_to_the_publishable_form() -> None:
    clean = clean_draft(_full_guide())
    body, references = clean.split("## Referencias")
    assert "[1]" not in body and "Borrador generado" not in clean
    assert "| x | y |" in body
    assert references.strip().splitlines()[0].startswith("- AI/Books/Libro/01.md")
    assert check_structure(clean).score == 1.0  # still the series' shape


def test_evidence_comes_from_books_and_one_chapter_gives_two_fragments(indexed: Config) -> None:
    conn = open_index(indexed)
    hits = [
        Hit(i, doc, ("T", f"S{i}"), "texto", 1, 2, i, 0.0, "fake")
        for i, doc in enumerate(["AI/Books/L/01.md"] * 4 + ["mis/notas.md", "AI/Books/L/02.md"], 1)
    ]

    class Fixed:
        name = "fixed"

        def search(self, query: str, k: int) -> list[Hit]:
            return hits

        def close(self) -> None:
            pass

    drafter = Drafter(indexed, Fixed(), _ScriptedLLM(), WordCounter(), conn)
    picked = [h.doc_path for h in drafter._evidence("tema")]
    assert picked == ["AI/Books/L/01.md", "AI/Books/L/01.md", "AI/Books/L/02.md"]
    library = replace(indexed, draft=replace(indexed.draft, evidence="library"))
    assert "mis/notas.md" in [
        h.doc_path
        for h in Drafter(library, Fixed(), _ScriptedLLM(), WordCounter(), conn)._evidence("tema")
    ]


def test_the_topic_is_also_searched_in_english(indexed: Config) -> None:
    conn = open_index(indexed)
    asked: list[str] = []

    class Spy:
        name = "spy"

        def search(self, query: str, k: int) -> list[Hit]:
            asked.append(query)
            return []

        def close(self) -> None:
            pass

    drafter = Drafter(
        indexed, Spy(), _ScriptedLLM(), WordCounter(), conn, lambda _q: "cross validation"
    )
    drafter._evidence("validación cruzada")
    assert asked == ["validación cruzada", "cross validation"]


def test_guide_meta_takes_the_area_of_the_sources_and_survives_a_bad_reply(indexed: Config) -> None:
    registry = SourceRegistry()
    registry.number(frag(1, "Data Engineering/Books/Kafka/01.md"))
    registry.number(frag(2, "Data Engineering/Books/Kafka/02.md"))
    registry.number(frag(3, "Cloud/Books/X/01.md"))
    meta = guide_meta(
        indexed, "Kafka", registry, {"epigrafe": " Un log. ", "conceptos": ["topics", "", "ISR"]}
    )
    assert (meta.track, meta.series) == ("Data Engineering", "Data Engineering → Kafka")
    assert meta.concepts == ("topics", "ISR") and meta.epigraph == "Un log."
    assert guide_meta(indexed, "Kafka", registry, None).epigraph == ""


def test_notes_written_with_the_section_mark_stay_notes() -> None:
    text = (
        "## 1. Marco\n## Qué es y por qué importa\n## 2. Mecánica\n## Cómo funciona\n"
        "## 3. Comparativa\nfrente a otras\n## 4. Límites\n"
    )
    outline = parse_outline(text, "X")
    assert [s.title for s in outline.sections] == [
        "1. Marco",
        "2. Mecánica",
        "3. Comparativa",
        "4. Límites",
    ]
    assert outline.sections[0].notes == "Qué es y por qué importa"
    assert outline.sections[2].notes == "frente a otras"


def test_the_closing_sections_of_the_series_are_always_there() -> None:
    from biblio.generate.draft import ensure_closing_sections

    titles = [f"{i}. Tema {i}" for i in range(1, 12)]  # eleven body sections, none of the three
    full = ensure_closing_sections(Outline("T", tuple(OutlineSection(t) for t in titles)), 13)
    assert len(full.sections) == 13
    assert [role_of(s.title) for s in full.sections[-3:]] == [
        "comparison",
        "limitations",
        "summary",
    ]
    assert full.sections[0].title == "1. Tema 1" and full.sections[7].title == "8. Tema 8"
    # one the model wrote stays where it put it; the summary always goes last
    own = Outline(
        "T",
        (
            OutlineSection("1. Marco"),
            OutlineSection("2. Comparativa de motores"),
            OutlineSection("3. Resumen y decisión"),
            OutlineSection("4. Límites y cuándo no usarlo"),
            OutlineSection("5. Mecánica"),
        ),
    )
    titles_out = [s.title for s in ensure_closing_sections(own, 13).sections]
    assert titles_out[:3] == ["1. Marco", "2. Comparativa de motores", "5. Mecánica"]
    assert [role_of(t) for t in titles_out[-2:]] == ["limitations", "summary"]
    assert titles_out[-2] == "4. Límites y cuándo no usarlo"


def test_a_repeated_numbered_title_is_stripped() -> None:
    body = "### Limitaciones y cuándo no usarla\n\nTexto [1]."
    assert strip_repeated_title(body, "13. Limitaciones y cuándo no usarla") == "Texto [1]."
