from __future__ import annotations

import logging
from pathlib import Path

import pytest

from biblio.config import ChunkingConfig
from biblio.ingest.chunker import CHUNKER_VERSION, chunk_document
from biblio.ingest.parser import parse_file, parse_text
from biblio.ingest.types import Chunk, ParsedDocument

FIXTURES = Path(__file__).parent / "fixtures"
PATHOLOGICAL = FIXTURES / "pathological"
# Real guides from the library (Phase 1 task); the property tests cover them automatically.
LIBRARY = FIXTURES / "library"

# Small thresholds measured in words keep the fixtures short and the expectations readable.
CFG = ChunkingConfig(min_tokens=8, target_tokens=40, max_tokens=60)


class WordCounter:
    def count(self, text: str) -> int:
        return len(text.split())


def _chunk(name: str) -> tuple[ParsedDocument, list[Chunk]]:
    doc = parse_file(PATHOLOGICAL / name, name)
    return doc, chunk_document(doc, WordCounter(), CFG)


def _paths(chunks: list[Chunk]) -> list[tuple[str, ...]]:
    return [c.heading_path for c in chunks]


def _all_fixtures() -> list[Path]:
    return sorted(PATHOLOGICAL.glob("*.md")) + sorted(LIBRARY.rglob("*.md"))


# --- properties that hold for every document -------------------------------------------------


@pytest.mark.parametrize("path", _all_fixtures(), ids=lambda p: p.name)
def test_chunks_partition_the_body(path: Path) -> None:
    doc = parse_file(path, path.name)
    chunks = chunk_document(doc, WordCounter(), CFG)

    covered: list[int] = []
    for i, chunk in enumerate(chunks):
        assert chunk.ordinal == i
        assert chunk.text == "\n".join(doc.lines[chunk.start_line - 1 : chunk.end_line])
        assert chunk.text.strip()
        covered += range(chunk.start_line - 1, chunk.end_line)
    # In order, without duplication...
    assert covered == sorted(set(covered))
    # ...and without loss: every non-blank body line is in a chunk; nothing from the frontmatter is.
    body = [i for i in range(doc.body_start, len(doc.lines)) if doc.lines[i].strip()]
    assert set(body) <= set(covered)
    assert all(i >= doc.body_start for i in covered)


@pytest.mark.parametrize("path", _all_fixtures(), ids=lambda p: p.name)
def test_atomic_blocks_are_never_split(path: Path) -> None:
    doc = parse_file(path, path.name)
    chunks = chunk_document(doc, WordCounter(), CFG)
    for block in (b for b in doc.blocks if b.atomic):
        holders = [c for c in chunks if c.start_line - 1 <= block.start and block.end <= c.end_line]
        assert len(holders) == 1, f"{block.kind} at line {block.start + 1} is split"


# --- pathological cases ----------------------------------------------------------------------


def test_no_headings_uses_file_name_as_root() -> None:
    _, chunks = _chunk("no_headings.md")
    assert _paths(chunks) == [("no_headings",)]


def test_frontmatter_only_and_empty_produce_no_chunks() -> None:
    doc, chunks = _chunk("frontmatter_only.md")
    assert doc.title == "Plantilla vacía"
    assert chunks == []
    assert _chunk("empty.md")[1] == []
    assert chunk_document(parse_text("\n\n   \n", "blank.md"), WordCounter(), CFG) == []


def test_hash_lines_inside_code_are_not_headings() -> None:
    _, chunks = _chunk("code_with_hash.md")
    root = "Scripts de shell"
    assert _paths(chunks) == [(root, "Comentarios"), (root, "Variables")]
    assert "# Esto es un comentario" in chunks[0].text
    assert "## tampoco esto" in chunks[0].text
    # The bodyless H1 travels with the first section instead of becoming a chunk of its own.
    assert chunks[0].start_line == 1


def test_setext_headings() -> None:
    _, chunks = _chunk("setext.md")
    assert _paths(chunks) == [("Guía setext",), ("Guía setext", "Primera parte")]


def test_level_jumps() -> None:
    _, chunks = _chunk("level_jump.md")
    root = "Álgebra lineal"
    assert _paths(chunks) == [(root, "Autovalores"), (root, "Descomposiciones")]


def test_huge_table_stays_whole_and_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        _, chunks = _chunk("huge_table.md")
    table = [c for c in chunks if "| Bernoulli |" in c.text]
    assert len(table) == 1
    assert "| Wishart |" in table[0].text
    assert table[0].token_count > CFG.max_tokens
    assert "exceeds max_tokens" in caplog.text
    assert set(_paths(chunks)) == {("Distribuciones", "Tabla de distribuciones")}


def test_multiline_math_with_blank_line_is_one_unit() -> None:
    _, chunks = _chunk("multiline_math.md")
    assert len(chunks) > 1
    holders = [c for c in chunks if "$$" in c.text]
    assert len(holders) == 1
    assert holders[0].text.count("$$") == 2
    assert "\\text{con }" in holders[0].text


def test_duplicate_headings_keep_distinct_paths() -> None:
    _, chunks = _chunk("duplicate_headings.md")
    root = "Optimización"
    assert _paths(chunks) == [
        (root, "Gradiente descendente", "Ejemplo"),
        (root, "Newton", "Ejemplo"),
    ]


def test_tiny_last_section_merges_into_previous_with_common_path() -> None:
    _, chunks = _chunk("tiny_last_section.md")
    assert _paths(chunks) == [("Probabilidad", "Teorema de Bayes")]
    assert "Ver la regla." in chunks[0].text


def test_tiny_section_merges_into_next_sibling_with_common_path() -> None:
    _, chunks = _chunk("tiny_sibling.md")
    assert _paths(chunks) == [("Redes neuronales",)]
    assert "Poco aquí" in chunks[0].text and "ReLU" in chunks[0].text


def test_empty_heading_is_plain_content() -> None:
    _, chunks = _chunk("empty_heading.md")
    assert _paths(chunks) == [("Notas sueltas",)]
    assert "Segundo párrafo" in chunks[0].text


def test_html_block_is_kept() -> None:
    _, chunks = _chunk("html_block.md")
    assert len(chunks) == 1
    assert "<details>" in chunks[0].text and "</details>" in chunks[0].text


def test_invalid_yaml_does_not_stop_ingestion(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        doc, chunks = _chunk("invalid_yaml.md")
    assert "invalid YAML" in caplog.text
    assert doc.title == "invalid_yaml"
    assert len(chunks) == 1
    assert chunks[0].start_line == 6


def test_long_section_splits_between_blocks_with_same_path() -> None:
    _, chunks = _chunk("long_section.md")
    assert len(chunks) >= 3
    assert set(_paths(chunks)) == {("Arquitectura", "Flujo de datos")}
    assert all(c.token_count <= CFG.max_tokens for c in chunks)
    # No overlap: each paragraph appears exactly once.
    assert sum(c.text.count("El chunker nunca parte") for c in chunks) == 1


def test_long_list_splits_between_items() -> None:
    _, chunks = _chunk("long_list.md")
    assert len(chunks) >= 2
    items = [line for c in chunks for line in c.text.splitlines() if line.startswith("- ")]
    assert len(items) == 10


def test_links_are_collected_per_chunk() -> None:
    doc = parse_file(PATHOLOGICAL / "links.md", "notas/links.md")
    (chunk,) = chunk_document(doc, WordCounter(), CFG)
    assert chunk.links == (
        "Teorema de Bayes.md",
        "estadistica/fisher.md",
        "Gradiente.md",
        "algebra/svd.md",
        "indice.md",
    )


def test_embedding_text_prefixes_breadcrumb_but_text_does_not() -> None:
    _, chunks = _chunk("level_jump.md")
    chunk = chunks[0]
    assert chunk.breadcrumb == "Álgebra lineal › Autovalores"
    assert chunk.embedding_text.startswith("Álgebra lineal › Autovalores\n\n")
    assert "›" not in chunk.text


def test_crlf_and_bom_keep_line_numbers(tmp_path: Path) -> None:
    text = "﻿# Guía\r\n\r\n## A\r\n\r\nuno dos tres cuatro cinco seis siete ocho nueve\r\n"
    path = tmp_path / "crlf.md"
    path.write_bytes(text.encode("utf-8"))
    doc = parse_file(path, "crlf.md")
    (chunk,) = chunk_document(doc, WordCounter(), CFG)
    assert chunk.heading_path == ("Guía", "A")
    assert (chunk.start_line, chunk.end_line) == (1, 5)


def test_chunker_version_is_a_positive_int() -> None:
    assert isinstance(CHUNKER_VERSION, int) and CHUNKER_VERSION > 0


@pytest.mark.models
def test_real_tokenizer_counts_tokens_not_words() -> None:
    from biblio.config import load_config
    from biblio.ingest.tokens import load_token_counter

    config = load_config()
    counter = load_token_counter(config)
    assert counter.count("") == 0
    text = "La descomposición en valores singulares"
    assert counter.count(text) >= len(text.split())
    doc = parse_file(PATHOLOGICAL / "long_section.md", "long_section.md")
    chunks = chunk_document(doc, counter, config.chunking)
    assert chunks and all(c.token_count > 0 for c in chunks)


@pytest.mark.models
def test_cli_chunks_prints_table_and_texts() -> None:
    from typer.testing import CliRunner

    from biblio.cli import app

    result = CliRunner().invoke(app, ["chunks", str(PATHOLOGICAL / "code_with_hash.md")])
    assert result.exit_code == 0, result.output
    # At real thresholds both short sections merge and are cited by their common ancestor.
    assert "1 chunks" in result.output
    assert "Scripts de shell" in result.output
    assert "# Esto es un comentario" in result.output


@pytest.mark.parametrize("path", _all_fixtures(), ids=lambda p: p.name)
def test_only_a_single_chunk_document_can_be_below_min_tokens(path: Path) -> None:
    chunks = chunk_document(parse_file(path, path.name), WordCounter(), CFG)
    small = [c for c in chunks if c.token_count < CFG.min_tokens]
    assert not small or len(chunks) == 1


def test_lead_in_sentences_stay_with_what_they_introduce() -> None:
    _, chunks = _chunk("lead_in.md")
    assert len(chunks) > 1
    (formula,) = [c for c in chunks if "$$D(X)" in c.text]
    assert "se calcula como:" in formula.text
    (table,) = [c for c in chunks if "| Término |" in c.text]
    assert "**La actualización visualizada:**" in table.text


def test_unclosed_fence_is_reported(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        _, chunks = _chunk("unclosed_fence.md")
    assert "unclosed_fence.md:7: code fence is never closed" in caplog.text
    assert _paths(chunks) == [("Diagrama", "Ciclo")]
    assert "## Esto ya no es un encabezado" in chunks[0].text


def test_closed_fences_are_not_reported(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        _chunk("code_with_hash.md")
        _chunk("long_section.md")
    assert "never closed" not in caplog.text
