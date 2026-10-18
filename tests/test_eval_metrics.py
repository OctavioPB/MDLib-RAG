from __future__ import annotations

import math

import pytest

from biblio.eval.metrics import (
    all_found_at,
    matches_file,
    matches_section,
    ndcg_at,
    normalize,
    recall_at,
    reciprocal_rank,
    score_question,
)
from biblio.eval.questions import Expected, Location
from biblio.retrieve.types import Hit


def hit(rank: int, path: str, *heading: str, text: str = "texto") -> Hit:
    return Hit(
        chunk_id=rank,
        doc_path=path,
        heading_path=("Guía", *heading),
        text=text,
        start_line=1,
        end_line=2,
        rank=rank,
        score=1.0 / rank,
        source="test",
    )


def test_normalize_ignores_case_accents_and_spaces() -> None:
    assert normalize("  Regresión   LINEAL ") == normalize("regresion lineal")


def test_section_match_by_last_heading() -> None:
    exp = Expected("est/regresion.md", "Mínimos cuadrados")
    assert matches_section(hit(1, "est/regresion.md", "Modelo", "Minimos Cuadrados"), exp)
    assert not matches_section(hit(1, "est/regresion.md", "Mínimos cuadrados", "Ejemplo"), exp)
    assert not matches_section(hit(1, "otra.md", "Mínimos cuadrados"), exp)
    assert matches_file(hit(1, "EST/Regresion.md", "x"), exp)


def test_section_match_inside_a_merged_chunk() -> None:
    merged = hit(
        1, "est/regresion.md", "Modelo", text="## Modelo\n\nTexto\n\n### Mínimos cuadrados ##\n"
    )
    assert matches_section(merged, Expected("est/regresion.md", "Mínimos cuadrados"))


def test_scores_credit_each_expected_once() -> None:
    expected = [Expected("a.md", "Uno"), Expected("b.md", "Dos")]
    hits = [
        hit(1, "a.md", "Uno"),
        hit(2, "a.md", "Uno"),  # same section split in two chunks
        hit(3, "c.md", "Otro"),
        hit(4, "b.md", "Dos"),
    ]
    score = score_question(hits, expected, "section")
    assert score.expected_ranks == (1, 4)
    assert score.first_rank == 1
    assert recall_at(score, 1) == 0.5
    assert recall_at(score, 5) == 1.0
    assert reciprocal_rank(score) == 1.0
    assert all_found_at(score, 3) == 0.0 and all_found_at(score, 4) == 1.0
    ideal = 1 + 1 / math.log2(3)
    assert ndcg_at(score, 8) == pytest.approx((1 + 1 / math.log2(5)) / ideal)
    assert ndcg_at(score, 8) <= 1.0


def test_file_granularity_deduplicates_expected_files() -> None:
    expected = [Expected("a.md", "Uno"), Expected("a.md", "Dos")]
    score = score_question([hit(1, "x.md", "z"), hit(2, "a.md", "Otro")], expected, "file")
    assert score.expected_ranks == (2,)
    assert reciprocal_rank(score) == 0.5


def test_nothing_found() -> None:
    score = score_question([hit(1, "x.md", "z")], [Expected("a.md", "Uno")], "section")
    assert score.first_rank is None
    assert recall_at(score, 20) == 0.0
    assert reciprocal_rank(score) == 0.0
    assert ndcg_at(score, 8) == 0.0


def test_alternatives_satisfy_an_expected_item_once() -> None:
    exp = Expected("a.md", "Uno", (Location("b.md", "Otra forma"), Location("c.md", "Tercera")))
    hits = [hit(1, "x.md", "z"), hit(2, "b.md", "Otra forma"), hit(3, "a.md", "Uno")]
    section = score_question(hits, [exp], "section")
    assert section.expected_ranks == (2,)  # credited once, at the best option
    assert ndcg_at(section, 8) <= 1.0
    assert matches_file(hit(1, "c.md", "q"), exp)
    file = score_question([hit(1, "c.md", "q")], [exp, Expected("c.md", "Tercera")], "file")
    assert len(file.expected_ranks) == 2  # different acceptable-file sets stay separate items
