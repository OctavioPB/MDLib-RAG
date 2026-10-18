from __future__ import annotations

import pytest

from biblio.ingest.links import markdown_link_target, wikilink_targets


def test_wikilinks() -> None:
    text = "[[Bayes]] [[dir/Fisher|alias]] [[Gradiente#Ejemplo]] [[ya.md]] [[ ]] [[#solo-ancla]]"
    assert wikilink_targets(text) == ["Bayes.md", "dir/Fisher.md", "Gradiente.md", "ya.md"]


@pytest.mark.parametrize(
    ("href", "expected"),
    [
        ("otra.md", "notas/otra.md"),
        ("./otra.md#seccion", "notas/otra.md"),
        ("../algebra/svd.md", "algebra/svd.md"),
        ("/indice.md", "indice.md"),
        ("mi%20guia.md", "notas/mi guia.md"),
        ("OTRA.MD", "notas/OTRA.MD"),
        ("../../fuera.md", None),
        ("https://example.com/a.md", None),
        ("//example.com/a.md", None),
        ("mailto:a@b.md", None),
        ("imagen.png", None),
        ("#ancla", None),
        ("C:/notas/a.md", None),
    ],
)
def test_markdown_links(href: str, expected: str | None) -> None:
    assert markdown_link_target(href, "notas/actual.md") == expected
