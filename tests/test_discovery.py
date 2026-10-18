from __future__ import annotations

import os
from pathlib import Path

import pytest

from biblio.ingest.discovery import discover, glob_to_regex

INCLUDE = ("**/*.md",)
EXCLUDE = ("**/node_modules/**", "**/.git/**", "**/_drafts/**")


def _touch(root: Path, rel: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# x\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("pattern", "path", "matches"),
    [
        ("**/*.md", "a.md", True),
        ("**/*.md", "x/y/a.md", True),
        ("**/*.md", "a.txt", False),
        ("*.md", "x/a.md", False),
        ("**/_drafts/**", "_drafts/a.md", True),
        ("**/_drafts/**", "x/_drafts/y/a.md", True),
        ("**/_drafts/**", "x/drafts/a.md", False),
        ("notas/?.md", "notas/a.md", True),
        ("**/*.md", "A.MD", True),
    ],
)
def test_glob_to_regex(pattern: str, path: str, matches: bool) -> None:
    assert bool(glob_to_regex(pattern).fullmatch(path)) is matches


def test_discover_applies_include_and_exclude(tmp_path: Path) -> None:
    for rel in [
        "b.md",
        "a/guia.md",
        "a/b/profunda.md",
        "notas.txt",
        "_drafts/borrador.md",
        "x/node_modules/pkg/readme.md",
        ".git/info.md",
    ]:
        _touch(tmp_path, rel)
    assert discover(tmp_path, INCLUDE, EXCLUDE) == ["a/b/profunda.md", "a/guia.md", "b.md"]


def test_discover_ignores_symlinks(tmp_path: Path) -> None:
    library = tmp_path / "lib"
    outside = tmp_path / "outside"
    _touch(library, "real.md")
    _touch(outside, "fuera.md")
    try:
        os.symlink(outside / "fuera.md", library / "enlace.md")
        os.symlink(outside, library / "dir_enlazado", target_is_directory=True)
    except OSError:
        pytest.skip("symlinks not permitted on this system")
    assert discover(library, INCLUDE, EXCLUDE) == ["real.md"]
