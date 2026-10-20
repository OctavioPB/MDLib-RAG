"""Moving the library (ADR-039): the index is reused for the same library at another place."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from biblio.config import Config
from biblio.index.indexer import IndexMismatchError, index_library
from biblio.index.schema import read_meta
from biblio.library import REBUILD, RELOCATE, SAME, browse, check_library
from biblio.retrieve.store import open_index

from fakes import FakeEmbedder, WordCounter


def _index(cfg: Config, library: Path, **kw: object) -> object:
    return index_library(
        cfg,
        library,
        full=kw.pop("full", False),  # type: ignore[arg-type]
        dtype="float32",
        embedder_factory=FakeEmbedder,
        counter_factory=WordCounter,
        **kw,  # type: ignore[arg-type]
    )


def test_relocating_reuses_the_index_and_nothing_else_is_relaxed(
    config: Config, tmp_path: Path
) -> None:
    old = tmp_path / "old"
    (old / "a").mkdir(parents=True)
    for i in range(3):
        (old / "a" / f"n{i}.md").write_text(f"# N{i}\n\n## S\n\n" + f"texto {i} " * 30, "utf-8")
    cfg = replace(
        config,
        paths=replace(config.paths, library=old, index_db=tmp_path / "index.db"),
        chunking=replace(config.chunking, min_tokens=5, target_tokens=60, max_tokens=80),
    )
    _index(cfg, old)
    new = tmp_path / "new"
    shutil.copytree(old, new)
    (new / "a" / "n0.md").write_text("# N0\n\n## S\n\n" + "cambiado " * 30, "utf-8")

    conn = open_index(cfg)
    found = check_library(cfg, new, conn)
    assert (found.plan, found.notes, found.matching) == (RELOCATE, 3, 2)
    assert check_library(cfg, old, conn).plan == SAME
    conn.close()

    with pytest.raises(IndexMismatchError):
        _index(cfg, new)  # without relocate the root is still enforced
    stats = _index(cfg, new, relocate=True)
    assert (stats.unchanged, stats.updated) == (2, 1)  # type: ignore[attr-defined]
    conn = open_index(cfg)
    assert read_meta(conn)["library"] == new.resolve().as_posix()
    conn.close()
    with pytest.raises(IndexMismatchError):  # the model is still enforced under relocate
        _index(replace(cfg, chunking=replace(cfg.chunking, max_tokens=81)), new, relocate=True)

    other = tmp_path / "other"
    other.mkdir()
    (other / "x.md").write_text("# X\n\n## S\n\n" + "otra cosa " * 30, "utf-8")
    conn = open_index(cfg)
    assert check_library(cfg, other, conn).plan == REBUILD
    conn.close()


def test_browse_lists_folders_without_hidden_ones(tmp_path: Path) -> None:
    (tmp_path / "b").mkdir()
    (tmp_path / "A").mkdir()
    (tmp_path / ".git").mkdir()
    (tmp_path / "nota.md").write_text("x", "utf-8")
    listing = browse(str(tmp_path), 1)
    assert [f.name for f in listing.folders] == ["A"] and listing.truncated
    with pytest.raises(ValueError):
        browse(str(tmp_path / "nada"), 10)
