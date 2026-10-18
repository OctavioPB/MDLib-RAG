from __future__ import annotations

import socket
import sqlite3
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from biblio.config import Config
from biblio.index.embedder import DeviceChoice, Vectors
from biblio.index.indexer import (
    CounterFactory,
    EmbedderFactory,
    IndexMismatchError,
    IndexStats,
    LLMServerRunningError,
    ensure_gpu_free,
    index_library,
)
from biblio.index.schema import connect, read_meta

from fakes import DIM, FakeEmbedder, WordCounter


def _never() -> FakeEmbedder:
    raise AssertionError("the embedder must not be loaded when nothing changed")


def _never_counter() -> WordCounter:
    raise AssertionError("the tokenizer must not be loaded when nothing changed")


GUIDES = {
    "estadistica/regresion.md": "# Regresión\n\n## Mínimos cuadrados\n\n"
    + "La regresión lineal estima coeficientes minimizando la suma de residuos al cuadrado. " * 3,
    "estadistica/bayes.md": "# Bayes\n\n## Teorema\n\n"
    + "El teorema de Bayes invierte probabilidades condicionadas usando la marginal. " * 3,
    "algebra/svd.md": "# SVD\n\n## Definición\n\n"
    + "La descomposición en valores singulares factoriza cualquier matriz real. " * 3,
    "vacia.md": "",
}


@pytest.fixture
def library(tmp_path: Path) -> Path:
    root = tmp_path / "library"
    for rel, text in GUIDES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


@pytest.fixture
def cfg(config: Config, tmp_path: Path) -> Config:
    small = replace(config.chunking, min_tokens=5, target_tokens=40, max_tokens=60)
    return replace(
        config,
        chunking=small,
        paths=replace(config.paths, index_db=tmp_path / "data" / "index.db"),
        embedding=replace(config.embedding, index_flush_chunks=2),
    )


def _index(
    cfg: Config,
    library: Path,
    full: bool = False,
    embedder: EmbedderFactory | None = None,
    counter: CounterFactory | None = None,
) -> IndexStats:
    return index_library(
        cfg,
        library,
        full=full,
        dtype="float32",
        embedder_factory=embedder or FakeEmbedder,
        counter_factory=counter or WordCounter,
    )


def _db(cfg: Config) -> sqlite3.Connection:
    return connect(cfg.paths.index_db)


def _assert_consistent(conn: sqlite3.Connection) -> None:
    chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    vectors = conn.execute("SELECT count(*) FROM vec_chunks").fetchone()[0]
    assert chunks == vectors
    orphans = conn.execute(
        "SELECT count(*) FROM chunks WHERE document_id NOT IN (SELECT id FROM documents)"
    ).fetchone()[0]
    assert orphans == 0
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('integrity-check')")


def _fts(conn: sqlite3.Connection, query: str) -> list[str]:
    rows = conn.execute(
        "SELECT d.path FROM chunks_fts f JOIN chunks c ON c.id = f.rowid "
        "JOIN documents d ON d.id = c.document_id "
        "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts)",
        (query,),
    )
    return [r[0] for r in rows]


def test_full_build(cfg: Config, library: Path) -> None:
    stats = _index(cfg, library, full=True)
    assert (stats.added, stats.updated, stats.deleted, stats.unchanged) == (4, 0, 0, 0)
    assert stats.total_documents == 4
    assert stats.total_chunks == stats.chunks_written > 0
    assert not Path(f"{cfg.paths.index_db}.tmp").exists()

    conn = _db(cfg)
    meta = read_meta(conn)
    assert meta["embedding_model"] == "BAAI/bge-m3"
    assert meta["embedding_dim"] == str(DIM)
    assert meta["index_dtype"] == "float32"
    _assert_consistent(conn)
    # The empty note is recorded (so it is not re-read every run) but has no chunks.
    empty = conn.execute(
        "SELECT count(*) FROM chunks c JOIN documents d ON d.id = c.document_id "
        "WHERE d.path = 'vacia.md'"
    ).fetchone()[0]
    assert empty == 0


def test_incremental_without_changes_loads_no_model(cfg: Config, library: Path) -> None:
    _index(cfg, library)
    stats = _index(cfg, library, embedder=_never, counter=_never_counter)
    assert (stats.added, stats.updated, stats.deleted, stats.unchanged) == (0, 0, 0, 4)
    assert stats.chunks_written == 0


def test_modified_file_is_reindexed(cfg: Config, library: Path) -> None:
    _index(cfg, library)
    (library / "algebra/svd.md").write_text(
        "# SVD\n\n## Uso\n\nLos autovalores aparecen en el análisis de componentes principales.\n",
        encoding="utf-8",
    )
    stats = _index(cfg, library)
    assert (stats.added, stats.updated, stats.unchanged) == (0, 1, 3)
    conn = _db(cfg)
    assert _fts(conn, "autovalores") == ["algebra/svd.md"]
    assert _fts(conn, "singulares") == []
    _assert_consistent(conn)


def test_deleted_and_new_files(cfg: Config, library: Path) -> None:
    _index(cfg, library)
    (library / "estadistica/bayes.md").unlink()
    (library / "nueva.md").write_text(
        "# Nueva\n\nUna guía nueva sobre series temporales.\n", "utf-8"
    )
    stats = _index(cfg, library)
    assert (stats.added, stats.deleted, stats.unchanged) == (1, 1, 3)
    conn = _db(cfg)
    assert _fts(conn, "Bayes") == []
    assert _fts(conn, "temporales") == ["nueva.md"]
    _assert_consistent(conn)


def test_bm25_folds_accents(cfg: Config, library: Path) -> None:
    _index(cfg, library)
    conn = _db(cfg)
    assert _fts(conn, "regresion")[0] == "estadistica/regresion.md"
    assert _fts(conn, "definicion") == ["algebra/svd.md"]


def test_vector_search_finds_identical_text(cfg: Config, library: Path) -> None:
    _index(cfg, library)
    conn = _db(cfg)
    chunk_id, text, breadcrumb = conn.execute(
        "SELECT c.id, c.text, c.breadcrumb FROM chunks c JOIN documents d ON d.id = c.document_id "
        "WHERE d.path = 'estadistica/bayes.md' LIMIT 1"
    ).fetchone()
    query = FakeEmbedder().embed([f"{breadcrumb}\n\n{text}"])[0]
    (best, distance) = conn.execute(
        "SELECT chunk_id, distance FROM vec_chunks WHERE embedding MATCH ? AND k = 3 "
        "ORDER BY distance",
        (query.tobytes(),),
    ).fetchone()
    assert best == chunk_id
    assert distance == pytest.approx(0.0, abs=1e-6)


def test_config_change_requires_full(cfg: Config, library: Path) -> None:
    _index(cfg, library)
    changed = replace(cfg, chunking=replace(cfg.chunking, target_tokens=45))
    with pytest.raises(IndexMismatchError, match="chunking"):
        _index(changed, library)
    stats = _index(changed, library, full=True)
    assert stats.added == 4
    assert read_meta(_db(changed))["chunking"].count("45") == 1


def test_library_change_requires_full(cfg: Config, library: Path, tmp_path: Path) -> None:
    _index(cfg, library)
    other = tmp_path / "otra"
    other.mkdir()
    with pytest.raises(IndexMismatchError, match="library"):
        _index(cfg, other)


def test_dimension_change_is_refused(cfg: Config, library: Path) -> None:
    _index(cfg, library)
    (library / "nueva.md").write_text("# Nueva\n\nTexto nuevo.\n", "utf-8")

    class Wider(FakeEmbedder):
        @property
        def dimension(self) -> int:
            return DIM * 2

    with pytest.raises(IndexMismatchError, match="dimension"):
        _index(cfg, library, embedder=Wider)


def test_interrupted_incremental_run_keeps_committed_documents(cfg: Config, library: Path) -> None:
    class Flaky(FakeEmbedder):
        def embed(self, texts: Sequence[str]) -> Vectors:
            if self.calls >= 1:
                raise RuntimeError("boom")
            return super().embed(texts)

    with pytest.raises(RuntimeError, match="boom"):
        _index(cfg, library, embedder=Flaky)
    conn = _db(cfg)
    done = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
    assert 0 < done < 4
    _assert_consistent(conn)
    conn.close()

    stats = _index(cfg, library)
    assert stats.unchanged == done
    assert stats.total_documents == 4


def test_failed_full_rebuild_keeps_the_old_index(cfg: Config, library: Path) -> None:
    _index(cfg, library)

    class Broken(FakeEmbedder):
        def embed(self, texts: Sequence[str]) -> Vectors:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        _index(cfg, library, full=True, embedder=Broken)
    assert not Path(f"{cfg.paths.index_db}.tmp").exists()
    assert _db(cfg).execute("SELECT count(*) FROM documents").fetchone()[0] == 4


def test_embedder_is_released(cfg: Config, library: Path) -> None:
    created: list[FakeEmbedder] = []

    def factory() -> FakeEmbedder:
        created.append(FakeEmbedder())
        return created[-1]

    _index(cfg, library, embedder=factory)
    assert len(created) == 1 and created[0].closed


# --- GPU / LLM server guard (ADR-012) ---------------------------------------------------------


def test_gpu_indexing_refused_while_llm_server_listens(cfg: Config) -> None:
    with socket.create_server(("127.0.0.1", 0)) as server:
        port = server.getsockname()[1]
        running = replace(cfg, llm=replace(cfg.llm, base_url=f"http://127.0.0.1:{port}/v1"))
        with pytest.raises(LLMServerRunningError):
            ensure_gpu_free(running, DeviceChoice("cuda", "float32"))
        ensure_gpu_free(running, DeviceChoice("cpu", "float32"))  # CPU indexing is fine


def test_gpu_indexing_allowed_when_port_is_closed(cfg: Config) -> None:
    with socket.create_server(("127.0.0.1", 0)) as server:
        port = server.getsockname()[1]
    stopped = replace(cfg, llm=replace(cfg.llm, base_url=f"http://127.0.0.1:{port}/v1"))
    ensure_gpu_free(stopped, DeviceChoice("cuda", "float32"))


# --- CLI ----------------------------------------------------------------------------------------


def test_cli_index_incremental_and_mismatch(
    cfg: Config, library: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    import biblio.cli as cli
    import biblio.index.embedder as embedder_mod
    import biblio.ingest.tokens as tokens_mod

    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(embedder_mod, "load_index_embedder", lambda config, choice: FakeEmbedder())
    monkeypatch.setattr(
        embedder_mod,
        "resolve_index_device",
        lambda config, override: DeviceChoice("cpu", "float32"),
    )
    monkeypatch.setattr(tokens_mod, "load_token_counter", lambda config: WordCounter())
    runner = CliRunner()

    first = runner.invoke(cli.app, ["index", str(library)])
    assert first.exit_code == 0, first.output
    assert "4 added" in first.output

    second = runner.invoke(cli.app, ["index", str(library)])
    assert second.exit_code == 0, second.output
    assert "4 unchanged" in second.output

    changed = replace(cfg, chunking=replace(cfg.chunking, max_tokens=70))
    monkeypatch.setattr(cli, "load_config", lambda: changed)
    third = runner.invoke(cli.app, ["index", str(library)])
    assert third.exit_code == 2
    assert "--full" in third.output


def test_cli_index_requires_a_library(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    import biblio.cli as cli

    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    result = CliRunner().invoke(cli.app, ["index"])
    assert result.exit_code == 2
    assert "No library path" in result.output


def test_concurrent_run_is_refused_and_stale_lock_is_taken_over(cfg: Config, library: Path) -> None:
    import os

    from biblio.index.indexer import IndexLockedError

    lock = cfg.paths.index_db.with_name(cfg.paths.index_db.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(str(os.getpid()), encoding="utf-8")  # a live process holds it
    with pytest.raises(IndexLockedError, match="running"):
        _index(cfg, library)

    lock.write_text("999999999", encoding="utf-8")  # no such process: stale
    assert _index(cfg, library).added == 4
    assert not lock.exists()
