"""Incremental indexing: hash every file, re-chunk and re-embed only what changed.

Each document is written in its own transaction, so an interrupted run leaves a consistent index.
`full=True` builds a new index next to the old one and swaps it in only when it succeeds.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import psutil

from biblio.config import Config
from biblio.index.embedder import DeviceChoice, Embedder, Vectors
from biblio.index.schema import (
    SCHEMA_VERSION,
    connect,
    create_schema,
    has_schema,
    read_meta,
    write_meta,
)
from biblio.ingest.chunker import CHUNKER_VERSION, chunk_document
from biblio.ingest.discovery import discover
from biblio.ingest.parser import parse_bytes
from biblio.ingest.tokens import TokenCounter
from biblio.ingest.types import Chunk, ParsedDocument
from biblio.offline import loopback_port_open

log = logging.getLogger(__name__)

EmbedderFactory = Callable[[], Embedder]
CounterFactory = Callable[[], TokenCounter]
Progress = Callable[[int], None]  # called once per file with the number of chunks written


class IndexMismatchError(RuntimeError):
    """The existing index was built with a different model, chunker or library (rule 6)."""


class IndexLockedError(RuntimeError):
    """Another indexing run holds the index."""


class LLMServerRunningError(RuntimeError):
    """GPU indexing was requested while the LLM server holds the GPU (ADR-012)."""


@dataclass(frozen=True)
class IndexStats:
    full: bool
    added: int
    updated: int
    deleted: int
    unchanged: int
    failed: tuple[str, ...]
    chunks_written: int
    total_documents: int
    total_chunks: int
    seconds: float
    embed_seconds: float


@dataclass
class _Pending:
    path: str
    sha256: str
    doc: ParsedDocument
    chunks: list[Chunk]
    existing_id: int | None


@dataclass
class _Run:
    added: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0
    failed: list[str] = field(default_factory=list)
    chunks_written: int = 0
    embed_seconds: float = 0.0


def expected_meta(config: Config, library: Path, dtype: str) -> dict[str, str]:
    """Everything that changes the stored vectors or chunks; a difference requires `--full`."""
    c = config.chunking
    return {
        "schema_version": str(SCHEMA_VERSION),
        "embedding_model": config.embedding.model,
        "embedding_revision": config.embedding.revision,
        "index_dtype": dtype,
        "max_seq_length": str(config.embedding.max_seq_length),
        "chunker_version": str(CHUNKER_VERSION),
        "chunking": json.dumps(
            {"min": c.min_tokens, "target": c.target_tokens, "max": c.max_tokens}, sort_keys=True
        ),
        "library": library.resolve().as_posix(),
    }


def meta_differences(stored: dict[str, str], expected: dict[str, str]) -> list[str]:
    return [
        f"{key}: index has {stored.get(key)!r}, config needs {value!r}"
        for key, value in expected.items()
        if stored.get(key) != value
    ]


def ensure_gpu_free(config: Config, choice: DeviceChoice) -> None:
    """Refuse GPU indexing while the LLM server answers on its loopback port (ADR-012)."""
    if choice.device.startswith("cuda") and loopback_port_open(
        config.llm.base_url, config.llm.probe_timeout_s
    ):
        raise LLMServerRunningError(
            f"An LLM server is answering on {config.llm.base_url}. Stop it before indexing on the "
            "GPU, index on CPU with --device cpu, or pass --ignore-llm."
        )


def index_library(
    config: Config,
    library: Path,
    *,
    full: bool,
    dtype: str,
    embedder_factory: EmbedderFactory,
    counter_factory: CounterFactory,
    progress: Progress | None = None,
    on_start: Callable[[int], None] | None = None,
) -> IndexStats:
    """Index `library` into `paths.index_db`; only one run at a time (lock file next to the DB)."""
    with _index_lock(config.paths.index_db):
        return _index_library(
            config,
            library,
            full=full,
            dtype=dtype,
            embedder_factory=embedder_factory,
            counter_factory=counter_factory,
            progress=progress,
            on_start=on_start,
        )


@contextmanager
def _index_lock(db_path: Path) -> Iterator[None]:
    """Exclusive lock holding the owner's PID; a lock left by a dead process is taken over."""
    lock = db_path.with_name(db_path.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                owner = int(lock.read_text(encoding="utf-8").strip() or 0)
            except (OSError, ValueError):
                owner = 0
            if owner and psutil.pid_exists(owner):
                raise IndexLockedError(
                    f"Another `biblio index` (PID {owner}) is running; wait for it to finish."
                ) from None
            lock.unlink(missing_ok=True)  # stale: its process is gone
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(str(os.getpid()))
        break
    else:
        raise IndexLockedError(f"Could not acquire {lock}")
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def _index_library(
    config: Config,
    library: Path,
    *,
    full: bool,
    dtype: str,
    embedder_factory: EmbedderFactory,
    counter_factory: CounterFactory,
    progress: Progress | None,
    on_start: Callable[[int], None] | None,
) -> IndexStats:
    start = time.perf_counter()
    db_path = config.paths.index_db
    target = _tmp_path(db_path) if full else db_path
    if full:
        _remove_db(target)

    expected = expected_meta(config, library, dtype)
    run = _Run()
    lazy = _Lazy(embedder_factory, counter_factory)

    conn = connect(target)
    try:
        if full or not has_schema(conn):
            dim = lazy.embedder().dimension
            create_schema(conn, dim)
            write_meta(conn, {**expected, "embedding_dim": str(dim), "created_at": _now()})
        else:
            stored_meta = read_meta(conn)
            differences = meta_differences(stored_meta, expected)
            if differences:
                raise IndexMismatchError(
                    "The index was built with a different configuration; "
                    "run `biblio index --full`.\n  " + "\n  ".join(differences)
                )

        stored = {
            path: (doc_id, sha)
            for doc_id, path, sha in conn.execute("SELECT id, path, sha256 FROM documents")
        }
        paths = discover(library, config.paths.include, config.paths.exclude)
        if on_start:
            on_start(len(paths))

        for path in sorted(set(stored) - set(paths)):
            _delete_document(conn, stored[path][0])
            run.deleted += 1

        pending: list[_Pending] = []
        pending_chunks = 0
        for path in paths:
            raw = (library / path).read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            existing = stored.get(path)
            if existing is not None and existing[1] == sha:
                run.unchanged += 1
                if progress:
                    progress(0)
                continue
            try:
                doc = parse_bytes(raw, path)
                chunks = chunk_document(doc, lazy.counter(), config.chunking)
            except Exception:
                log.exception("%s: could not be chunked; skipped (it will be retried)", path)
                run.failed.append(path)
                if progress:
                    progress(0)
                continue
            pending.append(_Pending(path, sha, doc, chunks, existing[0] if existing else None))
            pending_chunks += len(chunks)
            if pending_chunks >= config.embedding.index_flush_chunks:
                _flush(conn, pending, lazy, run, progress)
                pending, pending_chunks = [], 0
        _flush(conn, pending, lazy, run, progress)

        total_docs = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
        total_chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        write_meta(conn, {"updated_at": _now()})
    except BaseException:
        conn.close()
        if full:
            _remove_db(target)
        raise
    finally:
        lazy.close()

    conn.close()
    if full:
        _remove_db(db_path)
        os.replace(target, db_path)

    return IndexStats(
        full=full,
        added=run.added,
        updated=run.updated,
        deleted=run.deleted,
        unchanged=run.unchanged,
        failed=tuple(run.failed),
        chunks_written=run.chunks_written,
        total_documents=total_docs,
        total_chunks=total_chunks,
        seconds=time.perf_counter() - start,
        embed_seconds=run.embed_seconds,
    )


class _Lazy:
    """Loads the embedder and the tokenizer only when a file actually changed."""

    def __init__(self, embedder_factory: EmbedderFactory, counter_factory: CounterFactory) -> None:
        self._embedder_factory = embedder_factory
        self._counter_factory = counter_factory
        self._embedder: Embedder | None = None
        self._counter: TokenCounter | None = None

    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = self._embedder_factory()
        return self._embedder

    def counter(self) -> TokenCounter:
        if self._counter is None:
            self._counter = self._counter_factory()
        return self._counter

    def close(self) -> None:
        if self._embedder is not None:
            self._embedder.close()
            self._embedder = None


def _flush(
    conn: sqlite3.Connection,
    pending: list[_Pending],
    lazy: _Lazy,
    run: _Run,
    progress: Progress | None,
) -> None:
    if not pending:
        return
    texts = [c.embedding_text for p in pending for c in p.chunks]
    vectors: Vectors = np.empty((0, 0), dtype=np.float32)
    if texts:
        model = lazy.embedder()
        _check_dimension(conn, model)
        t0 = time.perf_counter()
        vectors = model.embed(texts)
        run.embed_seconds += time.perf_counter() - t0
    offset = 0
    for item in pending:
        n = len(item.chunks)
        doc_vectors = vectors[offset : offset + n]
        offset += n
        _write_document(conn, item, doc_vectors)
        if item.existing_id is None:
            run.added += 1
        else:
            run.updated += 1
        run.chunks_written += n
        if progress:
            progress(n)


def _check_dimension(conn: sqlite3.Connection, model: Embedder) -> None:
    stored = read_meta(conn).get("embedding_dim")
    if stored is not None and int(stored) != model.dimension:
        raise IndexMismatchError(
            f"Index vectors have dimension {stored}, the embedder produces {model.dimension}; "
            "run `biblio index --full`."
        )


def _write_document(conn: sqlite3.Connection, item: _Pending, vectors: Vectors) -> None:
    with conn:
        if item.existing_id is not None:
            _delete_chunks(conn, item.existing_id)
            conn.execute(
                "UPDATE documents SET sha256 = ?, title = ?, metadata = ?, indexed_at = ? "
                "WHERE id = ?",
                (item.sha256, item.doc.title, _json(item.doc.metadata), _now(), item.existing_id),
            )
            doc_id = item.existing_id
        else:
            cur = conn.execute(
                "INSERT INTO documents(path, sha256, title, metadata, indexed_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (item.path, item.sha256, item.doc.title, _json(item.doc.metadata), _now()),
            )
            doc_id = int(cur.lastrowid or 0)
        for chunk, vector in zip(item.chunks, vectors, strict=True):
            cur = conn.execute(
                "INSERT INTO chunks(document_id, ordinal, heading_path, breadcrumb, text, "
                "start_line, end_line, token_count, links) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doc_id,
                    chunk.ordinal,
                    _json(list(chunk.heading_path)),
                    chunk.breadcrumb,
                    chunk.text,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.token_count,
                    _json(list(chunk.links)),
                ),
            )
            conn.execute(
                "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
                (cur.lastrowid, vector.tobytes()),
            )


def _delete_chunks(conn: sqlite3.Connection, doc_id: int) -> None:
    ids = [row[0] for row in conn.execute("SELECT id FROM chunks WHERE document_id = ?", (doc_id,))]
    conn.executemany("DELETE FROM vec_chunks WHERE chunk_id = ?", [(i,) for i in ids])
    conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))


def _delete_document(conn: sqlite3.Connection, doc_id: int) -> None:
    with conn:
        _delete_chunks(conn, doc_id)
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _tmp_path(db_path: Path) -> Path:
    return db_path.with_name(db_path.name + ".tmp")


def _remove_db(path: Path) -> None:
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        candidate.unlink(missing_ok=True)
