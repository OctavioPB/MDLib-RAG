"""SQLite index: relational tables + FTS5 (BM25) + sqlite-vec (dense) in one file (ADR-005)."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from pathlib import Path

# Bump when the table layout changes; a different version requires `biblio index --full`.
SCHEMA_VERSION = 1


def connect(path: Path, *, shared_across_threads: bool = False) -> sqlite3.Connection:
    """Open the index with sqlite-vec loaded and foreign keys on.

    `shared_across_threads` is for the web server, which serializes access with a lock.
    """
    import sqlite_vec

    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=not shared_across_threads)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def create_schema(conn: sqlite3.Connection, embedding_dim: int) -> None:
    with conn:
        conn.executescript(
            """
            CREATE TABLE index_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE documents (
                id         INTEGER PRIMARY KEY,
                path       TEXT NOT NULL UNIQUE,   -- relative to the library root, "/" separator
                sha256     TEXT NOT NULL,
                title      TEXT NOT NULL,
                metadata   TEXT NOT NULL,          -- frontmatter as JSON
                indexed_at TEXT NOT NULL
            );

            CREATE TABLE chunks (
                id           INTEGER PRIMARY KEY,
                document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                ordinal      INTEGER NOT NULL,
                heading_path TEXT NOT NULL,        -- JSON array
                breadcrumb   TEXT NOT NULL,        -- heading_path joined with " › " (FTS column)
                text         TEXT NOT NULL,
                start_line   INTEGER NOT NULL,
                end_line     INTEGER NOT NULL,
                token_count  INTEGER NOT NULL,
                links        TEXT NOT NULL,        -- JSON array
                UNIQUE (document_id, ordinal)
            );

            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                breadcrumb, text,
                content='chunks', content_rowid='id',
                tokenize='unicode61 remove_diacritics 2'
            );

            CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, breadcrumb, text)
                VALUES (new.id, new.breadcrumb, new.text);
            END;

            CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, breadcrumb, text)
                VALUES ('delete', old.id, old.breadcrumb, old.text);
            END;
            """
        )
        conn.execute(
            "CREATE VIRTUAL TABLE vec_chunks USING vec0("
            f"chunk_id INTEGER PRIMARY KEY, embedding float[{int(embedding_dim)}] "
            "distance_metric=cosine)"
        )


def has_schema(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'index_meta'"
    ).fetchone()
    return row is not None


def read_meta(conn: sqlite3.Connection) -> dict[str, str]:
    return dict(conn.execute("SELECT key, value FROM index_meta").fetchall())


def write_meta(conn: sqlite3.Connection, values: Mapping[str, str]) -> None:
    with conn:
        conn.executemany(
            "INSERT INTO index_meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            sorted(values.items()),
        )
