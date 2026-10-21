"""Read access to the index for retrieval: open with guards, turn ranked chunk ids into `Hit`s."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence

from biblio.config import Config
from biblio.index.schema import connect, has_schema, read_meta
from biblio.retrieve.types import Hit


class IndexNotFoundError(RuntimeError):
    """There is no index yet."""


class QueryEmbedderMismatchError(RuntimeError):
    """The query embedder differs from the one that built the index (CLAUDE.md, rule 6)."""


def open_index(config: Config, *, shared_across_threads: bool = False) -> sqlite3.Connection:
    path = config.paths.index_db
    if not path.is_file():
        raise IndexNotFoundError(f"No index at {path}; run `biblio index` first.")
    conn = connect(path, shared_across_threads=shared_across_threads)
    if not has_schema(conn):
        conn.close()
        raise IndexNotFoundError(f"{path} is not a biblio index; run `biblio index --full`.")
    return conn


def check_query_embedder(conn: sqlite3.Connection, config: Config) -> None:
    """Query vectors must come from the same model, revision and sequence length as the index."""
    meta = read_meta(conn)
    wanted = {
        "embedding_model": config.embedding.model,
        "embedding_revision": config.embedding.revision,
        "max_seq_length": str(config.embedding.max_seq_length),
    }
    diffs = [
        f"{k}: index {meta.get(k)!r} vs config {v!r}" for k, v in wanted.items() if meta.get(k) != v
    ]
    if diffs:
        raise QueryEmbedderMismatchError(
            "The index was built with a different embedder; run `biblio index --full`.\n  "
            + "\n  ".join(diffs)
        )


def load_hits(
    conn: sqlite3.Connection, ranked: Sequence[tuple[int, float]], source: str
) -> list[Hit]:
    """`ranked` is [(chunk_id, score)] best first; returns hits in the same order."""
    if not ranked:
        return []
    ids = [chunk_id for chunk_id, _ in ranked]
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        "SELECT c.id, d.path, c.heading_path, c.text, c.start_line, c.end_line "
        f"FROM chunks c JOIN documents d ON d.id = c.document_id WHERE c.id IN ({placeholders})",
        ids,
    ).fetchall()
    by_id = {row[0]: row for row in rows}
    hits = []
    for rank, (chunk_id, score) in enumerate(ranked, start=1):
        row = by_id.get(chunk_id)
        if row is None:  # deleted between the ranking and the lookup
            continue
        _, path, heading_path, text, start, end = row
        hits.append(
            Hit(
                chunk_id=chunk_id,
                doc_path=path,
                heading_path=tuple(json.loads(heading_path)),
                text=text,
                start_line=start,
                end_line=end,
                rank=rank,
                score=score,
                source=source,
            )
        )
    return hits


def scope_prefix(folder: str) -> str:
    """`"Data Science/Syllabus"` → `"Data Science/Syllabus/"`, so "Math" never matches "MathX"."""
    return folder.strip("/") + "/"


def scope_chunk_ids_json(conn: sqlite3.Connection, folder: str) -> str:
    """Ids of every chunk under `folder`, as a JSON array for `json_each` (ADR-019); a path ending
    in `.md` is one note (search inside a note, ADR-045)."""
    if folder.endswith(".md"):
        rows = conn.execute(
            "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id WHERE d.path = ?",
            (folder,),
        )
        return json.dumps([row[0] for row in rows])
    prefix = scope_prefix(folder)
    rows = conn.execute(
        "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id "
        "WHERE substr(d.path, 1, ?) = ?",
        (len(prefix), prefix),
    )
    return json.dumps([row[0] for row in rows])


def library_folders(conn: sqlite3.Connection) -> list[tuple[str, int]]:
    """Every folder that holds indexed documents, at any depth, with its document count."""
    counts: dict[str, int] = {}
    for (path,) in conn.execute("SELECT path FROM documents"):
        parts = path.split("/")[:-1]
        for depth in range(1, len(parts) + 1):
            folder = "/".join(parts[:depth])
            counts[folder] = counts.get(folder, 0) + 1
    return sorted(counts.items())
