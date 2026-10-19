"""Generated concept maps (ADR-024): `data/maps.db`, apart from the index.

A map is stored as the page receives it (nodes, edges, layout) together with its source fragments,
so reopening it needs neither the LLM nor the index. Search is FTS5 over the topic and the concept
names, with the index's accent folding.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from biblio.conversations import fts_all_words
from biblio.generate.context import Fragment
from biblio.maps.build import ConceptMap, MapEdge, MapNode


@dataclass(frozen=True)
class MapSummary:
    id: int
    topic: str
    created: str
    concepts: int
    snippet: str = ""  # where a search matched


@dataclass(frozen=True)
class StoredMap:
    summary: MapSummary
    map: dict[str, Any]  # as sent to the page (map_json)
    fragments: list[dict[str, Any]]  # as sent to the page (fragment_json)
    positions: dict[str, tuple[int, int]]  # nodes the user moved: id → (x, y)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class MapStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode = WAL")
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS maps (
                    id        INTEGER PRIMARY KEY,
                    topic     TEXT NOT NULL,
                    created   TEXT NOT NULL,
                    concepts  INTEGER NOT NULL,
                    names     TEXT NOT NULL,        -- concept names, for search
                    map       TEXT NOT NULL,        -- JSON, as sent to the page
                    fragments TEXT NOT NULL,        -- JSON, as sent to the page
                    positions TEXT NOT NULL DEFAULT '{}'  -- nodes moved by the user
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS maps_fts USING fts5(
                    topic, names, content='maps', content_rowid='id',
                    tokenize='unicode61 remove_diacritics 2'
                );
                CREATE TRIGGER IF NOT EXISTS maps_ai AFTER INSERT ON maps BEGIN
                    INSERT INTO maps_fts(rowid, topic, names) VALUES (new.id, new.topic, new.names);
                END;
                CREATE TRIGGER IF NOT EXISTS maps_ad AFTER DELETE ON maps BEGIN
                    INSERT INTO maps_fts(maps_fts, rowid, topic, names)
                    VALUES ('delete', old.id, old.topic, old.names);
                END;
                """
            )
            columns = {row[1] for row in self._conn.execute("PRAGMA table_info(maps)")}
            if "positions" not in columns:  # maps.db created before nodes could be moved
                self._conn.execute(
                    "ALTER TABLE maps ADD COLUMN positions TEXT NOT NULL DEFAULT '{}'"
                )

    def close(self) -> None:
        self._conn.close()

    def add(self, topic: str, page_map: dict[str, Any], fragments: list[dict[str, Any]]) -> int:
        concepts = [n["label"] for n in page_map["nodes"] if n["level"] > 0]
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO maps (topic, created, concepts, names, map, fragments) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    topic,
                    _now(),
                    len(concepts),
                    " · ".join(concepts),
                    json.dumps(page_map, ensure_ascii=False),
                    json.dumps(fragments, ensure_ascii=False),
                ),
            )
            return int(cur.lastrowid or 0)

    def set_positions(self, map_id: int, positions: dict[str, tuple[int, int]]) -> bool:
        """Replace the positions of the nodes the user moved; `{}` restores the automatic layout.
        Ids that are not nodes of the map are ignored."""
        with self._lock, self._conn:
            row = self._conn.execute("SELECT map FROM maps WHERE id = ?", (map_id,)).fetchone()
            if row is None:
                return False
            ids = {n["id"] for n in json.loads(row[0])["nodes"]}
            kept = {k: [int(x), int(y)] for k, (x, y) in positions.items() if k in ids}
            self._conn.execute(
                "UPDATE maps SET positions = ? WHERE id = ?", (json.dumps(kept), map_id)
            )
            return True

    def delete(self, map_id: int) -> bool:
        with self._lock, self._conn:
            return self._conn.execute("DELETE FROM maps WHERE id = ?", (map_id,)).rowcount == 1

    def maps(self, limit: int, query: str | None = None) -> list[MapSummary]:
        """Newest first; with `query`, maps whose topic or concepts contain every word."""
        with self._lock:
            if not query or (match := fts_all_words(query)) is None:
                rows = self._conn.execute(
                    "SELECT id, topic, created, concepts FROM maps ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [MapSummary(*row) for row in rows]
            rows = self._conn.execute(
                "SELECT m.id, m.topic, m.created, m.concepts, "
                "snippet(maps_fts, 1, '«', '»', '…', 10) FROM maps_fts "
                "JOIN maps m ON m.id = maps_fts.rowid WHERE maps_fts MATCH ? "
                "ORDER BY m.id DESC LIMIT ?",
                (match, limit),
            ).fetchall()
        return [MapSummary(*row) for row in rows]

    def get(self, map_id: int) -> StoredMap | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, topic, created, concepts, map, fragments, positions FROM maps "
                "WHERE id = ?",
                (map_id,),
            ).fetchone()
        if row is None:
            return None
        positions = {k: (int(v[0]), int(v[1])) for k, v in json.loads(row[6]).items()}
        return StoredMap(MapSummary(*row[:4]), json.loads(row[4]), json.loads(row[5]), positions)


# --- back to domain objects, for saving to drafts/maps ------------------------------------------


def concept_map_from_page(
    page_map: dict[str, Any], positions: dict[str, tuple[int, int]] | None = None
) -> ConceptMap:
    """The stored map as domain objects, with the nodes the user moved where they left them."""
    moved = positions or {}
    nodes = tuple(
        MapNode(
            n["id"],
            n["label"],
            n["level"],
            n["parent"],
            tuple(n["sources"]),
            *moved.get(n["id"], (n["x"], n["y"])),
        )
        for n in page_map["nodes"]
    )
    edges = tuple(
        MapEdge(e["source"], e["target"], e["label"], e["kind"]) for e in page_map["edges"]
    )
    return ConceptMap(page_map["topic"], nodes, edges, page_map.get("omitted", 0))


def fragment_from_page(f: dict[str, Any]) -> Fragment:
    return Fragment(
        number=f["number"],
        doc_path=f["doc_path"],
        heading_path=(f["guide"], *f["sections"]),
        text=f["text"],
        start_line=f["lines"][0],
        end_line=f["lines"][1],
    )
