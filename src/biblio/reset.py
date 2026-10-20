"""«Reset» (ADR-040): the app's memory moved aside, the library and its index untouched.

The memory is what the app wrote by itself: conversations, maps and diagrams, practice (quizzes,
reviews, «Explícamelo tú», plans, read marks, pending topics, dictionaries) and the drafts folder.
Nothing is deleted: it all goes to `data/backups/reset-<date>/`.
"""

from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from biblio.config import Config

_SQLITE_SIDES = ("", "-wal", "-shm")


@dataclass(frozen=True)
class MemoryItem:
    key: str  # conversations | maps | practice | drafts
    path: Path
    count: int  # conversations, maps, questions or files, for the warning


@dataclass(frozen=True)
class ResetResult:
    backup: Path
    moved: tuple[str, ...]  # file names moved into the backup


def _inside(path: Path, folder: Path | None) -> bool:
    if folder is None:
        return False
    try:
        path.resolve().relative_to(folder.resolve())
    except ValueError:
        return False
    return True


def _count(db: Path, sql: str) -> int:
    if not db.is_file():
        return 0
    try:
        conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
        try:
            return int(conn.execute(sql).fetchone()[0])
        finally:
            conn.close()
    except sqlite3.Error:
        return 0


def memory_items(config: Config) -> list[MemoryItem]:
    """What a reset would move, with counts for the warning."""
    drafts = config.paths.drafts_dir
    return [
        MemoryItem(
            "conversations",
            config.conversations.db,
            _count(config.conversations.db, "SELECT count(*) FROM conversations"),
        ),
        MemoryItem(
            "maps",
            config.maps.history_db,
            _count(config.maps.history_db, "SELECT count(*) FROM maps"),
        ),
        MemoryItem(
            "practice",
            config.practice.db,
            _count(config.practice.db, "SELECT count(*) FROM questions"),
        ),
        MemoryItem(
            "drafts",
            drafts,
            sum(1 for p in drafts.rglob("*") if p.is_file()) if drafts.is_dir() else 0,
        ),
    ]


def check_safe(config: Config, paths: list[Path]) -> None:
    """Refuse anything inside the library, the index, or the models: a reset never touches them,
    whatever the config says."""
    protected = [config.paths.library, config.paths.models_dir]
    index = config.paths.index_db.resolve()
    for path in paths:
        if any(_inside(path, folder) for folder in protected) or any(
            _inside(folder, path) for folder in protected if folder is not None
        ):
            raise ValueError(f"{path} está dentro de la biblioteca o la contiene: no se toca.")
        if path.resolve() == index or _inside(index, path):
            raise ValueError(f"{path} es el índice o lo contiene: no se toca.")


def reset_memory(config: Config, now: datetime | None = None) -> ResetResult:
    """Move the app's memory to `data/backups/reset-<date>/`. The stores must be closed first."""
    items = memory_items(config)
    check_safe(config, [i.path for i in items])
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    backup = config.paths.index_db.parent / "backups" / f"reset-{stamp}"
    moved: list[str] = []
    for item in items:
        sides = (
            [item.path]
            if item.key == "drafts"
            else [item.path.with_name(item.path.name + s) for s in _SQLITE_SIDES]
        )
        for path in sides:
            if not path.exists():
                continue
            backup.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(backup / path.name))
            moved.append(path.name)
    return ResetResult(backup, tuple(moved))
