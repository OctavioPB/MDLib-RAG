"""Where the library is, and changing it (ADR-039).

A new folder is compared with the index note by note (path relative to the root, SHA-256 of the
file): when most notes match it is the same library moved, and the index is updated in place;
otherwise it is another library and gets a full index of its own, which replaces the current one
only if it completes.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import string
from dataclasses import dataclass
from pathlib import Path

from biblio.config import Config
from biblio.ingest.discovery import discover
from biblio.launcher import set_local_library

SAME = "same"  # already the library
RELOCATE = "relocate"  # the same library moved: the index is reused
REBUILD = "rebuild"  # another library: a full index
INVALID = "invalid"  # not a folder with notes


@dataclass(frozen=True)
class LibraryCheck:
    path: str  # as resolved
    plan: str  # same | relocate | rebuild | invalid
    notes: int  # Markdown notes the index would take
    matching: int  # of them, with the same relative path and content as in the index
    indexed: int  # notes in the index now
    minutes_gpu: int  # estimate of a full index (rebuild only)
    minutes_cpu: int
    reason: str = ""  # why it is invalid


def _indexed(conn: sqlite3.Connection | None) -> dict[str, str]:
    if conn is None:
        return {}
    try:
        return dict(conn.execute("SELECT path, sha256 FROM documents").fetchall())
    except sqlite3.Error:
        return {}


def check_library(config: Config, path: Path, conn: sqlite3.Connection | None) -> LibraryCheck:
    """How changing to `path` would go; reads every note once to hash it."""
    cfg = config.library
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        resolved = path
    if not resolved.is_dir():
        return LibraryCheck(str(resolved), INVALID, 0, 0, 0, 0, 0, "No existe o no es una carpeta.")
    notes = discover(resolved, config.paths.include, config.paths.exclude)
    indexed = _indexed(conn)
    if not notes:
        return LibraryCheck(
            str(resolved), INVALID, 0, 0, len(indexed), 0, 0, "No tiene notas Markdown (.md)."
        )
    matching = 0
    for rel in notes:
        sha = indexed.get(rel)
        if sha is not None:
            try:
                if hashlib.sha256((resolved / rel).read_bytes()).hexdigest() == sha:
                    matching += 1
            except OSError:
                pass
    current = config.paths.library
    if current is not None and current.resolve() == resolved:
        plan = SAME
    elif matching >= cfg.relocate_min_overlap * max(len(notes), len(indexed)):
        plan = RELOCATE
    else:
        plan = REBUILD
    thousands = len(notes) / 1000
    return LibraryCheck(
        str(resolved),
        plan,
        len(notes),
        matching,
        len(indexed),
        round(thousands * cfg.minutes_per_1000_gpu) if plan == REBUILD else 0,
        round(thousands * cfg.minutes_per_1000_cpu) if plan == REBUILD else 0,
    )


@dataclass(frozen=True)
class Folder:
    name: str
    path: str


@dataclass(frozen=True)
class Listing:
    path: str  # "" for the list of drives
    parent: str | None  # None at the top
    folders: tuple[Folder, ...]
    truncated: bool


def drives() -> list[str]:
    """The roots to start browsing from: drive letters on Windows, / elsewhere."""
    if os.name != "nt":
        return ["/"]
    return [f"{letter}:\\" for letter in string.ascii_uppercase if Path(f"{letter}:\\").exists()]


def browse(path: str | None, limit: int) -> Listing:
    """The folders inside `path` (hidden ones left out), or the drives when `path` is empty."""
    if not path:
        roots = drives()
        return Listing("", None, tuple(Folder(r, r) for r in roots), False)
    folder = Path(path).expanduser()
    if not folder.is_dir():
        raise ValueError("Esa carpeta no existe.")
    folder = folder.resolve()
    names: list[Folder] = []
    try:
        entries = sorted(os.scandir(folder), key=lambda e: e.name.casefold())
    except OSError as exc:
        raise ValueError(f"No se puede abrir esa carpeta: {exc.strerror or exc}") from exc
    for entry in entries:
        if entry.name.startswith((".", "$")):
            continue
        try:
            if not entry.is_dir():
                continue
        except OSError:
            continue
        names.append(Folder(entry.name, str(folder / entry.name)))
    parent = str(folder.parent) if folder.parent != folder else ""
    return Listing(str(folder), parent, tuple(names[:limit]), len(names) > limit)


def save_library(config: Config, path: Path) -> None:
    """Keep the new location in `config/local.toml` (the rest of the file untouched)."""
    set_local_library(config.root / "config" / "local.toml", path)
