"""Books in the library (ADR-053): a book's chapters in order, apart from the user's own notes.

A book is a folder inside a folder named `library.books_folder` («Books»): `Area/Books/<Book>/…`.
Its chapters are the notes under that folder, in the natural order of their file names
(`02-preface.md`, `03-1-machine-learning.md`, …, `14-12-object-detection.md`). A note named like
the folder is the book's cover and not a chapter; the title is the folder's name (some covers'
headings are «A» or «FPO»), or the cover's title when the folder is a slug. A note directly inside
`Books/` is a book of one chapter.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True)
class Chapter:
    path: str
    title: str


@dataclass(frozen=True)
class Book:
    id: str  # the book's folder (or its single note), relative to the library
    title: str
    area: str  # the library's top folder: «AI Engineering», «Math»…
    cover: str | None  # the note named like the folder, if any
    chapters: tuple[Chapter, ...]


def book_id(path: str, folder: str) -> str | None:
    """The book a note belongs to, or None for the user's own notes."""
    parts = path.split("/")
    if folder not in parts[:-1]:
        return None
    i = parts.index(folder)
    return "/".join(parts[: i + 2]) if i + 2 < len(parts) else path


def _natural(name: str) -> list[object]:
    return [int(p) if p.isdigit() else p.casefold() for p in re.split(r"(\d+)", name)]


def find_books(notes: Iterable[tuple[str, str]], folder: str) -> list[Book]:
    """Every book among the notes (path, title), its chapters in order; books by title."""
    grouped: dict[str, list[tuple[str, str]]] = {}
    for path, title in notes:
        key = book_id(path, folder)
        if key is not None:
            grouped.setdefault(key, []).append((path, title))
    books = []
    for key, members in grouped.items():
        name = PurePosixPath(key).name
        cover = next((p for p, _ in members if PurePosixPath(p).stem == name), None)
        chapters = sorted(
            (Chapter(p, t) for p, t in members if p != cover),
            key=lambda c: _natural(c.path),
        )
        cover_title = next((t for p, t in members if p == cover), None)
        if key.endswith(".md"):
            title = chapters[0].title
        elif " " not in name and "-" in name and cover_title:  # a slug: the cover reads better
            title = cover_title
        elif " " not in name and "-" in name:  # a slug and no cover: «machine-learning-for-dummies»
            words = name.replace("-", " ").replace("_", " ").split()
            title = " ".join(words)[:1].upper() + " ".join(words)[1:]
        else:  # the folder, as the user named it: some covers' titles are «A» or «FPO»
            title = name
        books.append(Book(key, title, key.split("/")[0], cover, tuple(chapters)))
    return sorted(books, key=lambda b: b.title.casefold())
