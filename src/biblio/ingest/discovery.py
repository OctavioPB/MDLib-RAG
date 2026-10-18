"""Library file discovery: configurable globs and exclusions, symlinks ignored."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path

_RECURSIVE_SUFFIX = "/**"


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a glob over "/"-separated relative paths: `**/` = zero or more directories,
    `**` = anything, `*` and `?` never cross "/". Matching is case-insensitive."""
    out = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out), re.IGNORECASE)


def discover(root: Path, include: Iterable[str], exclude: Iterable[str]) -> list[str]:
    """Relative "/"-separated paths of the library files to ingest, sorted."""
    includes = [glob_to_regex(p) for p in include]
    excludes = [glob_to_regex(p) for p in exclude]
    # `x/**` exclusions also prune whole directories so they are never walked.
    pruned_dirs = [
        glob_to_regex(p[: -len(_RECURSIVE_SUFFIX)])
        for p in exclude
        if p.endswith(_RECURSIVE_SUFFIX)
    ]

    found = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        prefix = "" if rel_dir == "." else f"{rel_dir}/"
        dirnames[:] = [
            d
            for d in dirnames
            if not os.path.islink(os.path.join(dirpath, d))
            and not any(rx.fullmatch(prefix + d) for rx in pruned_dirs)
        ]
        for name in filenames:
            rel = prefix + name
            if os.path.islink(os.path.join(dirpath, name)):
                continue
            if any(rx.fullmatch(rel) for rx in includes) and not any(
                rx.fullmatch(rel) for rx in excludes
            ):
                found.append(rel)
    return sorted(found)
