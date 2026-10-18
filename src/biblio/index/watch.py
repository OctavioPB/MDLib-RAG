"""Automatic reindexing by polling file metadata (ADR-018).

Every `watch.interval_s` the library is snapshotted as {path: (mtime_ns, size)}. A change is
indexed once two snapshots `watch.settle_s` apart agree, so a note is not reindexed while
Obsidian is still autosaving it. Indexing is the usual incremental run: content hashes still
decide what is re-embedded, and the same lock and version guards apply.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from biblio.config import Config, WatchConfig
from biblio.index.embedder import DeviceChoice
from biblio.index.indexer import IndexLockedError, IndexStats
from biblio.ingest.discovery import discover

log = logging.getLogger(__name__)

Snapshot = dict[str, tuple[int, int]]  # relative path -> (mtime_ns, size)


@dataclass(frozen=True)
class Change:
    added: int
    modified: int
    removed: int

    def __str__(self) -> str:
        return f"{self.added} added, {self.modified} modified, {self.removed} removed"


@dataclass(frozen=True)
class WatchEvent:
    kind: Literal["indexed", "deferred", "locked"]
    change: Change | None  # None for the catch-up run at start
    device: DeviceChoice | None = None
    stats: IndexStats | None = None
    detail: str = ""


def snapshot(config: Config, library: Path) -> Snapshot:
    """Modification time and size of every library file the indexer would read."""
    snap: Snapshot = {}
    for rel in discover(library, config.paths.include, config.paths.exclude):
        try:
            st = os.stat(library / rel)
        except OSError:
            continue  # removed (or moved by the sync client) between listing and stat
        snap[rel] = (st.st_mtime_ns, st.st_size)
    return snap


def diff(old: Snapshot, new: Snapshot) -> Change:
    return Change(
        added=sum(1 for p in new if p not in old),
        modified=sum(1 for p, v in new.items() if p in old and old[p] != v),
        removed=sum(1 for p in old if p not in new),
    )


def choose_device(preferred: DeviceChoice, llm_running: bool) -> DeviceChoice | None:
    """GPU when free; CPU while the LLM server holds it (ADR-018); None = defer the update.

    A float16 index cannot be updated on CPU: its vectors would not match the stored ones.
    """
    if not (preferred.device.startswith("cuda") and llm_running):
        return preferred
    if preferred.dtype != "float32":
        return None
    return DeviceChoice("cpu", "float32")


Reindex = Callable[[], tuple[DeviceChoice | None, IndexStats | None]]


class Watcher:
    """Polls the library and runs `reindex` after a settled change.

    `reindex` returns the device used and the stats, or (None, None) when the update had to be
    deferred. `IndexLockedError` (another `biblio index` running) is retried on the next change
    check; any other error, such as a version mismatch, stops the watcher.
    """

    def __init__(
        self,
        settings: WatchConfig,
        scan: Callable[[], Snapshot],
        reindex: Reindex,
        report: Callable[[WatchEvent], None],
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._scan = scan
        self._reindex = reindex
        self._report = report
        self._sleep = sleep
        self._indexed: Snapshot | None = None  # state of the library at the last successful run

    def catch_up(self) -> None:
        """Index whatever changed while nobody was watching."""
        self._index(self._scan(), None)

    def poll(self) -> None:
        """One check: if the library differs from the last indexed state, settle and index."""
        current = self._scan()
        if current == self._indexed:
            return
        while True:
            self._sleep(self._settings.settle_s)
            later = self._scan()
            if later == current:
                break
            current = later
        if current != self._indexed:  # edits may have been undone while settling
            self._index(current, diff(self._indexed or {}, current))

    def run_forever(self) -> None:
        self.catch_up()
        while True:
            self._sleep(self._settings.interval_s)
            self.poll()

    def _index(self, snap: Snapshot, change: Change | None) -> None:
        try:
            device, stats = self._reindex()
        except IndexLockedError as exc:
            self._report(WatchEvent("locked", change, detail=str(exc)))
            return
        if stats is None:
            self._report(WatchEvent("deferred", change, device))
            return
        # The snapshot taken before the run: edits made during it are picked up next time.
        self._indexed = snap
        self._report(WatchEvent("indexed", change, device, stats))
