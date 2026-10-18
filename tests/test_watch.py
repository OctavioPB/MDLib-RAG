from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from biblio.config import Config, ConfigError, WatchConfig, _validate
from biblio.index.embedder import DeviceChoice
from biblio.index.indexer import IndexLockedError, IndexMismatchError, IndexStats, index_library
from biblio.index.watch import (
    Change,
    Snapshot,
    Watcher,
    WatchEvent,
    choose_device,
    diff,
    snapshot,
)

from fakes import FakeEmbedder, WordCounter

SETTINGS = WatchConfig(interval_s=30, settle_s=10)
CPU = DeviceChoice("cpu", "float32")


def _stats(added: int = 0, updated: int = 0) -> IndexStats:
    return IndexStats(False, added, updated, 0, 0, (), 0, 0, 0, 0.0, 0.0)


class _Scans:
    """Returns the queued snapshots one per scan; the last one repeats."""

    def __init__(self, *snaps: Snapshot) -> None:
        self.snaps = list(snaps)
        self.count = 0

    def __call__(self) -> Snapshot:
        self.count += 1
        return self.snaps.pop(0) if len(self.snaps) > 1 else self.snaps[0]


def _watcher(
    scans: _Scans, results: Iterator[tuple[DeviceChoice | None, IndexStats | None]] | None = None
) -> tuple[Watcher, list[WatchEvent], list[float]]:
    events: list[WatchEvent] = []
    sleeps: list[float] = []
    queue = results or iter(lambda: (CPU, _stats()), None)
    watcher = Watcher(SETTINGS, scans, lambda: next(queue), events.append, sleeps.append)
    return watcher, events, sleeps


A: Snapshot = {"a.md": (1, 10)}
B: Snapshot = {"a.md": (2, 11)}
C: Snapshot = {"a.md": (3, 12), "b.md": (1, 5)}


# --- pure helpers -------------------------------------------------------------------------------


def test_diff_counts_added_modified_removed() -> None:
    old = {"a.md": (1, 1), "b.md": (1, 1), "c.md": (1, 1)}
    new = {"a.md": (1, 1), "b.md": (2, 1), "d.md": (1, 1)}
    assert diff(old, new) == Change(added=1, modified=1, removed=1)


def test_choose_device_gives_the_gpu_to_the_llm() -> None:
    gpu = DeviceChoice("cuda", "float32")
    assert choose_device(gpu, llm_running=False) == gpu
    assert choose_device(gpu, llm_running=True) == CPU
    assert choose_device(CPU, llm_running=True) == CPU
    assert choose_device(DeviceChoice("cuda", "float16"), llm_running=True) is None


def test_snapshot_follows_discovery_rules(config: Config, tmp_path: Path) -> None:
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "nota.md").write_text("hola", encoding="utf-8")
    (tmp_path / "imagen.png").write_bytes(b"png")
    snap = snapshot(config, tmp_path)
    assert list(snap) == ["x/nota.md"] and snap["x/nota.md"][1] == 4


def test_watch_settings_are_validated(config: Config) -> None:
    with pytest.raises(ConfigError, match="watch"):
        _validate(replace(config, watch=WatchConfig(interval_s=0, settle_s=10)))


# --- watcher ------------------------------------------------------------------------------------


def test_catch_up_then_quiet_polls_do_not_reindex() -> None:
    scans = _Scans(A)
    watcher, events, sleeps = _watcher(scans)
    watcher.catch_up()
    watcher.poll()
    watcher.poll()
    assert [e.kind for e in events] == ["indexed"] and events[0].change is None
    assert sleeps == []  # nothing changed, no settle wait


def test_change_is_indexed_after_it_settles() -> None:
    scans = _Scans(A, B, C, C)  # catch-up A; poll sees B, still moving to C, then C is stable
    watcher, events, sleeps = _watcher(scans)
    watcher.catch_up()
    watcher.poll()
    assert sleeps == [10, 10]
    assert [e.kind for e in events] == ["indexed", "indexed"]
    assert events[1].change == Change(added=1, modified=1, removed=0)
    watcher.poll()  # C is now the indexed state
    assert len(events) == 2


def test_change_undone_while_settling_is_not_indexed() -> None:
    watcher, events, _ = _watcher(_Scans(A, B, A))
    watcher.catch_up()
    watcher.poll()
    assert len(events) == 1


def test_locked_index_is_retried_on_the_next_poll() -> None:
    calls = 0

    def reindex() -> tuple[DeviceChoice | None, IndexStats | None]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise IndexLockedError("Another `biblio index` (PID 1) is running.")
        return CPU, _stats(updated=calls - 1)

    events: list[WatchEvent] = []
    watcher = Watcher(SETTINGS, _Scans(A, B), reindex, events.append, lambda s: None)
    watcher.catch_up()
    watcher.poll()  # locked
    watcher.poll()  # B still differs from the indexed A: retried
    watcher.poll()  # nothing new
    assert [e.kind for e in events] == ["indexed", "locked", "indexed"]
    assert events[2].stats is not None and events[2].stats.updated == 2


def test_deferred_update_stays_pending() -> None:
    results = iter([(None, None), (CPU, _stats(updated=1))])
    watcher, events, _ = _watcher(_Scans(A), results)
    watcher.catch_up()
    watcher.poll()
    assert [e.kind for e in events] == ["deferred", "indexed"]


def test_version_mismatch_stops_the_watcher() -> None:
    def reindex() -> tuple[DeviceChoice | None, IndexStats | None]:
        raise IndexMismatchError("chunker_version differs; run `biblio index --full`")

    watcher = Watcher(SETTINGS, _Scans(A), reindex, lambda e: None, lambda s: None)
    with pytest.raises(IndexMismatchError):
        watcher.catch_up()


# --- end to end with the real indexer -----------------------------------------------------------


def test_edit_reaches_the_index(config: Config, tmp_path: Path) -> None:
    lib = tmp_path / "library"
    lib.mkdir()
    note = lib / "nota.md"
    note.write_text("# Nota\n\n" + "Texto original sobre regresión lineal. " * 5, encoding="utf-8")
    cfg = replace(
        config,
        paths=replace(config.paths, index_db=tmp_path / "data" / "index.db"),
        chunking=replace(config.chunking, min_tokens=5, target_tokens=40, max_tokens=60),
    )

    def reindex() -> tuple[DeviceChoice | None, IndexStats | None]:
        stats = index_library(
            cfg,
            lib,
            full=False,
            dtype="float32",
            embedder_factory=FakeEmbedder,
            counter_factory=WordCounter,
        )
        return CPU, stats

    events: list[WatchEvent] = []
    watcher = Watcher(SETTINGS, lambda: snapshot(cfg, lib), reindex, events.append, lambda s: None)
    watcher.catch_up()
    note.write_text("# Nota\n\n" + "Texto editado sobre teorema de Bayes. " * 5, encoding="utf-8")
    st = note.stat()
    os.utime(note, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))  # coarse clocks
    watcher.poll()
    assert [e.kind for e in events] == ["indexed", "indexed"]
    assert events[0].stats is not None and events[0].stats.added == 1
    assert events[1].stats is not None and events[1].stats.updated == 1
