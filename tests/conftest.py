"""Global test configuration.

The network is blocked by `pytest-socket` (see addopts in pyproject.toml); only localhost is
allowed.
A test that needs the network is a design bug (ADR-002).
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from biblio.config import PROJECT_ROOT, Config, load_config


@pytest.fixture
def project_copy(tmp_path: Path) -> Path:
    """A throwaway project root containing a copy of the real default config."""
    (tmp_path / "config").mkdir()
    shutil.copy(PROJECT_ROOT / "config" / "default.toml", tmp_path / "config" / "default.toml")
    return tmp_path


@pytest.fixture
def config(project_copy: Path) -> Config:
    """The default config, with the conversation planner off (ADR-038): the fake LLMs of the
    answer tests do not speak its JSON. `tests/test_thread.py` turns it on."""
    loaded = load_config(project_copy)
    return replace(loaded, thread=replace(loaded.thread, plan=False))
