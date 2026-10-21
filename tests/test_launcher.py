from __future__ import annotations

import socket
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from biblio.config import PROJECT_ROOT, Config
from biblio.launcher import (
    SERVE_LLM_SCRIPT,
    find_ollama,
    llm_server_env,
    looks_like_library,
    set_local_library,
    wait_for_port,
)

# --- library path in config/local.toml ----------------------------------------------------------


def test_library_is_added_to_a_new_or_unrelated_local_toml(tmp_path: Path) -> None:
    lib = tmp_path / "Mis notas"
    lib.mkdir()
    local = tmp_path / "config" / "local.toml"
    set_local_library(local, lib)
    assert tomllib.loads(local.read_text(encoding="utf-8"))["paths"]["library"] == lib.as_posix()

    local.write_text('[llm]\nmodel = "x"\n', encoding="utf-8")
    set_local_library(local, lib)
    data = tomllib.loads(local.read_text(encoding="utf-8"))
    assert data["llm"]["model"] == "x" and data["paths"]["library"] == lib.as_posix()


def test_library_replaces_the_old_value_and_keeps_the_rest(tmp_path: Path) -> None:
    local = tmp_path / "local.toml"
    local.write_text(
        '[paths]\nlibrary = "C:/viejo"\nindex_db = "data/x.db"\n\n[web]\nport = 9000\n',
        encoding="utf-8",
    )
    set_local_library(local, tmp_path)
    data = tomllib.loads(local.read_text(encoding="utf-8"))
    assert data["paths"] == {"library": tmp_path.as_posix(), "index_db": "data/x.db"}
    assert data["web"]["port"] == 9000

    local.write_text('[paths]\nindex_db = "data/x.db"\n', encoding="utf-8")
    set_local_library(local, tmp_path)
    data = tomllib.loads(local.read_text(encoding="utf-8"))
    assert data["paths"] == {"library": tmp_path.as_posix(), "index_db": "data/x.db"}


def test_looks_like_library(tmp_path: Path) -> None:
    assert not looks_like_library(tmp_path / "no-existe")
    assert not looks_like_library(tmp_path)
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "nota.md").write_text("# x", encoding="utf-8")
    assert looks_like_library(tmp_path)


# --- the LLM server -----------------------------------------------------------------------------


def test_llm_settings_come_from_the_serve_script() -> None:
    env = llm_server_env(PROJECT_ROOT / SERVE_LLM_SCRIPT)
    assert env == {
        "OLLAMA_HOST": "127.0.0.1:11434",
        "OLLAMA_FLASH_ATTENTION": "1",
        "OLLAMA_KV_CACHE_TYPE": "q8_0",
        "OLLAMA_CONTEXT_LENGTH": "8192",
    }


def test_find_ollama_prefers_the_configured_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = tmp_path / "ollama.exe"
    exe.write_bytes(b"")
    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "nada"))
    assert find_ollama(str(exe)) == exe
    assert find_ollama("") is None


def test_wait_for_port() -> None:
    with socket.create_server(("127.0.0.1", 0)) as server:
        url = f"http://127.0.0.1:{server.getsockname()[1]}/v1"
        assert wait_for_port(url, 2, 0.2)
    # closed port and a dead process: gives up at once instead of waiting for the timeout
    assert not wait_for_port(url, 30, 0.2, alive=lambda: False, sleep=lambda _s: None)


# --- biblio start -------------------------------------------------------------------------------


def _cli_with(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    import biblio.cli as cli

    monkeypatch.setattr(cli, "load_config", lambda *a, **k: cfg)


def test_start_opens_the_browser_when_already_running(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    import webbrowser

    import biblio.cli as cli

    opened: list[str] = []
    monkeypatch.setattr(webbrowser, "open", opened.append)
    with socket.create_server(("127.0.0.1", 0)) as server:
        port = server.getsockname()[1]
        _cli_with(replace(config, web=replace(config.web, port=port)), monkeypatch)
        result = CliRunner().invoke(cli.app, ["start"])
    assert result.exit_code == 0, result.output
    assert "already running" in result.output and opened == [f"http://127.0.0.1:{port}"]


def test_start_never_downloads_without_consent(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import biblio.cli as cli
    import biblio.models as models

    def no_download(*_a: object, **_k: object) -> None:
        raise AssertionError("downloaded without asking")

    monkeypatch.setattr(models, "pull_model", no_download)
    cfg = replace(
        config,
        paths=replace(config.paths, models_dir=tmp_path / "models"),
        web=replace(config.web, port=1),  # nothing listens there
    )
    _cli_with(cfg, monkeypatch)
    result = CliRunner().invoke(cli.app, ["start"])  # no terminal: questions are declined
    assert result.exit_code == 2, result.output
    assert "Not asked" in result.output and "models pull" in result.output


# --- switching the LLM off and on from «Ajustes» (ADR-043) ---------------------------------------


class _Process:
    def __init__(self) -> None:
        self.running = True

    def poll(self) -> int | None:
        return None if self.running else 0


class _Server:
    def __init__(self) -> None:
        self.process = _Process()
        self.stops = 0

    def stop(self) -> None:
        self.stops += 1
        self.process.running = False


def test_the_switch_stops_only_a_server_the_app_started(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    import biblio.launcher as launcher

    unloaded: list[str] = []
    monkeypatch.setattr(
        launcher, "unload_model", lambda url, model, timeout: unloaded.append(model) or True
    )
    server = _Server()
    switch = launcher.LLMSwitch(config, server)  # type: ignore[arg-type]
    assert switch.owned
    assert switch.stop() == "stopped" and server.stops == 1 and not switch.owned
    assert unloaded == []
    # an Ollama opened outside the app is never killed: its model leaves memory instead
    assert switch.stop() == "unloaded" and unloaded == [config.llm.model]
    monkeypatch.setattr(launcher, "unload_model", lambda url, model, timeout: False)
    with pytest.raises(RuntimeError):
        switch.stop()


def test_closing_the_app_stops_only_its_own_server(config: Config) -> None:
    from biblio.launcher import LLMSwitch

    mine = _Server()
    LLMSwitch(config, mine).close()  # type: ignore[arg-type]
    assert mine.stops == 1
    LLMSwitch(config).close()  # nothing started here: nothing to stop


def test_starting_without_ollama_says_so(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    import biblio.launcher as launcher

    monkeypatch.setattr(launcher, "find_ollama", lambda configured: None)
    switch = launcher.LLMSwitch(config)
    assert not switch.can_start()
    with pytest.raises(RuntimeError, match="no está instalado"):
        switch.start()
