from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

import biblio.cli as cli
from biblio.config import Config, load_config
from biblio.doctor import Check, HardwareInfo, Status, collect_hardware, run_checks
from biblio.models import ModelSpec, model_specs, pull_model
from biblio.offline import OFFLINE_ENV

FAKE_HW = HardwareInfo(
    os="TestOS",
    cpu="Test CPU",
    physical_cores=4,
    logical_cores=8,
    ram_total_bytes=16 * 1024**3,
    ram_available_bytes=8 * 1024**3,
    gpus=(),
    apple_silicon=False,
)


@pytest.fixture
def offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in OFFLINE_ENV.items():
        monkeypatch.setenv(key, value)


def _fake_download(spec: ModelSpec) -> None:
    spec.local_dir.mkdir(parents=True, exist_ok=True)
    (spec.local_dir / spec.marker_file).write_text("{}", encoding="utf-8")


def _pull_fakes(config: Config) -> None:
    for spec in model_specs(config):
        pull_model(spec, config.paths.models_dir, _fake_download, lambda _: "mit")


def _by_name(checks: list[Check]) -> dict[str, Check]:
    return {check.name: check for check in checks}


def test_collect_hardware_reads_local_machine() -> None:
    hw = collect_hardware()
    assert hw.ram_total_bytes > 0
    assert hw.cpu


@pytest.mark.usefixtures("offline_env")
def test_all_green_with_models_present(config: Config) -> None:
    _pull_fakes(config)
    checks = run_checks(config, FAKE_HW)
    failures = [c for c in checks if c.status is Status.FAIL]
    assert failures == []
    named = _by_name(checks)
    assert named["sqlite-vec"].status is Status.OK
    assert named["FTS5"].status is Status.OK
    # No GPU and no library are warnings, not failures.
    assert named["GPU"].status is Status.WARN
    assert named["paths.library"].status is Status.WARN


@pytest.mark.usefixtures("offline_env")
def test_missing_models_fail(config: Config) -> None:
    named = _by_name(run_checks(config, FAKE_HW))
    assert named["embedding: BAAI/bge-m3"].status is Status.FAIL
    assert "biblio models pull" in named["embedding: BAAI/bge-m3"].detail


def test_missing_offline_env_fails(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in OFFLINE_ENV:
        monkeypatch.delenv(key, raising=False)
    assert _by_name(run_checks(config, FAKE_HW))["environment"].status is Status.FAIL


@pytest.mark.usefixtures("offline_env")
def test_remote_llm_endpoint_fails(config: Config) -> None:
    remote = replace(config, llm=replace(config.llm, base_url="https://api.example.com/v1"))
    assert _by_name(run_checks(remote, FAKE_HW))["llm.base_url"].status is Status.FAIL


@pytest.mark.usefixtures("offline_env")
def test_library_path_is_checked(config: Config, tmp_path: Path) -> None:
    library = tmp_path / "lib"
    missing = replace(config, paths=replace(config.paths, library=library))
    assert _by_name(run_checks(missing, FAKE_HW))["paths.library"].status is Status.FAIL

    (library / "sub").mkdir(parents=True)
    (library / "sub" / "a.md").write_text("# A\n", encoding="utf-8")
    check = _by_name(run_checks(missing, FAKE_HW))["paths.library"]
    assert check.status is Status.OK
    assert "1 .md files" in check.detail


def test_cli_doctor_applies_offline_env_and_fails_without_models(
    project_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "load_config", lambda: load_config(project_copy))
    for key in OFFLINE_ENV:
        # setenv first so that monkeypatch restores the original state, whatever the CLI sets.
        monkeypatch.setenv(key, "0")
        monkeypatch.delenv(key)

    result = CliRunner().invoke(cli.app, ["doctor"])

    assert result.exit_code == 1, result.output  # no models in the temp project
    assert "biblio models pull" in result.output
    assert all(os.environ.get(key) == value for key, value in OFFLINE_ENV.items())


def test_cli_models_pull_leaves_download_env_unset(
    project_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`models pull` must not get the offline env from the root callback."""
    import biblio.models as models

    seen: dict[str, str | None] = {}

    def fake_pull(spec: ModelSpec, models_dir: Path) -> models.ManifestEntry:
        seen["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE")
        return pull_model(spec, models_dir, _fake_download, lambda _: "mit")

    monkeypatch.setattr(cli, "load_config", lambda: load_config(project_copy))
    monkeypatch.setattr(models, "pull_model", fake_pull)
    for key in OFFLINE_ENV:
        monkeypatch.setenv(key, "1")

    result = CliRunner().invoke(cli.app, ["models", "pull"])

    assert result.exit_code == 0, result.output
    assert seen == {"HF_HUB_OFFLINE": None}
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"


# --- the LLM server's context (ADR-015) ---------------------------------------------------------


class _FakeOllama:
    """Answers the Ollama API on loopback: the pinned model is served; `context` is what /api/ps
    reports once it is loaded (None: not loaded until a generate call loads it)."""

    def __init__(self, config: Config, context: int, loaded: bool) -> None:
        self.config, self.context, self.loaded = config, context, loaded
        self.generate: list[object] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        model, path = self.config.llm.model, request.url.path
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": model}]})
        if path == "/api/tags":
            digest = self.config.llm.model_digest
            return httpx.Response(200, json={"models": [{"name": model, "digest": digest}]})
        if path == "/api/generate":
            keep = json.loads(request.content)["keep_alive"]
            self.generate.append(keep)
            self.loaded = keep != 0
            return httpx.Response(200, json={"done": True})
        assert path == "/api/ps"
        loaded = [{"name": model, "context_length": self.context}] if self.loaded else []
        return httpx.Response(200, json={"models": loaded})


@pytest.fixture
def fake_ollama(monkeypatch: pytest.MonkeyPatch) -> Callable[[Config, int, bool], _FakeOllama]:
    def install(config: Config, context: int, loaded: bool) -> _FakeOllama:
        server = _FakeOllama(config, context, loaded)
        real = httpx.Client
        monkeypatch.setattr(
            httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(server), **kw)
        )
        return server

    return install


def _context_check(config: Config, load: bool = False) -> Check:
    if load:  # only the context check, without loading the embedding models of --load-models
        from biblio.doctor import _llm_context_check

        return _llm_context_check(config, True)
    return _by_name(run_checks(config, FAKE_HW))["llm context"]


def test_a_short_served_context_fails(
    config: Config, fake_ollama: Callable[[Config, int, bool], _FakeOllama]
) -> None:
    fake_ollama(config, 4096, True)
    check = _context_check(config)
    assert check.status is Status.FAIL
    assert "4096 tokens < 7024" in check.detail and "serve-llm.ps1" in check.detail


def test_the_scripts_context_passes(
    config: Config, fake_ollama: Callable[[Config, int, bool], _FakeOllama]
) -> None:
    fake_ollama(config, 8192, True)
    check = _context_check(config)
    assert check.status is Status.OK and check.detail == "8192 tokens (needs ≥ 7024)"


def test_an_unloaded_model_is_loaded_only_when_asked(
    config: Config, fake_ollama: Callable[[Config, int, bool], _FakeOllama]
) -> None:
    server = fake_ollama(config, 4096, False)
    check = _context_check(config)
    assert check.status is Status.OK and "--load-models" in check.detail
    assert server.generate == []
    check = _context_check(config, load=True)
    assert check.status is Status.FAIL
    assert server.generate == ["30s", 0] and not server.loaded  # loaded, read, unloaded


def test_the_reasoning_model_is_checked_on_the_same_server(
    config: Config, fake_ollama: Callable[[Config, int, bool], _FakeOllama]
) -> None:
    fake_ollama(config, 8192, True)  # serves only the tutor's model
    check = _by_name(run_checks(config, FAKE_HW))[f"llm: {config.reason.model}"]
    assert check.status is Status.FAIL and "ollama pull" in check.detail
