"""Helpers for `biblio start`: everything needed to use the app, in the safe order (ADR-021).

Downloads (models, the LLM) happen only when something is missing and only after the user agrees;
with everything in place a start touches no network. The LLM server settings are read from
`scripts/serve-llm.ps1`, so they live in one place (ADR-015).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from biblio.offline import loopback_port_open, require_loopback

if TYPE_CHECKING:
    from biblio.config import Config

SERVE_LLM_SCRIPT = Path("scripts") / "serve-llm.ps1"
_ENV_LINE_RE = re.compile(r'^\s*\$env:([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"([^"]*)"\s*$')
_LIBRARY_LINE_RE = re.compile(r"^library\s*=.*$", re.MULTILINE)
_PATHS_HEADER_RE = re.compile(r"^\[paths\]\s*$", re.MULTILINE)


# --- library path ---------------------------------------------------------------------------------


def set_local_library(local_toml: Path, library: Path) -> None:
    """Write `paths.library` into `config/local.toml`, keeping everything else in the file."""
    value = f'library = "{library.resolve().as_posix()}"'
    text = local_toml.read_text(encoding="utf-8") if local_toml.is_file() else ""
    paths = _PATHS_HEADER_RE.search(text)
    if paths is not None:
        section_end = text.find("\n[", paths.end())
        section = text[paths.end() : None if section_end < 0 else section_end]
        if _LIBRARY_LINE_RE.search(section):
            section = _LIBRARY_LINE_RE.sub(lambda _: value, section, count=1)
        else:
            section = f"\n{value}{section}"
        text = text[: paths.end()] + section + ("" if section_end < 0 else text[section_end:])
    else:
        text = (text.rstrip("\n") + "\n\n" if text.strip() else "") + f"[paths]\n{value}\n"
    local_toml.parent.mkdir(parents=True, exist_ok=True)
    local_toml.write_text(text, encoding="utf-8", newline="\n")


def looks_like_library(path: Path) -> bool:
    """A folder with at least one Markdown file somewhere inside."""
    return path.is_dir() and next(path.rglob("*.md"), None) is not None


# --- the LLM server -------------------------------------------------------------------------------


def llm_server_env(script: Path) -> dict[str, str]:
    """The `$env:NAME = "value"` lines of `scripts/serve-llm.ps1`."""
    env = {}
    for line in script.read_text(encoding="utf-8").splitlines():
        if match := _ENV_LINE_RE.match(line):
            env[match.group(1)] = match.group(2)
    return env


def find_ollama(configured: str) -> Path | None:
    """The Ollama executable: the configured path, then PATH, then its default Windows folder."""
    candidates = [configured] if configured else []
    found = shutil.which("ollama")
    if found:
        candidates.append(found)
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.append(str(Path(local) / "Programs" / "Ollama" / "ollama.exe"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def served_models(base_url: str) -> dict[str, str] | None:
    """{model name: digest} on a running Ollama server, or None if it does not answer."""
    import httpx

    require_loopback(base_url)
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        with httpx.Client(timeout=3.0, trust_env=False) as http:
            models = http.get(f"{root}/api/tags").json().get("models", [])
    except (httpx.HTTPError, ValueError):
        return None
    return {m.get("name", ""): m.get("digest", "") for m in models}


class LLMServer:
    """An `ollama serve` started by `biblio start`, stopped again when `start` exits."""

    def __init__(self, process: subprocess.Popen[bytes], log_path: Path) -> None:
        self.process = process
        self.log_path = log_path

    @classmethod
    def start(cls, exe: Path, env: dict[str, str], log_path: Path) -> LLMServer:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # no console window on Windows
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                [str(exe), "serve"],
                env={**os.environ, **env},
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=flags,
            )
        return cls(process, log_path)

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()


def wait_for_port(
    url: str,
    timeout_s: float,
    probe_timeout_s: float,
    alive: Callable[[], bool] = lambda: True,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """True once `url`'s loopback port answers; False on timeout or when `alive()` turns False."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if loopback_port_open(url, probe_timeout_s):
            return True
        if not alive():
            return False
        sleep(0.5)
    return False


# --- switching the LLM off and on from the app (ADR-043) ----------------------------------------


def unload_model(base_url: str, model: str, timeout_s: float) -> bool:
    """Ask Ollama to drop `model` from memory now (`keep_alive: 0`): the GPU is freed while a
    server the app did not start keeps running. False if the server did not agree."""
    import httpx

    require_loopback(base_url)
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        with httpx.Client(timeout=timeout_s, trust_env=False) as http:
            reply = http.post(f"{root}/api/generate", json={"model": model, "keep_alive": 0})
    except httpx.HTTPError:
        return False
    return reply.status_code == 200


class LLMSwitch:
    """The tutor's LLM, switched off and on from «Ajustes» (ADR-043). It stops only an Ollama it
    started itself (here or in `biblio start`); one opened outside the app (the tray app, the
    script) only has its model unloaded, which frees the same memory."""

    def __init__(self, config: Config, server: LLMServer | None = None) -> None:
        self.config = config
        self.server = server

    @property
    def owned(self) -> bool:
        return self.server is not None and self.server.process.poll() is None

    def can_start(self) -> bool:
        return find_ollama(self.config.launcher.ollama_path) is not None

    def stop(self) -> str:
        """ "stopped" (the app's server closed) or "unloaded" (the model dropped from memory)."""
        if self.owned:
            assert self.server is not None
            self.server.stop()
            self.server = None
            return "stopped"
        llm = self.config.llm
        if unload_model(llm.base_url, llm.model, self.config.launcher.llm_start_timeout_s):
            return "unloaded"
        raise RuntimeError("Ollama no respondió; no se pudo apagar el modelo.")

    def start(self) -> None:
        """Start `ollama serve` with the settings of `scripts/serve-llm.ps1` (ADR-015) and wait
        until its API answers; raises with a message for the page if it cannot."""
        exe = find_ollama(self.config.launcher.ollama_path)
        if exe is None:
            raise RuntimeError("Ollama no está instalado en este PC.")
        llm, timeout = self.config.llm, self.config.launcher.llm_start_timeout_s
        env = llm_server_env(self.config.root / SERVE_LLM_SCRIPT)
        log = self.config.paths.index_db.parent / "ollama.log"
        server = LLMServer.start(exe, env, log)
        alive = lambda: server.process.poll() is None  # noqa: E731
        if not wait_for_port(llm.base_url, timeout, llm.probe_timeout_s, alive):
            server.stop()
            raise RuntimeError(f"El modelo no arrancó; mira {log}.")
        deadline = time.monotonic() + timeout  # the port opens a moment before the API answers
        while served_models(llm.base_url) is None and time.monotonic() < deadline:
            time.sleep(0.5)
        self.server = server

    def close(self) -> None:
        """When the app exits: stop the server only if the app started it."""
        if self.owned:
            assert self.server is not None
            self.server.stop()
