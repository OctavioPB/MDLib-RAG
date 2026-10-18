"""Environment checks behind `biblio doctor`.

Everything here is local: hardware is read from the OS (and `nvidia-smi`, a local process), and no
check opens a socket. A `FAIL` makes the command exit non-zero; a `WARN` does not.
"""

from __future__ import annotations

import platform
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import psutil

from biblio.config import Config
from biblio.models import local_model_path, model_specs, model_status
from biblio.offline import is_loopback_url, missing_offline_env

GIB = 1024**3
_MIN_PYTHON = (3, 11)
_NVIDIA_SMI_TIMEOUT_S = 10


class Status(Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class Check:
    group: str
    name: str
    status: Status
    detail: str


@dataclass(frozen=True)
class GpuInfo:
    name: str
    vram_total_bytes: int
    vram_used_bytes: int
    driver: str


@dataclass(frozen=True)
class HardwareInfo:
    os: str
    cpu: str
    physical_cores: int | None
    logical_cores: int | None
    ram_total_bytes: int
    ram_available_bytes: int
    gpus: tuple[GpuInfo, ...]
    apple_silicon: bool


def collect_hardware() -> HardwareInfo:
    memory = psutil.virtual_memory()
    return HardwareInfo(
        os=f"{platform.system()} {platform.release()} ({platform.version()}, {platform.machine()})",
        cpu=_cpu_name(),
        physical_cores=psutil.cpu_count(logical=False),
        logical_cores=psutil.cpu_count(logical=True),
        ram_total_bytes=memory.total,
        ram_available_bytes=memory.available,
        gpus=_nvidia_gpus(),
        apple_silicon=platform.system() == "Darwin" and platform.machine() == "arm64",
    )


def _cpu_name() -> str:
    system = platform.system()
    try:
        if system == "Windows":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            with key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if system == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            return out.stdout.strip()
        if system == "Linux":
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def _nvidia_gpus() -> tuple[GpuInfo, ...]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return ()
    try:
        out = subprocess.run(
            [
                exe,
                "--query-gpu=name,memory.total,memory.used,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=_NVIDIA_SMI_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        name, total_mib, used_mib, driver = parts
        try:
            gpus.append(GpuInfo(name, int(total_mib) * 1024**2, int(used_mib) * 1024**2, driver))
        except ValueError:
            continue
    return tuple(gpus)


def run_checks(config: Config, hardware: HardwareInfo, load_models: bool = False) -> list[Check]:
    checks: list[Check] = []
    checks += _hardware_checks(hardware)
    checks += _python_checks()
    checks += _sqlite_checks()
    checks += _offline_checks(config)
    checks += _model_checks(config, load_models)
    checks.append(_index_device_check(config))
    checks.append(_reranker_device_check(config))
    checks.append(_llm_server_check(config))
    checks += _config_checks(config)
    return checks


def _hardware_checks(hw: HardwareInfo) -> list[Check]:
    cores = f"{hw.physical_cores} physical / {hw.logical_cores} logical"
    checks = [
        Check("hardware", "OS", Status.OK, hw.os),
        Check("hardware", "CPU", Status.OK, f"{hw.cpu} ({cores})"),
        Check(
            "hardware",
            "RAM",
            Status.OK,
            f"{hw.ram_total_bytes / GIB:.1f} GiB total, "
            f"{hw.ram_available_bytes / GIB:.1f} GiB available",
        ),
    ]
    if hw.gpus:
        for i, gpu in enumerate(hw.gpus):
            free = (gpu.vram_total_bytes - gpu.vram_used_bytes) / GIB
            checks.append(
                Check(
                    "hardware",
                    f"GPU {i}",
                    Status.OK,
                    f"{gpu.name}, {gpu.vram_total_bytes / GIB:.1f} GiB VRAM "
                    f"({free:.1f} GiB free now), driver {gpu.driver}",
                )
            )
    elif hw.apple_silicon:
        checks.append(
            Check("hardware", "GPU", Status.OK, "Apple Silicon (unified memory shared with RAM)")
        )
    else:
        checks.append(
            Check(
                "hardware", "GPU", Status.WARN, "no NVIDIA GPU detected; the LLM would run on CPU"
            )
        )
    return checks


def _python_checks() -> list[Check]:
    version = ".".join(map(str, sys.version_info[:3]))
    ok = sys.version_info[:2] >= _MIN_PYTHON
    return [
        Check(
            "python",
            "Python",
            Status.OK if ok else Status.FAIL,
            f"{version} at {sys.executable}",
        )
    ]


def _sqlite_checks() -> list[Check]:
    checks = [Check("sqlite", "SQLite", Status.OK, sqlite3.sqlite_version)]
    conn = sqlite3.connect(":memory:")
    try:
        if not hasattr(conn, "enable_load_extension"):
            checks.append(
                Check(
                    "sqlite",
                    "enable_load_extension",
                    Status.FAIL,
                    "this Python build cannot load SQLite extensions; use `uv python install 3.12`",
                )
            )
        else:
            checks.append(Check("sqlite", "enable_load_extension", Status.OK, "available"))
            checks.append(_sqlite_vec_check(conn))
        checks.append(_fts5_check(conn))
    finally:
        conn.close()
    return checks


def _sqlite_vec_check(conn: sqlite3.Connection) -> Check:
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        (version,) = conn.execute("select vec_version()").fetchone()
    except Exception as exc:  # any failure here means the index cannot work
        return Check("sqlite", "sqlite-vec", Status.FAIL, f"cannot load: {exc}")
    return Check("sqlite", "sqlite-vec", Status.OK, f"loaded {version}")


def _fts5_check(conn: sqlite3.Connection) -> Check:
    try:
        conn.execute(
            "create virtual table t using fts5(body, tokenize='unicode61 remove_diacritics 2')"
        )
        conn.execute("insert into t(body) values ('regresión lineal')")
        hits = conn.execute("select count(*) from t where t match 'regresion'").fetchone()[0]
    except sqlite3.Error as exc:
        return Check("sqlite", "FTS5", Status.FAIL, f"unavailable: {exc}")
    if hits != 1:
        return Check("sqlite", "FTS5", Status.FAIL, "remove_diacritics does not fold accents")
    return Check("sqlite", "FTS5", Status.OK, "unicode61 remove_diacritics 2 folds accents")


def _offline_checks(config: Config) -> list[Check]:
    checks = []
    if not config.runtime.offline:
        checks.append(Check("offline", "runtime.offline", Status.FAIL, "disabled in config"))
    missing = missing_offline_env()
    checks.append(
        Check(
            "offline",
            "environment",
            Status.FAIL if missing else Status.OK,
            f"not set: {', '.join(missing)}"
            if missing
            else "HF/transformers offline, telemetry off",
        )
    )
    loopback = is_loopback_url(config.llm.base_url)
    checks.append(
        Check(
            "offline",
            "llm.base_url",
            Status.OK if loopback else Status.FAIL,
            config.llm.base_url + ("" if loopback else " is not loopback"),
        )
    )
    return checks


def _model_checks(config: Config, load_models: bool) -> list[Check]:
    checks = []
    models_dir = config.paths.models_dir
    for spec in model_specs(config):
        status = model_status(spec, models_dir)
        name = f"{spec.role}: {spec.repo_id}"
        if not status.available:
            checks.append(
                Check("models", name, Status.FAIL, f"{status.detail}; run `biblio models pull`")
            )
            continue
        entry = status.entry
        size = f", {entry.size_bytes / GIB:.2f} GiB, license {entry.license}" if entry else ""
        checks.append(Check("models", name, Status.OK, status.detail + size))
        if load_models:
            checks.append(_load_check(spec.role, local_model_path(spec, models_dir), config))
    return checks


def _load_check(role: str, path: Path, config: Config) -> Check:
    """Load the model from disk and run one inference. Slow (~seconds, ~2.3 GB RAM each)."""
    name = f"{role}: load + inference"
    start = time.perf_counter()
    try:
        if role == "embedding":
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer(
                str(path), device=config.embedding.device, local_files_only=True
            )
            vector = embedder.encode(["prueba de carga"], normalize_embeddings=True)
            result = f"dim {vector.shape[1]}"
            del embedder
        else:
            from sentence_transformers import CrossEncoder

            reranker = CrossEncoder(str(path), device=config.reranker.device, local_files_only=True)
            query = "¿Qué es la fusión de rankings recíproca?"
            relevant = "Reciprocal Rank Fusion (RRF) suma 1/(k + rango) de cada lista ordenada."
            unrelated = "La receta de pan requiere harina, agua, sal y levadura."
            scores = reranker.predict([(query, relevant), (query, unrelated)])
            del reranker
            if not scores[0] > scores[1]:
                return Check("models", name, Status.FAIL, f"ranks unrelated text first: {scores}")
            result = f"relevant {float(scores[0]):.3f} > unrelated {float(scores[1]):.3f}"
    except Exception as exc:  # report, do not crash the whole doctor run
        return Check("models", name, Status.FAIL, f"{type(exc).__name__}: {exc}")
    return Check("models", name, Status.OK, f"{result} in {time.perf_counter() - start:.1f}s")


def _index_device_check(config: Config) -> Check:
    """Batch indexing embeds on `index_device` (ADR-012); without CUDA it falls back to CPU."""
    device, dtype = config.embedding.index_device, config.embedding.index_dtype
    name = "embedding.index_device"
    if not device.startswith("cuda"):
        return Check("models", name, Status.OK, f"{device} ({dtype})")
    import torch

    if not torch.cuda.is_available():
        return Check(
            "models",
            name,
            Status.WARN,
            f"{device} requested but torch {torch.__version__} sees no CUDA; indexing will use CPU",
        )
    gpu = torch.cuda.get_device_name(0)
    return Check(
        "models", name, Status.OK, f"{device} ({dtype}) on {gpu}, torch {torch.__version__}"
    )


def _reranker_device_check(config: Config) -> Check:
    """Where `search` would rerank right now (ADR-014): a fallback is a warning, not a failure."""
    from biblio.offline import NetworkPolicyError
    from biblio.retrieve.rerank import resolve_rerank_plan

    try:
        plan = resolve_rerank_plan(config)
    except NetworkPolicyError:
        # Reported by the llm.base_url check; never probe a non-loopback endpoint.
        return Check("models", "reranker (now)", Status.FAIL, "llm.base_url is not loopback")
    if plan.device is None:
        detail = "skipped (plain hybrid)"
    else:
        detail = f"{plan.device}, {plan.candidates} candidates"
    if plan.fallback_reason:
        return Check("models", "reranker (now)", Status.WARN, f"{detail}: {plan.fallback_reason}")
    return Check("models", "reranker (now)", Status.OK, detail)


def _llm_server_check(config: Config) -> Check:
    """The tutor's LLM server (ADR-015): down is a warning (only `ask` needs it); a wrong model
    or digest fails."""
    import httpx

    from biblio.offline import NetworkPolicyError, require_loopback

    name = f"llm: {config.llm.model}"
    try:
        require_loopback(config.llm.base_url)
    except NetworkPolicyError:
        return Check("models", name, Status.FAIL, "llm.base_url is not loopback")
    base = config.llm.base_url.rstrip("/")
    try:
        with httpx.Client(timeout=2.0, trust_env=False) as http:
            listed = http.get(f"{base}/models").json().get("data", [])
            tags = http.get(f"{base.removesuffix('/v1')}/api/tags")
    except (httpx.HTTPError, ValueError):
        return Check(
            "models", name, Status.WARN, "server not running; start it with scripts/serve-llm.ps1"
        )
    if config.llm.model not in {m.get("id") for m in listed}:
        return Check(
            "models", name, Status.FAIL, f"not on the server; `ollama pull {config.llm.model}`"
        )
    if tags.status_code != 200:
        return Check("models", name, Status.OK, "served (digest not checked: not an Ollama server)")
    digests = {m.get("name"): m.get("digest") for m in tags.json().get("models", [])}
    if digests.get(config.llm.model) != config.llm.model_digest:
        return Check(
            "models",
            name,
            Status.FAIL,
            f"digest {str(digests.get(config.llm.model))[:12]} "
            f"!= pinned {config.llm.model_digest[:12]}",
        )
    return Check("models", name, Status.OK, f"served, digest {config.llm.model_digest[:12]}")


def _config_checks(config: Config) -> list[Check]:
    library = config.paths.library
    if library is None:
        return [
            Check(
                "config",
                "paths.library",
                Status.WARN,
                "not set; add it to config/local.toml (needed from Phase 1)",
            )
        ]
    if not library.is_dir():
        return [Check("config", "paths.library", Status.FAIL, f"{library} does not exist")]
    count = sum(1 for _ in library.rglob("*.md"))
    return [Check("config", "paths.library", Status.OK, f"{library} ({count} .md files)")]
