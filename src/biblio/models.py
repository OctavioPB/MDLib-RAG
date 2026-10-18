"""Local model weights: download at a pinned revision and resolve them offline.

`pull_model` is the only code path in the project that may use the network (ADR-002).
Everything else resolves weights through `local_model_path`, which never downloads.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from biblio.config import Config

MANIFEST_NAME = "manifest.json"
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
# Created by huggingface_hub inside `local_dir`; not part of the model.
_HF_CACHE_DIR = ".cache"


class ModelNotAvailableError(RuntimeError):
    """The weights are missing locally or do not match the pinned revision."""


@dataclass(frozen=True)
class ModelSpec:
    role: str
    repo_id: str
    revision: str
    ignore_patterns: tuple[str, ...]
    local_dir: Path
    allow_patterns: tuple[str, ...] = ()  # when set, download only these files
    marker_file: str = "config.json"  # its presence means the download is complete


@dataclass(frozen=True)
class ManifestEntry:
    repo_id: str
    revision: str
    local_dir: str
    downloaded_at: str
    license: str | None
    files: int
    size_bytes: int


@dataclass(frozen=True)
class ModelStatus:
    spec: ModelSpec
    available: bool
    detail: str
    entry: ManifestEntry | None


Downloader = Callable[[ModelSpec], None]
LicenseLookup = Callable[[ModelSpec], "str | None"]


def model_specs(config: Config) -> tuple[ModelSpec, ...]:
    models_dir = config.paths.models_dir
    return (
        ModelSpec(
            role="embedding",
            repo_id=config.embedding.model,
            revision=config.embedding.revision,
            ignore_patterns=config.embedding.download_ignore,
            local_dir=models_dir / _dir_name(config.embedding.model),
        ),
        ModelSpec(
            role="reranker",
            repo_id=config.reranker.model,
            revision=config.reranker.revision,
            ignore_patterns=config.reranker.download_ignore,
            local_dir=models_dir / _dir_name(config.reranker.model),
        ),
        ModelSpec(
            role="llm-tokenizer",
            repo_id=config.llm.tokenizer,
            revision=config.llm.tokenizer_revision,
            ignore_patterns=(),
            local_dir=models_dir / f"{_dir_name(config.llm.tokenizer)}-tokenizer",
            allow_patterns=config.llm.tokenizer_files,
            marker_file="tokenizer.json",
        ),
    )


def _dir_name(repo_id: str) -> str:
    return repo_id.rsplit("/", 1)[-1]


def validate_revision(spec: ModelSpec) -> None:
    if not _REVISION_RE.match(spec.revision):
        raise ValueError(
            f"{spec.role} model {spec.repo_id}: revision must be a 40-character commit hash, "
            f"got {spec.revision!r}. Branch names are not allowed (ADR-002)."
        )


def read_manifest(models_dir: Path) -> dict[str, ManifestEntry]:
    path = models_dir / MANIFEST_NAME
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {repo: ManifestEntry(**entry) for repo, entry in raw.get("models", {}).items()}


def write_manifest(models_dir: Path, entries: dict[str, ManifestEntry]) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    payload = {"models": {repo: asdict(entry) for repo, entry in sorted(entries.items())}}
    (models_dir / MANIFEST_NAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )


def model_status(spec: ModelSpec, models_dir: Path) -> ModelStatus:
    entry = read_manifest(models_dir).get(spec.repo_id)
    if entry is None:
        return ModelStatus(spec, False, "not downloaded", None)
    if entry.revision != spec.revision:
        return ModelStatus(
            spec,
            False,
            f"manifest has revision {entry.revision[:12]}, config pins {spec.revision[:12]}",
            entry,
        )
    if not (spec.local_dir / spec.marker_file).is_file():
        return ModelStatus(spec, False, f"files missing in {spec.local_dir}", entry)
    return ModelStatus(spec, True, f"revision {spec.revision[:12]}", entry)


def local_model_path(spec: ModelSpec, models_dir: Path) -> Path:
    """Path to load the model from, offline. Never downloads."""
    status = model_status(spec, models_dir)
    if not status.available:
        raise ModelNotAvailableError(
            f"{spec.role} model {spec.repo_id} is not available locally ({status.detail}). "
            "Run `biblio models pull`."
        )
    return spec.local_dir


def pull_model(
    spec: ModelSpec,
    models_dir: Path,
    download: Downloader | None = None,
    lookup_license: LicenseLookup | None = None,
) -> ManifestEntry:
    """Download `spec` at its pinned revision and record it in the manifest."""
    validate_revision(spec)
    (download or _hf_download)(spec)
    files, size = _dir_stats(spec.local_dir)
    entry = ManifestEntry(
        repo_id=spec.repo_id,
        revision=spec.revision,
        local_dir=spec.local_dir.relative_to(models_dir).as_posix(),
        downloaded_at=datetime.now(UTC).isoformat(timespec="seconds"),
        license=(lookup_license or _hf_license)(spec),
        files=files,
        size_bytes=size,
    )
    entries = read_manifest(models_dir)
    entries[spec.repo_id] = entry
    write_manifest(models_dir, entries)
    return entry


def _dir_stats(directory: Path) -> tuple[int, int]:
    files = [
        p
        for p in directory.rglob("*")
        if p.is_file() and _HF_CACHE_DIR not in p.relative_to(directory).parts
    ]
    return len(files), sum(p.stat().st_size for p in files)


def _hf_download(spec: ModelSpec) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=spec.repo_id,
        revision=spec.revision,
        local_dir=spec.local_dir,
        ignore_patterns=list(spec.ignore_patterns) or None,
        allow_patterns=list(spec.allow_patterns) or None,
    )


def _hf_license(spec: ModelSpec) -> str | None:
    from huggingface_hub import HfApi

    info = HfApi().model_info(spec.repo_id, revision=spec.revision)
    value = info.card_data.get("license") if info.card_data else None
    return str(value) if value is not None else None
