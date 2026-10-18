from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from biblio.config import Config
from biblio.models import (
    ModelNotAvailableError,
    ModelSpec,
    local_model_path,
    model_specs,
    model_status,
    pull_model,
    read_manifest,
    validate_revision,
)


def _fake_download(spec: ModelSpec) -> None:
    spec.local_dir.mkdir(parents=True, exist_ok=True)
    (spec.local_dir / spec.marker_file).write_text("{}", encoding="utf-8")
    (spec.local_dir / "weights.bin").write_bytes(b"\0" * 1000)
    cache = spec.local_dir / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    (cache / "lock").write_text("x", encoding="utf-8")


def _license(spec: ModelSpec) -> str:
    return "mit"


def test_specs_come_from_config(config: Config) -> None:
    embedding, reranker, tokenizer = model_specs(config)
    assert embedding.repo_id == "BAAI/bge-m3"
    assert embedding.local_dir == config.paths.models_dir / "bge-m3"
    assert reranker.repo_id == "BAAI/bge-reranker-v2-m3"
    assert "onnx/*" in embedding.ignore_patterns
    assert tokenizer.repo_id == "Qwen/Qwen3-4B-Instruct-2507"
    assert tokenizer.allow_patterns == ("tokenizer.json", "tokenizer_config.json", "LICENSE")
    assert tokenizer.marker_file == "tokenizer.json"


def test_default_config_pins_commit_hashes(config: Config) -> None:
    for spec in model_specs(config):
        validate_revision(spec)


@pytest.mark.parametrize("revision", ["", "main", "v1.0", "5617a9f", "Z" * 40])
def test_branch_names_and_short_hashes_are_rejected(config: Config, revision: str) -> None:
    spec = replace(model_specs(config)[0], revision=revision)
    with pytest.raises(ValueError, match="commit hash"):
        validate_revision(spec)


def test_missing_model_is_not_available(config: Config) -> None:
    spec = model_specs(config)[0]
    assert not model_status(spec, config.paths.models_dir).available
    with pytest.raises(ModelNotAvailableError, match="biblio models pull"):
        local_model_path(spec, config.paths.models_dir)


def test_pull_records_manifest_and_resolves_locally(config: Config) -> None:
    spec = model_specs(config)[0]
    models_dir = config.paths.models_dir
    entry = pull_model(spec, models_dir, download=_fake_download, lookup_license=_license)

    assert entry.revision == spec.revision
    assert entry.local_dir == "bge-m3"
    assert entry.files == 2  # the .cache directory is not counted
    assert entry.size_bytes == 1002
    assert read_manifest(models_dir)[spec.repo_id] == entry
    assert local_model_path(spec, models_dir) == spec.local_dir


def test_pull_keeps_other_manifest_entries(config: Config) -> None:
    embedding, reranker, tokenizer = model_specs(config)
    models_dir = config.paths.models_dir
    pull_model(embedding, models_dir, download=_fake_download, lookup_license=_license)
    pull_model(reranker, models_dir, download=_fake_download, lookup_license=_license)
    pull_model(tokenizer, models_dir, download=_fake_download, lookup_license=_license)
    assert set(read_manifest(models_dir)) == {
        embedding.repo_id,
        reranker.repo_id,
        tokenizer.repo_id,
    }


def test_revision_mismatch_is_not_available(config: Config) -> None:
    spec = model_specs(config)[0]
    models_dir = config.paths.models_dir
    pull_model(spec, models_dir, download=_fake_download, lookup_license=_license)
    repinned = replace(spec, revision="0" * 40)
    status = model_status(repinned, models_dir)
    assert not status.available
    assert "config pins" in status.detail


def test_deleted_files_are_not_available(config: Config) -> None:
    spec = model_specs(config)[0]
    models_dir = config.paths.models_dir
    pull_model(spec, models_dir, download=_fake_download, lookup_license=_license)
    (spec.local_dir / "config.json").unlink()
    assert not model_status(spec, models_dir).available


def test_pull_refuses_unpinned_revision_before_downloading(config: Config, tmp_path: Path) -> None:
    spec = replace(model_specs(config)[0], revision="main")

    def must_not_run(spec: ModelSpec) -> None:
        raise AssertionError("download attempted")

    with pytest.raises(ValueError):
        pull_model(spec, tmp_path, download=must_not_run, lookup_license=_license)
