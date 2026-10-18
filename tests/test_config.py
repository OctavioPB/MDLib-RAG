from __future__ import annotations

from pathlib import Path

import pytest

from biblio.config import PROJECT_ROOT, ConfigError, load_config


def _write_local(root: Path, text: str) -> None:
    (root / "config" / "local.toml").write_text(text, encoding="utf-8")


def test_real_default_config_loads() -> None:
    config = load_config(PROJECT_ROOT, local_path=PROJECT_ROOT / "config" / "absent.toml")
    assert config.embedding.model == "BAAI/bge-m3"
    assert config.retrieval.rrf_k == 60
    assert config.eval.ks == (1, 5, 8, 20)
    assert config.paths.library is None


def test_relative_paths_resolve_against_root(project_copy: Path) -> None:
    config = load_config(project_copy)
    assert config.paths.models_dir == project_copy / "models"
    assert config.paths.index_db == project_copy / "data" / "index.db"


def test_local_overrides_are_merged_key_by_key(project_copy: Path, tmp_path: Path) -> None:
    library = tmp_path / "library"
    _write_local(
        project_copy,
        f'[paths]\nlibrary = "{library.as_posix()}"\n\n[retrieval]\nfinal_k = 5\n',
    )
    config = load_config(project_copy)
    assert config.paths.library == library
    assert config.retrieval.final_k == 5
    # Untouched keys of the same sections keep their defaults.
    assert config.paths.include == ("**/*.md",)
    assert config.retrieval.rrf_k == 60


def test_unknown_key_in_local_fails(project_copy: Path) -> None:
    _write_local(project_copy, "[retrieval]\nfinalk = 5\n")
    with pytest.raises(ConfigError, match=r"retrieval\.finalk"):
        load_config(project_copy)


def test_unknown_section_in_local_fails(project_copy: Path) -> None:
    _write_local(project_copy, "[retreival]\nfinal_k = 5\n")
    with pytest.raises(ConfigError, match="retreival"):
        load_config(project_copy)


@pytest.mark.parametrize(
    "override",
    [
        '[retrieval]\nfinal_k = "8"\n',
        "[retrieval]\nrerank = 1\n",
        "[chunking]\nmin_tokens = true\n",
        '[paths]\ninclude = "**/*.md"\n',
        "[llm]\ntemperature = false\n",
    ],
)
def test_wrong_types_fail(project_copy: Path, override: str) -> None:
    _write_local(project_copy, override)
    with pytest.raises(ConfigError):
        load_config(project_copy)


def test_int_is_accepted_for_float(project_copy: Path) -> None:
    _write_local(project_copy, "[llm]\ntemperature = 0\n")
    assert load_config(project_copy).llm.temperature == 0.0


def test_inconsistent_chunking_thresholds_fail(project_copy: Path) -> None:
    _write_local(project_copy, "[chunking]\nmin_tokens = 900\n")
    with pytest.raises(ConfigError, match="chunking"):
        load_config(project_copy)


def test_final_k_above_rerank_candidates_fails(project_copy: Path) -> None:
    _write_local(project_copy, "[retrieval]\nfinal_k = 100\n")
    with pytest.raises(ConfigError, match="final_k"):
        load_config(project_copy)


def test_unknown_index_dtype_fails(project_copy: Path) -> None:
    _write_local(project_copy, '[embedding]\nindex_dtype = "int8"\n')
    with pytest.raises(ConfigError, match="index_dtype"):
        load_config(project_copy)


def test_invalid_toml_fails(project_copy: Path) -> None:
    _write_local(project_copy, "[paths\n")
    with pytest.raises(ConfigError, match="Invalid TOML"):
        load_config(project_copy)


def test_missing_default_fails(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path)
