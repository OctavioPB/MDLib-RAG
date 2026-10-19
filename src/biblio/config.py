"""Typed configuration loaded from `config/default.toml`, merged with `config/local.toml`.

The defaults file is the schema: `local.toml` may only override keys that already exist there, so a
typo fails loudly instead of being silently ignored.
"""

from __future__ import annotations

import tomllib
import types
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Each mode adds one piece to the previous ones (PLAN Phase 4): dense → bm25 → hybrid → reranker.
RETRIEVAL_MODES = ("dense", "bm25", "hybrid", "dense-rerank", "hybrid-rerank")


class ConfigError(ValueError):
    """Invalid or inconsistent configuration."""


@dataclass(frozen=True)
class PathsConfig:
    library: Path | None
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    models_dir: Path
    index_db: Path
    drafts_dir: Path


@dataclass(frozen=True)
class RuntimeConfig:
    offline: bool


@dataclass(frozen=True)
class ChunkingConfig:
    min_tokens: int
    target_tokens: int
    max_tokens: int


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str
    revision: str
    device: str
    index_device: str
    index_dtype: str
    batch_size: int
    max_seq_length: int
    index_flush_chunks: int
    download_ignore: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalConfig:
    default_mode: str
    dense_top_n: int
    bm25_top_n: int
    rrf_k: int
    rerank_candidates: int
    final_k: int
    latency_p95_target_ms: int


@dataclass(frozen=True)
class RerankerConfig:
    model: str
    revision: str
    device: str
    fallback: str
    cpu_candidates: int
    max_tokens: int
    batch_size: int
    download_ignore: tuple[str, ...]


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    model: str
    model_digest: str
    tokenizer: str
    tokenizer_revision: str
    tokenizer_files: tuple[str, ...]
    request_timeout_s: float
    temperature: float
    context_budget_tokens: int
    max_fragments: int
    map_reduce: str
    map_reduce_min_guides: int
    map_reduce_max_fragments: int
    max_answer_tokens: int
    probe_timeout_s: float


@dataclass(frozen=True)
class DraftConfig:
    min_sections: int
    max_sections: int
    sibling_guides: int
    sibling_outline_lines: int


@dataclass(frozen=True)
class WebConfig:
    host: str
    port: int


@dataclass(frozen=True)
class ConversationsConfig:
    db: Path
    list_limit: int


@dataclass(frozen=True)
class PracticeConfig:
    db: Path
    max_questions: int
    min_section_tokens: int
    max_section_tokens: int
    question_max_tokens: int
    grade_max_tokens: int
    grader_min_agreement: float
    grader_min_samples: int


@dataclass(frozen=True)
class MapsConfig:
    fragments: int
    max_fragment_chars: int
    concepts_per_fragment: int
    extract_max_tokens: int
    merge_similarity: float
    level1_max: int
    children_max: int
    max_nodes: int
    history_db: Path
    history_limit: int


@dataclass(frozen=True)
class LauncherConfig:
    ollama_path: str
    llm_start_timeout_s: float
    open_browser: bool
    watch: bool


@dataclass(frozen=True)
class WatchConfig:
    interval_s: float
    settle_s: float


@dataclass(frozen=True)
class EvalConfig:
    questions: Path
    answers: Path
    results_dir: Path
    ks: tuple[int, ...]


@dataclass(frozen=True)
class Config:
    root: Path
    paths: PathsConfig
    runtime: RuntimeConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    reranker: RerankerConfig
    llm: LLMConfig
    draft: DraftConfig
    web: WebConfig
    conversations: ConversationsConfig
    practice: PracticeConfig
    maps: MapsConfig
    launcher: LauncherConfig
    watch: WatchConfig
    eval: EvalConfig


_SECTIONS: dict[str, type[Any]] = {
    "paths": PathsConfig,
    "runtime": RuntimeConfig,
    "chunking": ChunkingConfig,
    "embedding": EmbeddingConfig,
    "retrieval": RetrievalConfig,
    "reranker": RerankerConfig,
    "llm": LLMConfig,
    "draft": DraftConfig,
    "web": WebConfig,
    "conversations": ConversationsConfig,
    "practice": PracticeConfig,
    "maps": MapsConfig,
    "launcher": LauncherConfig,
    "watch": WatchConfig,
    "eval": EvalConfig,
}

T = TypeVar("T")


def load_config(root: Path = PROJECT_ROOT, local_path: Path | None = None) -> Config:
    """Load `<root>/config/default.toml` and merge `local_path` over it.

    `local_path` defaults to `<root>/config/local.toml`; it is optional.
    """
    default_path = root / "config" / "default.toml"
    if not default_path.is_file():
        raise ConfigError(f"Default config not found: {default_path}")
    data = _read_toml(default_path)

    local_path = local_path if local_path is not None else root / "config" / "local.toml"
    if local_path.is_file():
        data = _merge(data, _read_toml(local_path), source=local_path)

    missing = set(_SECTIONS) - set(data)
    if missing:
        raise ConfigError(f"Missing config sections: {', '.join(sorted(missing))}")

    sections = {
        name: _build(cls, data[name], section=name, root=root) for name, cls in _SECTIONS.items()
    }
    config = Config(root=root, **sections)
    _validate(config)
    return config


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc


def _merge(
    base: dict[str, Any], override: dict[str, Any], source: Path, prefix: str = ""
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        dotted = f"{prefix}{key}"
        if key not in base:
            raise ConfigError(f"Unknown config key '{dotted}' in {source}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ConfigError(f"'{dotted}' in {source} must be a table")
            merged[key] = _merge(base[key], value, source, prefix=f"{dotted}.")
        else:
            merged[key] = value
    return merged


def _build(cls: type[T], raw: Any, section: str, root: Path) -> T:
    if not isinstance(raw, dict):
        raise ConfigError(f"[{section}] must be a table")
    hints = get_type_hints(cls)
    names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(raw) - names
    if unknown:
        raise ConfigError(f"Unknown keys in [{section}]: {', '.join(sorted(unknown))}")
    missing = names - set(raw)
    if missing:
        raise ConfigError(f"Missing keys in [{section}]: {', '.join(sorted(missing))}")
    values = {
        name: _convert(raw[name], hints[name], key=f"{section}.{name}", root=root) for name in names
    }
    return cls(**values)


def _convert(value: Any, hint: Any, key: str, root: Path) -> Any:
    origin = get_origin(hint)
    if origin in (Union, types.UnionType):
        # Only `X | None` is used: an empty string means "not set".
        (inner,) = [a for a in get_args(hint) if a is not type(None)]
        if value == "":
            return None
        return _convert(value, inner, key, root)
    if origin is tuple:
        item_type = get_args(hint)[0]
        if not isinstance(value, list):
            raise ConfigError(f"'{key}' must be a list")
        return tuple(_convert(v, item_type, f"{key}[]", root) for v in value)
    if hint is Path:
        if not isinstance(value, str) or not value:
            raise ConfigError(f"'{key}' must be a non-empty path string")
        path = Path(value).expanduser()
        return path if path.is_absolute() else root / path
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"'{key}' must be a number")
        return float(value)
    if hint is int and isinstance(value, bool):
        raise ConfigError(f"'{key}' must be an integer")
    if not isinstance(value, hint):
        raise ConfigError(f"'{key}' must be of type {hint.__name__}, got {type(value).__name__}")
    return value


def _validate(config: Config) -> None:
    c = config.chunking
    if not 0 < c.min_tokens <= c.target_tokens <= c.max_tokens:
        raise ConfigError("chunking: expected 0 < min_tokens <= target_tokens <= max_tokens")
    r = config.retrieval
    if r.final_k <= 0 or r.rerank_candidates < r.final_k:
        raise ConfigError("retrieval: expected 0 < final_k <= rerank_candidates")
    if config.embedding.index_dtype not in ("float32", "float16"):
        raise ConfigError("embedding.index_dtype must be 'float32' or 'float16'")
    if config.llm.map_reduce not in ("auto", "on", "off"):
        raise ConfigError("llm.map_reduce must be 'auto', 'on' or 'off'")
    if config.reranker.fallback not in ("skip", "cpu"):
        raise ConfigError("reranker.fallback must be 'skip' or 'cpu'")
    if r.default_mode not in RETRIEVAL_MODES:
        raise ConfigError(f"retrieval.default_mode must be one of {', '.join(RETRIEVAL_MODES)}")
    d = config.draft
    if not 0 < d.min_sections <= d.max_sections:
        raise ConfigError("draft: expected 0 < min_sections <= max_sections")
    p = config.practice
    if not (0 < p.min_section_tokens < p.max_section_tokens and p.max_questions > 0):
        raise ConfigError("practice: expected 0 < min_section_tokens < max_section_tokens")
    if not 0 <= p.grader_min_agreement <= 1:
        raise ConfigError("practice.grader_min_agreement must be between 0 and 1")
    m = config.maps
    if min(m.fragments, m.concepts_per_fragment, m.level1_max, m.children_max) <= 0:
        raise ConfigError("maps: counts must be positive")
    if not 0 < m.merge_similarity <= 1:
        raise ConfigError("maps.merge_similarity must be in (0, 1]")
    if config.launcher.llm_start_timeout_s <= 0:
        raise ConfigError("launcher.llm_start_timeout_s must be positive")
    if not (config.watch.interval_s > 0 and config.watch.settle_s > 0):
        raise ConfigError("watch: interval_s and settle_s must be positive")
    if not config.eval.ks or any(k <= 0 for k in config.eval.ks):
        raise ConfigError("eval.ks must be a non-empty list of positive integers")
