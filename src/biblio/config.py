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
    skip_navigation: bool  # tables of contents are not results (ADR-044)
    navigation_margin: int  # extra hits asked for, to make up for the ones dropped
    bm25_heading_weight: float  # BM25 weight of the heading path against the text (1 = equal)
    group_depth: int  # grouped search asks for this many hits per note shown (ADR-045)
    bilingual: bool  # the chat also searches the question in English (ADR-052)
    bilingual_head: int  # its own best hits keep their places (ADR-052)
    translate_max_tokens: int
    translate_max_ratio: float


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
    lan: bool
    pairing_minutes: int
    device_days: int
    devices: Path


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
    daily_reviews: int
    desired_retention: float
    max_interval_days: int
    review_minutes_per_question: float
    fsrs_weights: tuple[float, ...]
    mastered_at: float
    partial_at: float


@dataclass(frozen=True)
class FeynmanConfig:
    fragments: int
    max_fragment_chars: int
    min_fragments: int
    max_ideas: int
    max_claims: int
    min_explanation_chars: int
    max_explanation_chars: int
    ideas_max_tokens: int
    check_max_tokens: int
    correct_at: float
    partial_at: float
    min_agreement: float
    min_samples: int
    eval_match_similarity: float


@dataclass(frozen=True)
class PlanConfig:
    max_guides: int
    headings_per_guide: int
    order_max_tokens: int
    reason_max_chars: int
    mastered_coverage: float


@dataclass(frozen=True)
class GapsConfig:
    topic_max_tokens: int
    topic_max_words: int
    candidate_similarity: float
    max_candidates: int
    same_max_tokens: int
    map_min_concepts: int


@dataclass(frozen=True)
class ReaderConfig:
    passage_min_chars: int
    passage_max_chars: int
    passage_questions: int
    asset_max_mb: int  # the largest image the reader shows (ADR-048)
    asset_index_seconds: int  # how often a missed embed may walk the library again
    outline_depth: int  # heading levels in the reader's index (ADR-049)
    words_per_minute: int  # reading speed for the time left
    callout_headings: tuple[str, ...]  # headings that are a book's boxes, not sections (ADR-053)


TUTOR_LEVELS = ("beginner", "intermediate", "advanced")


@dataclass(frozen=True)
class TutorConfig:
    default_level: str
    beginner_max_tokens: int
    intermediate_max_tokens: int
    advanced_max_tokens: int

    def max_tokens(self, level: str) -> int:
        return int(getattr(self, f"{level}_max_tokens"))


@dataclass(frozen=True)
class DictionaryConfig:
    fragments: int
    guide_max_fragments: int
    max_fragment_chars: int
    entries_per_fragment: int
    extract_max_tokens: int
    max_entries: int
    definition_max_chars: int
    history_limit: int


@dataclass(frozen=True)
class SpotterConfig:  # «Gap Spotter» in Guías (ADR-047)
    sections: int
    max_fragment_chars: int
    batch: int
    terms_per_batch: int
    unexplained_max: int
    syllabus_size: int
    judge_hits: int
    guides_min: int
    guides_max: int
    questions_per_guide: int
    name_max_chars: int
    quote_max_chars: int
    quote_run_words: int
    extract_max_tokens: int
    syllabus_max_tokens: int
    judge_max_tokens: int
    guides_max_tokens: int
    history_limit: int


@dataclass(frozen=True)
class MapsConfig:
    fragments: int
    candidates: int  # first-stage hits per query (ADR-051)
    define_query: str  # the definitional question; {topic} is the map's topic
    rerank: bool  # order the candidates with the CPU reranker
    max_fragment_chars: int
    concepts_per_fragment: int
    extract_max_tokens: int
    merge_similarity: float
    level1_max: int
    children_max: int
    max_nodes: int
    organize: bool  # the LLM chooses the key branches (ADR-051)
    organize_concepts: int
    organize_ask_branches: int
    organize_min_branches: int
    organize_max_tokens: int
    organize_children: bool  # the LLM confirms each branch's children (ADR-051)
    organize_child_candidates: int
    focus_max_chars: int  # how a fragment treats the topic (ADR-051 stage 3)
    focus_max_tokens: int
    general_score: float  # a fragment explains the topic in general (ADR-051 stage 3)
    min_general: int
    history_db: Path
    history_limit: int


@dataclass(frozen=True)
class LibraryConfig:
    relocate_min_overlap: float
    browse_max_entries: int
    minutes_per_1000_gpu: float
    minutes_per_1000_cpu: float
    books_folder: str  # a folder named so holds books (ADR-053)


@dataclass(frozen=True)
class ThreadConfig:
    plan: bool
    plan_max_tokens: int
    history_turns: int
    gist_chars: int
    max_searches: int
    search_depth: int
    question_head: int
    fill: int
    evidence_fragments: int
    frame_max_chars: int


@dataclass(frozen=True)
class DiagramsConfig:
    fragments: int
    max_fragment_chars: int
    extract_max_tokens: int
    steps_per_fragment: int
    min_steps: int
    messages_per_fragment: int
    max_participants: int
    causes_per_fragment: int
    max_causes: int
    categories_min: int
    categories_max: int
    group_max_tokens: int
    six_m: tuple[str, ...]
    suggest_fragments: int
    suggest_max_tokens: int


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
    feynman: Path
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
    feynman: FeynmanConfig
    plan: PlanConfig
    gaps: GapsConfig
    reader: ReaderConfig
    tutor: TutorConfig
    dictionary: DictionaryConfig
    spotter: SpotterConfig
    maps: MapsConfig
    diagrams: DiagramsConfig
    thread: ThreadConfig
    library: LibraryConfig
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
    "feynman": FeynmanConfig,
    "plan": PlanConfig,
    "gaps": GapsConfig,
    "reader": ReaderConfig,
    "tutor": TutorConfig,
    "dictionary": DictionaryConfig,
    "spotter": SpotterConfig,
    "maps": MapsConfig,
    "diagrams": DiagramsConfig,
    "thread": ThreadConfig,
    "library": LibraryConfig,
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
    if r.group_depth < 1:
        raise ConfigError("retrieval.group_depth must be at least 1")
    if r.navigation_margin < 0 or r.bm25_heading_weight <= 0:
        raise ConfigError("retrieval: expected navigation_margin >= 0 and bm25_heading_weight > 0")
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
    if len(p.fsrs_weights) != 17 or not 0.5 < p.desired_retention < 1:
        raise ConfigError("practice: fsrs_weights needs 17 values and 0.5 < desired_retention < 1")
    if not 0 < p.partial_at < p.mastered_at <= 1 or p.daily_reviews <= 0:
        raise ConfigError("practice: expected 0 < partial_at < mastered_at <= 1, daily_reviews > 0")
    pl = config.plan
    if pl.max_guides < 2 or not 0 < pl.mastered_coverage <= 1:
        raise ConfigError("plan: expected max_guides >= 2 and 0 < mastered_coverage <= 1")
    if config.tutor.default_level not in TUTOR_LEVELS:
        raise ConfigError(f"tutor.default_level must be one of {', '.join(TUTOR_LEVELS)}")
    gp = config.gaps
    if not 0 < gp.candidate_similarity <= 1 or gp.topic_max_words < 2 or gp.max_candidates < 1:
        raise ConfigError(
            "gaps: expected 0 < candidate_similarity <= 1, topic_max_words >= 2, "
            "max_candidates >= 1"
        )
    dg = config.diagrams
    if not 2 <= dg.categories_min <= dg.categories_max <= 8 or len(dg.six_m) != 6:
        raise ConfigError(
            "diagrams: expected 2 <= categories_min <= categories_max <= 8, six_m of 6"
        )
    fy = config.feynman
    if not 0 < fy.partial_at < fy.correct_at <= 1 or not 0 < fy.min_fragments <= fy.fragments:
        raise ConfigError(
            "feynman: expected 0 < partial_at < correct_at <= 1, 0 < min_fragments <= fragments"
        )
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
