"""Token counting with the embedding model's tokenizer: chunk sizes are in its tokens, not words."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from biblio.config import Config
from biblio.models import local_model_path, model_specs

_TOKENIZER_FILE = "tokenizer.json"


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class HFTokenCounter:
    """Counts with a Hugging Face `tokenizer.json`, special tokens excluded."""

    def __init__(self, tokenizer_file: Path) -> None:
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)


def load_token_counter(config: Config) -> HFTokenCounter:
    """The embedding model's tokenizer, loaded offline from `models/`."""
    embedding = next(s for s in model_specs(config) if s.role == "embedding")
    return HFTokenCounter(local_model_path(embedding, config.paths.models_dir) / _TOKENIZER_FILE)


def load_llm_token_counter(config: Config) -> HFTokenCounter:
    """The LLM's tokenizer (ADR-015), loaded offline from `models/`: counts the context budget."""
    spec = next(s for s in model_specs(config) if s.role == "llm-tokenizer")
    return HFTokenCounter(local_model_path(spec, config.paths.models_dir) / _TOKENIZER_FILE)


def load_token_counter_safe(config: Config, console: Any) -> HFTokenCounter:
    """`load_llm_token_counter`, turning a missing tokenizer into a clear CLI message."""
    from biblio.models import ModelNotAvailableError

    try:
        return load_llm_token_counter(config)
    except ModelNotAvailableError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(2) from exc
