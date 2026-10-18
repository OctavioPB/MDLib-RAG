"""Dense embeddings: the `Embedder` interface and the bge-m3 implementation (ADR-004, ADR-012)."""

from __future__ import annotations

import gc
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from biblio.config import Config
from biblio.models import local_model_path, model_specs

log = logging.getLogger(__name__)

Vectors = npt.NDArray[np.float32]


class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> Vectors:
        """L2-normalized float32 vectors, one row per text."""
        ...

    def close(self) -> None:
        """Release the model (RAM/VRAM)."""
        ...


@dataclass(frozen=True)
class DeviceChoice:
    device: str  # "cpu", "cuda", "cuda:1", ...
    dtype: str  # "float32" | "float16"


class SentenceTransformerEmbedder:
    """Any sentence-transformers model loaded from a local directory, never from the Hub."""

    def __init__(
        self, config: Config, choice: DeviceChoice, batch_size: int, max_seq_length: int
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        spec = next(s for s in model_specs(config) if s.role == "embedding")
        path = local_model_path(spec, config.paths.models_dir)
        dtype = torch.float16 if choice.dtype == "float16" else torch.float32
        self._model: Any = SentenceTransformer(
            str(path), device=choice.device, local_files_only=True, model_kwargs={"dtype": dtype}
        )
        self._model.max_seq_length = max_seq_length
        self._batch_size = batch_size
        self._device = choice.device
        dim = self._model.get_embedding_dimension()
        if dim is None:
            raise RuntimeError(f"{spec.repo_id} does not report an embedding dimension")
        self._dimension = int(dim)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> Vectors:
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def close(self) -> None:
        del self._model
        gc.collect()
        if self._device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()


def resolve_index_device(config: Config, override: str | None = None) -> DeviceChoice:
    """Device for batch indexing (ADR-012): CUDA if requested and available, else CPU."""
    device = override or config.embedding.index_device
    dtype = config.embedding.index_dtype
    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            log.warning("CUDA is not available; indexing on CPU (~20x slower, ADR-012)")
            device = "cpu"
    if device == "cpu" and dtype == "float16":
        raise ValueError("index_dtype = 'float16' requires a CUDA device")
    return DeviceChoice(device, dtype)


def load_index_embedder(config: Config, choice: DeviceChoice) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        config, choice, config.embedding.batch_size, config.embedding.max_seq_length
    )


def load_query_embedder(config: Config) -> SentenceTransformerEmbedder:
    """Query-time embedder: CPU fp32 by default, identical vectors to a GPU fp32 index."""
    choice = DeviceChoice(config.embedding.device, "float32")
    return SentenceTransformerEmbedder(
        config, choice, config.embedding.batch_size, config.embedding.max_seq_length
    )
