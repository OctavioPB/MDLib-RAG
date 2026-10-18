"""Test doubles shared by the index, retrieval and eval tests (no model weights needed)."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from biblio.index.embedder import Vectors

DIM = 16


class FakeEmbedder:
    """Bag-of-words vectors hashed into DIM buckets: identical texts get identical vectors."""

    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    @property
    def dimension(self) -> int:
        return DIM

    def embed(self, texts: Sequence[str]) -> Vectors:
        self.calls += 1
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                out[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.where(norms == 0, 1, norms)).astype(np.float32)

    def close(self) -> None:
        self.closed = True


class WordCounter:
    def count(self, text: str) -> int:
        return len(text.split())
