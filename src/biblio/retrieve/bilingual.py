"""Bilingual first stage (ADR-052): a Spanish question is also searched in English, translated by
the LLM; the question's own best hits keep their places and the English search fills the rest.

Most of the library is English book chapters; BM25 cannot match «rodamientos» with «bearing», and
the multilingual reranker is skipped while the LLM holds the GPU (ADR-014), which is when the chat
runs. On four book-only probes asked in Spanish, the chapter ranked 3 to 10 (1 to 2 in English).
Fusing both searches with equal weight cost the Spanish notes 6 points of recall@8 on the eval set;
keeping the question's own top 6 cost 1 (noise).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from biblio.generate.thread import reserve_and_fill
from biblio.maps.build import fold, in_english
from biblio.retrieve.fusion import rrf
from biblio.retrieve.types import Hit, Retriever

Translate = Callable[[str], "str | None"]


class BilingualRetriever:
    def __init__(
        self, base: Retriever, translate: Translate, rrf_k: int, head: int | None = None
    ) -> None:
        """`head`: the question's own best hits that keep their places, the English search only
        filling the rest of the top k; None fuses both with equal weight (RRF)."""
        self._base = base
        self._translate = translate
        self._rrf_k = rrf_k
        self._head = head
        self._cache: dict[str, str | None] = {}
        self.name = f"{base.name}+en"
        self.last_translation: str | None = None  # what the last search added, for the page

    def english(self, query: str) -> str | None:
        """The query in English, or None when it already is English or the translation adds
        nothing (same words, or unusable)."""
        if in_english(query):
            return None
        if query not in self._cache:
            text = self._translate(query)
            self._cache[query] = text if text and fold(text) != fold(query) else None
        return self._cache[query]

    def search(self, query: str, k: int) -> list[Hit]:
        english = self.english(query)
        self.last_translation = english
        own = self._base.search(query, k)
        if english is None:
            return own
        other = self._base.search(english, k)
        if self._head is None:
            return rrf([own, other], self._rrf_k)[:k]
        head = min(self._head, k)
        ranked = reserve_and_fill(own, other, head, k - head)[:k]
        # ranks renumbered: the English hits come with their rank in the other search
        return [replace(h, rank=i) for i, h in enumerate(ranked, start=1)]

    def close(self) -> None:
        self._base.close()
