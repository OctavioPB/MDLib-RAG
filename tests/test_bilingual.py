"""The bilingual first stage (ADR-052): a Spanish question also searched in English."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import replace

from biblio.config import Config
from biblio.generate.llm import Message
from biblio.generate.translate import translate_query
from biblio.retrieve.bilingual import BilingualRetriever
from biblio.retrieve.types import Hit


def _hit(chunk: int, rank: int) -> Hit:
    return Hit(chunk, f"n{chunk}.md", ("Guía", "Sección"), "texto", 1, 2, rank, 0.0, "t")


class _Base:
    name = "hybrid"

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.rankings = {
            "¿Cómo se usa PCA para predecir fallas en rodamientos?": [1, 2, 3, 4],
            "How is PCA used to predict bearing failure?": [9, 2, 8, 7],
        }

    def search(self, query: str, k: int) -> list[Hit]:
        self.queries.append(query)
        ids = self.rankings.get(query, [5])
        return [_hit(c, r) for r, c in enumerate(ids[:k], start=1)]

    def close(self) -> None:
        pass


QUESTION = "¿Cómo se usa PCA para predecir fallas en rodamientos?"
ENGLISH = "How is PCA used to predict bearing failure?"


def test_the_question_keeps_its_head_and_the_english_search_fills_the_rest() -> None:
    asked: list[str] = []

    def translate(q: str) -> str:
        asked.append(q)
        return ENGLISH

    base = _Base()
    retriever = BilingualRetriever(base, translate, rrf_k=60, head=2)
    hits = retriever.search(QUESTION, 4)
    # its own 1 and 2 stay first; English adds 9 and 8 (2 is already in)
    assert [h.chunk_id for h in hits] == [1, 2, 9, 8]
    assert [h.rank for h in hits] == [1, 2, 3, 4]  # renumbered: the metrics read `rank`
    assert retriever.last_translation == ENGLISH
    retriever.search(QUESTION, 4)
    assert asked == [QUESTION]  # translated once
    fused = BilingualRetriever(_Base(), translate, rrf_k=60).search(QUESTION, 4)
    assert fused[0].chunk_id == 2  # equal weight: found by both first


def test_english_questions_and_useless_translations_search_once() -> None:
    base = _Base()
    calls: list[str] = []
    retriever = BilingualRetriever(base, lambda q: calls.append(q) or q, rrf_k=60, head=6)
    retriever.search(ENGLISH, 4)  # already English: not translated
    assert calls == [] and base.queries == [ENGLISH]
    retriever.search("ETL vs ELT", 4)  # the «translation» is the same words
    assert base.queries[1:] == ["ETL vs ELT"] and retriever.last_translation is None
    none = BilingualRetriever(_Base(), lambda q: None, rrf_k=60, head=6)
    assert [h.chunk_id for h in none.search(QUESTION, 2)] == [1, 2]


class _Translator:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
        yield self.reply

    def close(self) -> None:
        pass


def test_a_translation_is_one_line_and_not_an_answer(config: Config) -> None:
    cfg = replace(config.retrieval, translate_max_ratio=2.0)
    assert translate_query(_Translator(f"«{ENGLISH}»\n"), cfg, QUESTION) == ENGLISH
    answer = "PCA reduces the dimensions of the vibration data. " * 6  # it answered instead
    assert translate_query(_Translator(answer), cfg, QUESTION) is None
    assert translate_query(_Translator("  \n"), cfg, QUESTION) is None
