"""The «Razonar» mode (ADR-056): layers, quote checks and the topic gate, with fakes only."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from biblio.config import Config
from biblio.generate.ask import Tutor
from biblio.generate.context import assemble
from biblio.generate.llm import Message
from biblio.generate.reason import (
    REASON,
    Gate,
    Reasoner,
    check_answer,
    parse_subjects,
    split_layers,
)
from biblio.retrieve.types import Hit

from fakes import WordCounter

SPARK = "Las transformaciones son perezosas y solo construyen el plan de ejecución del job."


def hit(rank: int, path: str, text: str = SPARK, heading: str = "Spark") -> Hit:
    return Hit(rank, path, ("Guía", heading), text, rank * 10, rank * 10 + 5, rank, 1.0, "t")


class _Retriever:
    name = "fake"

    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.queries: list[str] = []

    def search(self, query: str, k: int) -> list[Hit]:
        self.queries.append(query)
        return self.hits[:k]

    def close(self) -> None:
        pass


class _LLM:
    """The field's names for the subject call, `answer` for the answer call."""

    def __init__(self, subjects: str, answer: str) -> None:
        self.subjects, self.answer = subjects, answer
        self.systems: list[str] = []

    def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
        self.systems.append(messages[0].content)
        yield self.subjects if "campo de estudio" in messages[0].content else self.answer

    def close(self) -> None:
        pass


class _Scores:
    """A reranker score per definitional query: the name inside it decides."""

    def __init__(self, by_name: dict[str, float]) -> None:
        self.by_name = by_name
        self.queries: list[str] = []

    def __call__(self, query: str, passages: Sequence[str]) -> list[float]:
        self.queries.append(query)
        score = next((s for name, s in self.by_name.items() if name in query), 0.0)
        return [score] * len(passages)


ANSWER = """**Según tus notas:** las transformaciones construyen el plan [1].

**Razonamiento:**
1. «las transformaciones son perezosas» [1]: sin una acción no se ejecuta nada.
2. «el clúster procesa todo de inmediato» [1]: esto no lo dice la nota.

**Conocimiento general (no está en tus notas):** Spark 3 añadió AQE [1].

**Respuesta:** el clúster no hace trabajo."""


def _reasoner(
    config: Config, subjects: str, answer: str, scores: _Scores | None, hits: list[Hit]
) -> tuple[Reasoner, _LLM, _Retriever]:
    retriever = _Retriever(hits)
    llm = _LLM(subjects, answer)
    tutor = Tutor(config, retriever, llm, WordCounter(), "off")
    return Reasoner(config, tutor, llm, retriever, scores), llm, retriever


def test_layers_are_split_by_their_titles() -> None:
    layers = split_layers(ANSWER)
    assert set(layers) == {"notes", "steps", "general", "answer"}
    assert layers["notes"] == "las transformaciones construyen el plan [1]."
    assert layers["answer"] == "el clúster no hace trabajo."
    assert layers["steps"].startswith("1. «las transformaciones")
    other = split_layers("### Razonamiento\n1. paso\nRespuesta corta sin dos puntos")
    assert other == {"steps": "1. paso\nRespuesta corta sin dos puntos"}


def test_quotes_are_checked_against_the_fragment_they_cite() -> None:
    context = assemble([hit(1, "spark.md")], WordCounter(), 10_000, 6, 0)
    checks = check_answer(ANSWER, context, Gate(("Spark",), True, "Spark", 3))
    assert (checks.quotes, checks.quotes_found) == (2, 1)
    assert checks.general_citations == 1  # a citation where none may go
    assert checks.layers == ("notes", "steps", "general", "answer")


def test_the_fields_names_are_parsed_once_each() -> None:
    reply = 'Aquí: {"campos": ["Spark", "spark", " evaluación perezosa ", 3, ""]}'
    assert parse_subjects(reply, 4) == ("Spark", "evaluación perezosa")
    assert parse_subjects('{"campos": ["a", "b", "c"]}', 2) == ("a", "b")
    assert parse_subjects("no es JSON", 4) == ()


def test_the_gate_opens_on_the_first_name_the_library_explains(config: Config) -> None:
    scores = _Scores({"teoría de portafolios": 0.8})
    subjects = '{"campos": ["optimización de carteras", "teoría de portafolios", "frontera"]}'
    hits = [hit(1, "a.md"), hit(2, "b.md")]
    reasoner, _, _ = _reasoner(config, subjects, ANSWER, scores, hits)
    gate = reasoner.gate("¿Qué es Black-Litterman?")
    assert gate == Gate(
        ("optimización de carteras", "teoría de portafolios", "frontera"),
        True,
        "teoría de portafolios",
        2,
    )
    assert len(scores.queries) == 2  # it stops at the name that opened it


def test_a_closed_gate_abstains_without_searching_or_answering(config: Config) -> None:
    subjects = '{"campos": ["cocina del siglo XVII", "historia de la cocina"]}'
    reasoner, llm, retriever = _reasoner(
        config, subjects, ANSWER, _Scores({}), [hit(1, "a.md"), hit(2, "b.md")]
    )
    answer = reasoner.ask("¿Qué técnicas de cocina se usaban en el siglo XVII?")
    assert answer.reason is not None and not answer.reason.gate.open
    assert answer.text == reasoner.abstention and answer.validation.abstained
    assert answer.strategy == REASON and not answer.context.fragments
    assert len(llm.systems) == 1  # only the field's names were asked for
    # the gate searched each name twice (bare + definitional); the tutor never searched
    assert len(retriever.queries) == 4


def test_an_open_gate_allows_the_marked_general_layer(config: Config) -> None:
    reasoner, llm, _ = _reasoner(
        config,
        '{"campos": ["Spark"]}',
        ANSWER,
        _Scores({"Spark": 0.9}),
        [hit(1, "a.md"), hit(2, "b.md")],
    )
    answer = reasoner.ask("¿Qué hace Spark sin una acción?")
    assert answer.reason is not None and answer.reason.gate.open
    assert "Conocimiento general (no está en tus notas)" in llm.systems[-1]
    assert answer.validation.cited == (1,)


def test_an_open_gate_with_nothing_found_abstains(config: Config) -> None:
    class _Empty(_Retriever):
        def search(self, query: str, k: int) -> list[Hit]:
            self.queries.append(query)
            return [] if query == "¿Spark?" else self.hits[:k]

    retriever = _Empty([hit(1, "a.md"), hit(2, "b.md")])
    llm = _LLM('{"campos": ["Spark"]}', ANSWER)
    tutor = Tutor(config, retriever, llm, WordCounter(), "off")
    reasoner = Reasoner(config, tutor, llm, retriever, _Scores({"Spark": 0.9}))
    answer = reasoner.ask("¿Spark?")
    assert answer.reason is not None and answer.reason.gate.open
    assert answer.text == reasoner.abstention and len(llm.systems) == 1


def test_without_a_reranker_the_gate_stays_closed(config: Config) -> None:
    reasoner, _, _ = _reasoner(config, '{"campos": ["Spark"]}', ANSWER, None, [hit(1, "a.md")])
    assert reasoner.gate("¿Spark?") == Gate(("Spark",), False)


def test_the_mode_has_its_own_model_on_the_tutors_server(config: Config) -> None:
    from dataclasses import replace

    from biblio.config import reason_llm

    llm = reason_llm(config)
    assert (llm.model, llm.reasoning_effort) == (config.reason.model, "none")
    assert llm.base_url == config.llm.base_url
    assert llm.context_budget_tokens == config.llm.context_budget_tokens
    unset = replace(config, reason=replace(config.reason, model=""))
    assert reason_llm(unset) is unset.llm
