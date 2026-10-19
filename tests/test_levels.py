from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import ClassVar

import pytest
from test_web import _events, engine, request, server  # noqa: F401  (fixtures used by name)

from biblio.config import TUTOR_LEVELS, Config
from biblio.generate.ask import load_prompt
from biblio.generate.llm import Message
from biblio.web.engine import Engine


class _Recorder:
    """Answers with one cited sentence and remembers what it was asked."""

    calls: ClassVar[list[tuple[str, int]]] = []

    def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
        _Recorder.calls.append((messages[0].content, max_tokens))
        yield "Invierte probabilidades [1]."

    def close(self) -> None:
        pass


def test_every_level_has_a_prompt_block_and_a_budget(config: Config) -> None:
    system = load_prompt("tutor_system")
    assert "{level}" in system and "{abstention}" in system
    for level in TUTOR_LEVELS:
        block = load_prompt(f"tutor_level_{level}")
        assert block.strip() and "{" not in block  # inserted with str.format: no placeholders
        assert config.tutor.max_tokens(level) > 0
    assert config.tutor.default_level == "intermediate"
    # the evidence budget plus the longest answer fit the 8k context (ADR-015, ADR-032)
    assert config.llm.context_budget_tokens + config.tutor.advanced_max_tokens <= 8192


@pytest.mark.parametrize(
    ("level", "marker"),
    [("beginner", "PRINCIPIANTE"), ("intermediate", "INTERMEDIO"), ("advanced", "AVANZADO")],
)
def test_the_chosen_level_reaches_the_prompt_and_the_history(
    server: tuple[str, int],  # noqa: F811
    engine: Engine,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    level: str,
    marker: str,
) -> None:
    _Recorder.calls = []
    monkeypatch.setattr(engine, "llm_running", lambda: True)
    monkeypatch.setattr(engine, "_llm_factory", lambda _c: _Recorder())
    body = {"question": "¿Qué hace Bayes?", "level": level}
    _, _, raw = request(server, "POST", "/api/ask", body)
    done = next(d for n, d in _events(raw) if n == "done")
    system, budget = _Recorder.calls[0]
    assert marker in system and budget == engine.config.tutor.max_tokens(level)
    others = {"PRINCIPIANTE", "INTERMEDIO", "AVANZADO"} - {marker}
    assert not any(o in system for o in others)
    _, turns = engine.conversations.get(done["conversation_id"])  # type: ignore[misc]
    assert turns[0].payload["level"] == level


def test_no_level_means_the_default_and_a_wrong_one_is_refused(
    server: tuple[str, int],  # noqa: F811
    engine: Engine,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Recorder.calls = []
    monkeypatch.setattr(engine, "llm_running", lambda: True)
    monkeypatch.setattr(engine, "_llm_factory", lambda _c: _Recorder())
    _, _, raw = request(server, "POST", "/api/ask", {"question": "¿Qué hace Bayes?"})
    done = next(d for n, d in _events(raw) if n == "done")
    assert "INTERMEDIO" in _Recorder.calls[0][0]
    _, turns = engine.conversations.get(done["conversation_id"])  # type: ignore[misc]
    assert turns[0].payload["level"] == "intermediate"
    status, _, body = request(server, "POST", "/api/ask", {"question": "¿Y?", "level": "experto"})
    assert status == 400 and "level" in json.loads(body)["error"]
