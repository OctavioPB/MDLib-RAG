"""A question translated to English by the local LLM, to search the English notes too (ADR-052)."""

from __future__ import annotations

from biblio.config import RetrievalConfig
from biblio.generate.ask import load_prompt
from biblio.generate.llm import LLMClient, Message


def translate_query(llm: LLMClient, cfg: RetrievalConfig, question: str) -> str | None:
    """The question in English, one line; None when the reply is empty or far longer than the
    question (the model answered instead of translating)."""
    messages = (
        Message("system", load_prompt("translate_query_system")),
        Message("user", question),
    )
    reply = "".join(llm.stream(messages, cfg.translate_max_tokens))
    lines = [ln.strip().strip("«»\"'") for ln in reply.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    text = lines[0]
    if len(text) > cfg.translate_max_ratio * len(question) + 20:
        return None
    return text
