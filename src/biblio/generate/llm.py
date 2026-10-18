"""`LLMClient`: the only way the system talks to a language model (ADR-007, ADR-015).

The implementation speaks the OpenAI-compatible chat API that Ollama, llama-server and LM Studio
expose. It refuses any endpoint that is not on loopback (ADR-002, `offline-guarantee` layer 3).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx

from biblio.config import LLMConfig
from biblio.offline import require_loopback


class LLMError(RuntimeError):
    """The LLM server is unreachable or answered with an error."""


@dataclass(frozen=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMClient(Protocol):
    def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
        """Yield the answer as it is generated."""
        ...

    def close(self) -> None: ...


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig) -> None:
        require_loopback(config.base_url)
        self._config = config
        self._http = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            timeout=httpx.Timeout(config.request_timeout_s, connect=5.0),
            trust_env=False,  # never route through a proxy from the environment
        )

    def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
        body = {
            "model": self._config.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self._config.temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        try:
            with self._http.stream("POST", "/chat/completions", json=body) as response:
                if response.status_code != 200:
                    detail = response.read().decode("utf-8", errors="replace")[:300]
                    raise LLMError(f"LLM server answered {response.status_code}: {detail}")
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if payload == "[DONE]":
                        return
                    delta = json.loads(payload)["choices"][0].get("delta", {})
                    if content := delta.get("content"):
                        yield content
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Cannot reach the LLM server at {self._config.base_url} ({exc}). "
                "Start it with scripts/serve-llm.ps1."
            ) from exc

    def close(self) -> None:
        self._http.close()


def load_llm_client(config: LLMConfig) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(config)
