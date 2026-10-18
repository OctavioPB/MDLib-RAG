"""Post-validation of generated answers (`rag-prompting` skill: not optional)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# [2], [1][3], [1, 3], [1;3]
_CITATION_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
# A small model may copy the instruction's placeholder instead of a number: "[n]", "[x]", "[?]".
_PLACEHOLDER_RE = re.compile(r"\[(?:n|x|\?|n[úu]mero|fragmento)\]", re.IGNORECASE)
_NOISE_RE = re.compile(r"[\W_]+", re.UNICODE)


@dataclass(frozen=True)
class Validation:
    cited: tuple[int, ...]  # valid fragment numbers, in order of first citation
    invalid: tuple[int, ...]  # cited numbers with no such fragment
    abstained: bool  # the model said the library does not contain the answer
    grounded: bool  # answered with valid citations only: no invalid numbers, no placeholders
    placeholders: int = 0  # "[n]"-style citations that point to nothing


def count_placeholders(text: str) -> int:
    return len(_PLACEHOLDER_RE.findall(text))


def extract_citations(answer: str) -> list[int]:
    numbers: list[int] = []
    for group in _CITATION_RE.findall(answer):
        for part in re.split(r"[,;]", group):
            n = int(part.strip())
            if n not in numbers:
                numbers.append(n)
    return numbers


def _simplify(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    no_marks = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _NOISE_RE.sub(" ", no_marks).strip().casefold()


def is_abstention(answer: str, abstention: str) -> bool:
    """The answer is (essentially) only the fixed abstention sentence."""
    simple = _simplify(answer)
    target = _simplify(abstention)
    return simple == target or (simple.startswith(target) and len(simple) <= len(target) + 20)


def validate(answer: str, n_fragments: int, abstention: str) -> Validation:
    numbers = extract_citations(answer)
    cited = tuple(n for n in numbers if 1 <= n <= n_fragments)
    invalid = tuple(n for n in numbers if not 1 <= n <= n_fragments)
    abstained = is_abstention(answer, abstention)
    placeholders = count_placeholders(answer)
    return Validation(
        cited=cited,
        invalid=invalid,
        abstained=abstained,
        grounded=not abstained and bool(cited) and not invalid and not placeholders,
        placeholders=placeholders,
    )
