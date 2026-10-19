"""«Explícamelo tú» (ADR-027): an explanation in the user's words, checked idea by idea against the
retrieved notes.

Step 1 takes the key ideas of the concept from the notes, while the user writes. Step 2 checks which
of them the explanation covers and which of its claims the notes support. Every point the model
reports carries a quote that is checked against the literal text (accent and case folded, as maps,
ADR-022); a point that cannot be checked is dropped or, for coverage, counted as not found.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from biblio.config import FeynmanConfig
from biblio.generate.ask import load_prompt
from biblio.generate.context import Fragment
from biblio.generate.llm import LLMClient
from biblio.maps.build import fold
from biblio.practice.quiz import _complete, _json_object

COVERAGE = {"cubierta": "covered", "parcial": "partial", "falta": "missing"}
CLAIMS = {"respaldada": "supported", "contradice": "contradicts", "sin respaldo": "unsupported"}
_MAX_TEXT = 400  # characters of an idea, a claim or a quote
_MIN_QUOTE_WORDS = 2


@dataclass(frozen=True)
class Idea:
    text: str
    fragment: int  # Fragment.number
    quote: str  # literal text of that fragment


@dataclass(frozen=True)
class Coverage:
    verdict: str  # covered | partial | missing
    phrase: str  # literal text of the explanation; empty when missing


@dataclass(frozen=True)
class Claim:
    text: str  # literal text of the explanation
    verdict: str  # supported | contradicts | unsupported
    fragment: int  # 0 when unsupported
    quote: str


def contains(text: str, quote: str) -> bool:
    """Whether `quote` is written in `text`, ignoring accents, case and punctuation."""
    needle = fold(quote)
    return len(needle.split()) >= _MIN_QUOTE_WORDS and f" {needle} " in f" {fold(text)} "


def _clean(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split()).strip(" «»\"'")
    return text if 0 < len(text) <= _MAX_TEXT else None


def _number(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().strip("[]").isdigit():
        return int(value.strip().strip("[]"))
    return None


def render_fragments(fragments: Sequence[Fragment]) -> str:
    return "\n\n".join(f.render() for f in fragments)


# --- step 1: the key ideas of the notes ----------------------------------------------------------


def parse_ideas(reply: str, fragments: Sequence[Fragment], max_ideas: int) -> list[Idea] | None:
    """The model's ideas, each with a quote found in its fragment; None if the reply is unusable."""
    data = _json_object(reply)
    raw = data.get("ideas") if data is not None else None
    if not isinstance(raw, list):
        return None
    by_number = {f.number: f for f in fragments}
    ideas: list[Idea] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        text, quote = _clean(item.get("idea")), _clean(item.get("cita"))
        number = _number(item.get("fragmento"))
        fragment = by_number.get(number) if number is not None else None
        if text is None or quote is None or fragment is None or fold(text) in seen:
            continue
        if not contains(fragment.text, quote):
            continue  # not in the notes it cites: dropped (rule 2)
        seen.add(fold(text))
        ideas.append(Idea(text, fragment.number, quote))
        if len(ideas) == max_ideas:
            break
    return ideas


def extract_ideas(
    llm: LLMClient, cfg: FeynmanConfig, concept: str, fragments: Sequence[Fragment]
) -> list[Idea]:
    user = load_prompt("feynman_ideas_user").format(
        concept=concept, fragments=render_fragments(fragments), max_ideas=cfg.max_ideas
    )
    system = load_prompt("feynman_ideas_system")
    for _ in range(2):  # one retry
        ideas = parse_ideas(
            _complete(llm, system, user, cfg.ideas_max_tokens), fragments, cfg.max_ideas
        )
        if ideas:
            return ideas
    return []


# --- step 2a: which ideas the explanation covers -------------------------------------------------


def parse_coverage(reply: str, ideas: Sequence[Idea], explanation: str) -> list[Coverage] | None:
    """One verdict per idea, in order. A covered or partial verdict needs the phrase that says it,
    written in the explanation; without it, or when the model skipped the idea, it is «missing»."""
    data = _json_object(reply)
    raw = data.get("ideas") if data is not None else None
    if not isinstance(raw, list):
        return None
    found: dict[int, Coverage] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        number = _number(item.get("numero"))
        verdict = COVERAGE.get(fold(str(item.get("veredicto", ""))))
        if number is None or not 1 <= number <= len(ideas) or verdict is None or number in found:
            continue
        phrase = _clean(item.get("frase")) or ""
        if verdict != "missing" and not contains(explanation, phrase):
            verdict, phrase = "missing", ""
        found[number] = Coverage(verdict, phrase if verdict != "missing" else "")
    if not found:
        return None
    return [found.get(i, Coverage("missing", "")) for i in range(1, len(ideas) + 1)]


def check_coverage(
    llm: LLMClient, cfg: FeynmanConfig, concept: str, ideas: Sequence[Idea], explanation: str
) -> list[Coverage] | None:
    listed = "\n".join(f"{i}. {idea.text}" for i, idea in enumerate(ideas, start=1))
    user = load_prompt("feynman_cover_user").format(
        concept=concept, ideas=listed, explanation=explanation
    )
    system = load_prompt("feynman_cover_system")
    for _ in range(2):
        coverage = parse_coverage(
            _complete(llm, system, user, cfg.check_max_tokens), ideas, explanation
        )
        if coverage is not None:
            return coverage
    return None


# --- step 2b: whether the notes support the explanation's claims ---------------------------------


def parse_claims(
    reply: str, fragments: Sequence[Fragment], explanation: str, max_claims: int
) -> list[Claim] | None:
    """Claims written in the explanation; «supported» and «contradicts» need a quote found in the
    fragment they cite, or the claim is dropped."""
    data = _json_object(reply)
    raw = data.get("afirmaciones") if data is not None else None
    if not isinstance(raw, list):
        return None
    by_number = {f.number: f for f in fragments}
    claims: list[Claim] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = _clean(item.get("afirmacion"))
        verdict = CLAIMS.get(fold(str(item.get("veredicto", ""))))
        if text is None or verdict is None or fold(text) in seen or not contains(explanation, text):
            continue
        if verdict == "unsupported":
            claim = Claim(text, verdict, 0, "")
        else:
            number = _number(item.get("fragmento"))
            fragment = by_number.get(number) if number is not None else None
            quote = _clean(item.get("cita"))
            if fragment is None or quote is None or not contains(fragment.text, quote):
                continue
            claim = Claim(text, verdict, fragment.number, quote)
        seen.add(fold(text))
        claims.append(claim)
        if len(claims) == max_claims:
            break
    return claims


def check_claims(
    llm: LLMClient,
    cfg: FeynmanConfig,
    concept: str,
    fragments: Sequence[Fragment],
    explanation: str,
) -> list[Claim] | None:
    user = load_prompt("feynman_claims_user").format(
        concept=concept,
        fragments=render_fragments(fragments),
        explanation=explanation,
        max_claims=cfg.max_claims,
    )
    system = load_prompt("feynman_claims_system")
    for _ in range(2):
        claims = parse_claims(
            _complete(llm, system, user, cfg.check_max_tokens),
            fragments,
            explanation,
            cfg.max_claims,
        )
        if claims is not None:
            return claims
    return None


# --- the proposed grade --------------------------------------------------------------------------


def proposed_grade(
    coverage: Sequence[Coverage], claims: Sequence[Claim], cfg: FeynmanConfig
) -> tuple[float, str]:
    """(coverage score, correct | partial | wrong): what the user is asked to accept or change."""
    if not coverage:
        return 0.0, "wrong"
    points = {"covered": 1.0, "partial": 0.5, "missing": 0.0}
    score = sum(points[c.verdict] for c in coverage) / len(coverage)
    contradicts = any(c.verdict == "contradicts" for c in claims)
    if score >= cfg.correct_at and not contradicts:
        return score, "correct"
    if score >= cfg.partial_at:
        return score, "partial"
    return score, "wrong"
