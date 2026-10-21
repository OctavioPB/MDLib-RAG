"""«Gap Spotter» (ADR-047): the subtopics of a topic the notes leave unexplained, and 5 to 10 guides
to write so the study has no holes.

Candidates come from two marked sources: terms the notes mention without explaining (each with a
quote checked against its section), and a syllabus the model proposes. Every candidate is searched
in the whole library and the model judges the top sections; the notes are credited with a subtopic
only with a checked quote. The gaps are then grouped into guide cards.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

from biblio.config import SpotterConfig
from biblio.generate.ask import load_prompt
from biblio.generate.context import Fragment
from biblio.generate.llm import LLMClient
from biblio.maps.build import fold
from biblio.practice.feynman import contains, render_fragments
from biblio.practice.quiz import _complete, _json_object

UNEXPLAINED = "unexplained"  # «mencionado sin explicar»: named in the notes, never explained
SYLLABUS = "syllabus"  # «propuesto por el modelo»
VERDICTS = {"cubierto": "covered", "parcial": "partial", "no": "missing"}


@dataclass(frozen=True)
class Candidate:
    name: str
    about: str  # what it covers (syllabus) or the sentence where it is mentioned (unexplained)
    source: str  # UNEXPLAINED | SYLLABUS
    mention: tuple[int, str] | None = None  # (fragment number, quote) for an unexplained term


@dataclass(frozen=True)
class Coverage:
    candidate: Candidate
    verdict: str  # covered | partial | missing
    evidence: Fragment | None  # the section that covers it (covered, partial)
    quote: str  # checked against that section


@dataclass(frozen=True)
class GuideCard:
    title: str
    why: str
    gaps: tuple[int, ...]  # indices into the gaps (partial and missing coverages)
    questions: tuple[str, ...]
    notes: tuple[int, ...]  # fragment numbers of the topic's sections it connects to


@dataclass(frozen=True)
class Step:
    stage: str  # unexplained | syllabus | judge | guides
    done: int
    total: int
    coverage: tuple[Coverage, ...] = ()
    guides: tuple[GuideCard, ...] = ()


def _clean(value: object, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split()).strip(" «»\"'.")
    return text if 0 < len(text) <= limit else None


def _number(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().strip("[]").isdigit():
        return int(value.strip().strip("[]"))
    return None


def _same(a: str, b: str) -> bool:
    fa, fb = fold(a), fold(b)
    return bool(fa) and (fa == fb or f" {fa} " in f" {fb} " or f" {fb} " in f" {fa} ")


# --- 1. terms the notes mention without explaining -----------------------------------------------


def parse_unexplained(
    reply: str, fragments: Sequence[Fragment], cfg: SpotterConfig
) -> list[Candidate] | None:
    data = _json_object(reply)
    raw = data.get("terminos") if data is not None else None
    if not isinstance(raw, list):
        return None
    by_number = {f.number: f for f in fragments}
    out: list[Candidate] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        term = _clean(item.get("termino"), cfg.name_max_chars)
        quote = _clean(item.get("cita"), cfg.quote_max_chars)
        number = _number(item.get("fragmento"))
        fragment = by_number.get(number) if number is not None else None
        if not (term and quote and fragment):
            continue
        named = f" {fold(term)} " in f" {fold(fragment.text)} "
        if named and contains(fragment.text, quote) and not any(_same(term, c.name) for c in out):
            out.append(Candidate(term, quote, UNEXPLAINED, (fragment.number, quote)))
    return out


def find_unexplained(
    llm: LLMClient, cfg: SpotterConfig, topic: str, fragments: Sequence[Fragment]
) -> list[Candidate]:
    system = load_prompt("spotter_unexplained_system")
    user = load_prompt("spotter_unexplained_user").format(
        topic=topic, fragments=render_fragments(fragments), max_terms=cfg.terms_per_batch
    )
    for _ in range(2):
        found = parse_unexplained(
            _complete(llm, system, user, cfg.extract_max_tokens), fragments, cfg
        )
        if found is not None:
            return found
    return []


# --- 2. the model's syllabus ----------------------------------------------------------------------


def parse_syllabus(reply: str, cfg: SpotterConfig) -> list[Candidate] | None:
    data = _json_object(reply)
    raw = data.get("subtemas") if data is not None else None
    if not isinstance(raw, list):
        return None
    out: list[Candidate] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = _clean(item.get("nombre"), cfg.name_max_chars)
        about = _clean(item.get("descripcion"), cfg.quote_max_chars) or ""
        if name and not any(_same(name, c.name) for c in out):
            out.append(Candidate(name, about, SYLLABUS))
        if len(out) == cfg.syllabus_size:
            break
    return out


def propose_syllabus(
    llm: LLMClient, cfg: SpotterConfig, topic: str, headings: Sequence[str]
) -> list[Candidate]:
    system = load_prompt("spotter_syllabus_system")
    user = load_prompt("spotter_syllabus_user").format(
        topic=topic,
        headings="\n".join(f"- {h}" for h in headings) or "(ninguna)",
        size=cfg.syllabus_size,
    )
    for _ in range(2):
        found = parse_syllabus(_complete(llm, system, user, cfg.syllabus_max_tokens), cfg)
        if found is not None:
            return found
    return []


# --- 3. is each candidate covered somewhere in the notes? ----------------------------------------


def cfg_quote_chars(quote: str, limit: int = 400) -> int:
    """Where to cut a quote of more than `limit` characters: the last space before it."""
    if len(quote) <= limit:
        return len(quote)
    cut = quote.rfind(" ", 0, limit)
    return cut if cut > 0 else limit


def written_in(text: str, quote: str, run: int) -> bool:
    """Whether `run` consecutive words of `quote` are written in `text` (all of it when shorter).
    Exact quotes failed on near-verbatim ones: a long passage with one word changed, or
    «V^π(s)» against the note's LaTeX; the model had judged both subtopics right (live run)."""
    words = fold(quote).split()
    if len(words) <= run:
        return contains(text, quote)
    haystack = f" {fold(text)} "
    return any(f" {' '.join(words[i : i + run])} " in haystack for i in range(len(words) - run + 1))


def parse_judgement(
    reply: str, candidate: Candidate, hits: Sequence[Fragment], run: int
) -> Coverage | None:
    data = _json_object(reply)
    if data is None:
        return None
    verdict = VERDICTS.get(fold(str(data.get("veredicto", ""))))
    if verdict is None:
        return None
    if verdict == "missing":
        return Coverage(candidate, "missing", None, "")
    number = _number(data.get("fragmento"))
    raw_quote = data.get("cita")
    # a long quote is cut, not refused: one cited the whole paragraph (live run)
    quote = (
        _clean(raw_quote[: cfg_quote_chars(raw_quote)], 400) if isinstance(raw_quote, str) else None
    )
    # the cited fragment first, then any other one read (the number is sometimes off by one)
    ordered = sorted(hits, key=lambda f: f.number != number)
    fragment = next((f for f in ordered if quote and written_in(f.text, quote, run)), None)
    if fragment is None or quote is None:
        # the notes are not credited without evidence (ADR-047)
        return Coverage(candidate, "missing", None, "")
    return Coverage(candidate, verdict, fragment, quote)


def judge(
    llm: LLMClient, cfg: SpotterConfig, candidate: Candidate, hits: Sequence[Fragment]
) -> Coverage:
    if not hits:
        return Coverage(candidate, "missing", None, "")
    system = load_prompt("spotter_judge_system")
    user = load_prompt("spotter_judge_user").format(
        subtopic=candidate.name,
        about=candidate.about or "—",
        fragments=render_fragments(hits),
    )
    for _ in range(2):
        reply = _complete(llm, system, user, cfg.judge_max_tokens)
        found = parse_judgement(reply, candidate, hits, cfg.quote_run_words)
        if found is not None:
            return found
    return Coverage(candidate, "missing", None, "")


# --- 4. the gaps grouped into guides --------------------------------------------------------------


def _salvage(reply: str) -> dict[str, object] | None:
    """A reply cut by the token budget: the guides written before the cut, closed as a list.
    (Measured: the first live run lost five good guides to a cut in the middle of a question.)"""
    start = reply.find("{")
    end = reply.rfind("}")
    while start >= 0 and end > start:
        try:
            value = json.loads(reply[start : end + 1] + "]}")
        except ValueError:
            end = reply.rfind("}", start, end)
            continue
        return value if isinstance(value, dict) else None
    return None


_STRAY_BACKSLASH = re.compile(r'\\(?!["\\/bfnrtu])')


def parse_guides(
    reply: str, gaps: Sequence[Coverage], fragments: Sequence[Fragment], cfg: SpotterConfig
) -> list[GuideCard] | None:
    # LaTeX in a question («\mathbf{w}») is not a JSON escape: it once invalidated a whole reply
    reply = _STRAY_BACKSLASH.sub(r"\\\\", reply)
    data = _json_object(reply) or _salvage(reply)
    raw = data.get("guias") if data is not None else None
    if not isinstance(raw, list):
        return None
    numbers = {f.number for f in fragments}
    used: set[int] = set()
    cards: list[GuideCard] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        title = _clean(item.get("titulo"), cfg.name_max_chars * 2)
        picked = []
        raw_gaps = item.get("subtemas")
        for value in raw_gaps if isinstance(raw_gaps, list) else []:
            n = _number(value)
            if n is not None and 1 <= n <= len(gaps) and n - 1 not in used:
                used.add(n - 1)
                picked.append(n - 1)
        if not (title and picked):
            continue
        raw_questions, raw_notes = item.get("preguntas"), item.get("notas")
        asked = (
            (_clean(v, cfg.quote_max_chars) for v in raw_questions)
            if isinstance(raw_questions, list)
            else ()
        )
        questions = tuple(q for q in asked if q)[: cfg.questions_per_guide]
        cited = (_number(v) for v in raw_notes) if isinstance(raw_notes, list) else ()
        notes = tuple(dict.fromkeys(n for n in cited if n is not None and n in numbers))
        why = _clean(item.get("por_que"), cfg.quote_max_chars) or ""
        cards.append(GuideCard(title, why, tuple(picked), questions, notes))
        if len(cards) == cfg.guides_max:
            break
    return cards


def complete_cards(
    cards: list[GuideCard], gaps: Sequence[Coverage], cfg: SpotterConfig
) -> list[GuideCard]:
    """At least `guides_min` cards when there are that many gaps: a gap left out gets its own."""
    used = {g for c in cards for g in c.gaps}
    for i, gap in enumerate(gaps):
        if len(cards) >= min(cfg.guides_min, len(gaps)) or len(cards) == cfg.guides_max:
            break
        if i not in used:
            cards.append(GuideCard(gap.candidate.name, "", (i,), (), ()))
            used.add(i)
    return cards


def suggest_guides(
    llm: LLMClient,
    cfg: SpotterConfig,
    topic: str,
    gaps: Sequence[Coverage],
    fragments: Sequence[Fragment],
) -> list[GuideCard]:
    if not gaps:
        return []
    kinds = {"partial": "tratado a medias", "missing": "falta"}
    listed = "\n".join(
        f"{i}. {g.candidate.name} ({kinds[g.verdict]}"
        + (", mencionado sin explicar" if g.candidate.source == UNEXPLAINED else "")
        + (f"): {g.candidate.about}" if g.candidate.about else ")")
        for i, g in enumerate(gaps, start=1)
    )
    notes = "\n".join(f"[{f.number}] {f.citation}" for f in fragments)
    system = load_prompt("spotter_guides_system")
    user = load_prompt("spotter_guides_user").format(
        topic=topic,
        gaps=listed,
        notes=notes,
        low=min(cfg.guides_min, len(gaps)),
        high=min(cfg.guides_max, len(gaps)),
        questions=cfg.questions_per_guide,
    )
    for _ in range(2):
        cards = parse_guides(
            _complete(llm, system, user, cfg.guides_max_tokens), gaps, fragments, cfg
        )
        if cards is not None:
            return complete_cards(cards, gaps, cfg)
    return complete_cards([], gaps, cfg)


# --- the whole run --------------------------------------------------------------------------------


def spot_gaps(
    llm: LLMClient,
    cfg: SpotterConfig,
    topic: str,
    fragments: Sequence[Fragment],
    search: Callable[[str], list[Fragment]],
    llm_turn: Callable[[], AbstractContextManager[object]] = nullcontext,
) -> Iterator[Step]:
    """Steps as they happen; the last one carries the coverage and the guide cards."""
    batches = [fragments[i : i + cfg.batch] for i in range(0, len(fragments), cfg.batch)]
    terms: list[Candidate] = []
    for i, batch in enumerate(batches, start=1):
        with llm_turn():
            for term in find_unexplained(llm, cfg, topic, batch):
                if not any(_same(term.name, t.name) for t in terms):
                    terms.append(term)
        yield Step("unexplained", i, len(batches))
    terms = terms[: cfg.unexplained_max]
    yield Step("syllabus", 0, 1)
    headings = list(
        dict.fromkeys(" › ".join(f.heading_path[1:]) or f.heading_path[0] for f in fragments)
    )
    with llm_turn():
        syllabus = propose_syllabus(llm, cfg, topic, headings)
    candidates = list(terms)
    for c in syllabus:
        if not any(_same(c.name, t.name) for t in candidates):
            candidates.append(c)
    yield Step("syllabus", 1, 1)
    coverage: list[Coverage] = []
    for i, candidate in enumerate(candidates, start=1):
        hits = search(candidate.name)
        with llm_turn():
            judged = judge(llm, cfg, candidate, hits)
        # an unexplained term the library explains elsewhere is not a gap
        if not (candidate.source == UNEXPLAINED and judged.verdict == "covered"):
            coverage.append(judged)
        yield Step("judge", i, len(candidates), tuple(coverage))
    gaps = [c for c in coverage if c.verdict != "covered"]
    yield Step("guides", 0, 1, tuple(coverage))
    with llm_turn():
        cards = suggest_guides(llm, cfg, topic, gaps, fragments)
    yield Step("guides", 1, 1, tuple(coverage), tuple(cards))
