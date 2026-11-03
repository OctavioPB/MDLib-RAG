"""Choosing sections, and generating and grading questions with the local LLM (ADR-020).

Every question comes from one section and keeps it as its source (rule 2). The model's replies are
JSON and are validated; an invalid reply is asked once more and then skipped.
"""

from __future__ import annotations

import json
import math
import random
import re
import sqlite3
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from biblio.config import Config
from biblio.generate.ask import load_prompt
from biblio.generate.llm import LLMClient, Message
from biblio.practice.store import KINDS, QuestionContent, Section, StoredQuestion
from biblio.retrieve.store import scope_prefix
from biblio.retrieve.types import Retriever

SCOPE_KINDS = (
    "guide", "chapter", "book", "folder", "library", "topic", "review", "srs", "sections"
)  # fmt: skip
# "srs": today's spaced-repetition reviews (ADR-025); "sections": given sections (ADR-026)
_MAX_TEXT = 500  # characters of a question, statement or option
_MAX_EXPLANATION = 1200
# sections with nothing to ask about: navigation and reference lists
_NON_CONTENT = {
    "indice",
    "tabla de contenidos",
    "tabla de contenido",
    "contenido",
    "contenidos",
    "referencias",
    "referencias canonicas",
    "bibliografia",
    "fuentes",
    "lecturas recomendadas",
    "recursos",
    "enlaces",
}
# words that never name a topic; the rest compete by how rare they are in the library
_TOPIC_STOPWORDS = {
    "de", "del", "la", "las", "el", "los", "un", "una", "y", "o", "en", "con", "por", "para",
    "the", "of", "and", "to", "in", "for", "on", "a", "an", "que", "sobre", "como",
}  # fmt: skip
_VERDICTS = {"correcta": "correct", "parcial": "partial", "incorrecta": "wrong"}
_OPTION_LETTER_RE = re.compile(r"^\s*[A-Da-d]\s*[).:\-]\s+")  # "B) texto" → "texto"


def is_content_heading(title: str) -> bool:
    """False for navigation and reference sections (table of contents, references…)."""
    return _plain(title) not in _NON_CONTENT


def _plain(text: str) -> str:
    """Lower case, no accents, no numbering or punctuation: `"0. Tabla de Contenidos"` → ..."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"\s+", " ", re.sub(r"^[\d.\s]+|[^\w\s]", " ", ascii_text)).strip()


# --- choosing sections --------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    section: Section
    key: tuple[str, str]  # (doc_path, heading_path JSON): how the history names a section


_SELECT = (
    "SELECT c.id, d.path, c.heading_path, c.start_line, c.end_line, c.text FROM chunks c "
    "JOIN documents d ON d.id = c.document_id "
    "WHERE c.token_count BETWEEN ? AND ? "
)
_UNDER_A_HEADING = (
    "AND json_array_length(c.heading_path) >= 2 "  # not the text before the first one
)


def _fold(text: str) -> str:
    plain = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in plain if not unicodedata.combining(ch))


_COGNATE = (("th", "t"), ("ph", "f"), ("y", "i"))


def _spelled(text: str) -> str:
    """Folded, with the spellings English and Spanish part ways on made alike: «matemática» and
    «mathematical» both become «matematical», «analisis» and «analysis» both «analisis»."""
    folded = _fold(text)
    for old, new in _COGNATE:
        folded = folded.replace(old, new)
    return folded


def topic_stems(topic: str) -> list[str]:
    """The topic's words cut to a stem that also matches the other language and the plural:
    «optimización» → «optimiz» (optimization too), «básico» → «basi» (basic), «AWS» whole."""
    words = [w for w in re.findall(r"\w+", _spelled(topic)) if w not in _TOPIC_STOPWORDS]
    return list(dict.fromkeys(w if len(w) <= 4 else w[: max(4, len(w) // 2 + 1)] for w in words))


def _matched(stems: Sequence[str], text: str, share: float) -> bool:
    """Whether the text writes at least `share` of the topic's words (a stem starting a word)."""
    if not stems:
        return True
    folded = _spelled(text)
    found = sum(re.search(rf"(?<!\w){re.escape(stem)}", folded) is not None for stem in stems)
    return found >= math.ceil(share * len(stems) - 1e-9)


def _named(names: Sequence[Sequence[str]], text: str, share: float) -> bool:
    """`_matched` for the topic as written or as translated: either one is enough."""
    return any(_matched(stems, text, share) for stems in names)


def candidate_sections(
    conn: sqlite3.Connection,
    config: Config,
    kind: str,
    value: str,
    retriever: Retriever | None = None,
    review: Sequence[tuple[str, str]] = (),
    english: str | None = None,
) -> list[Candidate]:
    """Content sections of a guide or chapter, a book, a folder, the whole library, a topic
    (through retrieval) or the review list (ADR-063). `english`: the topic translated, whose words
    also count as the topic's (ADR-060)."""
    p = config.practice
    bounds = (p.min_section_tokens, p.max_section_tokens)
    if kind == "book":  # a book is a folder; a book of one note is that note
        kind = "guide" if value.endswith(".md") else "folder"
    if kind == "chapter":
        kind = "guide"
    if kind == "library":
        rows = conn.execute(
            _SELECT + _UNDER_A_HEADING + "ORDER BY random() LIMIT ?",
            (*bounds, p.max_questions * p.library_pool),
        ).fetchall()
    elif kind == "guide":
        rows = conn.execute(
            _SELECT + _UNDER_A_HEADING + "AND d.path = ? ORDER BY c.id", (*bounds, value)
        ).fetchall()
    elif kind == "folder":
        prefix = scope_prefix(value)
        # a range on the path uses its index: a scan with substr() took 1.2 s over 700 000 chunks
        end = prefix[:-1] + chr(ord("/") + 1)  # the first path after every one under the folder
        rows = conn.execute(
            _SELECT + _UNDER_A_HEADING + "AND d.path >= ? AND d.path < ? ORDER BY c.id",
            (*bounds, prefix, end),
        ).fetchall()
    elif kind == "topic":
        if retriever is None:
            raise ValueError("a topic needs a retriever")
        ids = [h.chunk_id for h in retriever.search(value, p.max_questions * 3)]
        # the opening of a note, before its first heading, is where a topic is defined (ADR-062)
        rows = conn.execute(
            _SELECT + "AND c.id IN (SELECT value FROM json_each(?))", (*bounds, json.dumps(ids))
        ).fetchall()
        order = {cid: i for i, cid in enumerate(ids)}  # keep the retrieval's relevance order
        rows.sort(key=lambda r: order.get(r[0], len(order)))
        if p.topic_match > 0:  # ADR-060: a section must write the topic's words, not only rank
            names = [topic_stems(value)] + ([topic_stems(english)] if english else [])
            text = [f"{r[1]} {r[2]} {r[5]}" for r in rows]
            # the retrieval's best `topic_head` need only one of the words: they rank for the topic
            # as a whole («Optimization» in a maths book), while the tail drifts to the common word
            keep = [
                _named(names, t, p.topic_match) or (i < p.topic_head and _named(names, t, 0.01))
                for i, t in enumerate(text)
            ]
            rows = [r for r, k in zip(rows, keep, strict=True) if k]
            # …and those whose guide or heading write them come first (stable inside each group)
            rows.sort(key=lambda r: not _named(names, f"{r[1]} {r[2]}", p.topic_match))
    elif kind == "review":
        wanted = set(review)
        rows = []
        for path in sorted({path for path, _ in wanted}):
            rows += [
                r
                for r in conn.execute(
                    _SELECT + _UNDER_A_HEADING + "AND d.path = ? ORDER BY c.id", (*bounds, path)
                )
                if (r[1], r[2]) in wanted
            ]
    else:
        raise ValueError(f"unknown scope kind {kind!r}")
    out = []
    for _id, path, heading_json, start, end, text in rows:
        heading = tuple(json.loads(heading_json))
        if not is_content_heading(heading[-1]):
            continue
        out.append(Candidate(Section(path, heading, start, end, text), (path, heading_json)))
    return out


def choose_sections(
    candidates: Sequence[Candidate],
    latest: dict[tuple[str, str], tuple[str, str]],
    flagged: set[tuple[str, str]],
    count: int,
    rng: random.Random,
    keep_order: bool = False,
) -> list[Section]:
    """Sections failed last time first, then never asked, then the longest ago; one question per
    section. `keep_order` keeps the given order inside each group (a topic's relevance order)."""
    failed, fresh, done = [], [], []
    seen: set[tuple[str, str]] = set()
    for c in candidates:
        if c.key in seen or c.key in flagged:
            continue
        seen.add(c.key)
        last = latest.get(c.key)
        if last is None:
            fresh.append(c)
        elif last[0] != "correct":
            failed.append(c)
        else:
            done.append(c)
    if not keep_order:
        rng.shuffle(failed)
        rng.shuffle(fresh)
    done.sort(key=lambda c: latest[c.key][1])  # answered longest ago first
    return [c.section for c in (failed + fresh + done)[:count]]


def assign_kinds(kinds: Sequence[str], count: int, rng: random.Random) -> list[str]:
    """Spread the chosen kinds evenly over the questions, in a shuffled order."""
    if not kinds or any(k not in KINDS for k in kinds):
        raise ValueError(f"kinds must be some of {', '.join(KINDS)}")
    spread = [kinds[i % len(kinds)] for i in range(count)]
    rng.shuffle(spread)
    return spread


# --- generating ---------------------------------------------------------------------------------


def _complete(llm: LLMClient, system: str, user: str, max_tokens: int) -> str:
    return "".join(llm.stream((Message("system", system), Message("user", user)), max_tokens))


def _json_object(text: str) -> dict[str, object] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _text(value: object, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    return text if 0 < len(text) <= limit else None


def parse_question(reply: str, kind: str, truth: bool | None = None) -> QuestionContent | None:
    """The model's JSON for one question, validated; None if it is not usable."""
    data = _json_object(reply)
    if data is None:
        return None
    explanation = _text(data.get("explicacion", ""), _MAX_EXPLANATION) or ""
    if kind == "mc":
        question = _text(data.get("pregunta"), _MAX_TEXT)
        raw = data.get("opciones")
        correct = data.get("correcta")
        if isinstance(correct, str) and correct.strip().upper() in "ABCD" and correct.strip():
            correct = "ABCD".index(correct.strip().upper())
        if question is None or not isinstance(raw, list) or len(raw) != 4:
            return None
        unlettered = [_OPTION_LETTER_RE.sub("", o) if isinstance(o, str) else o for o in raw]
        options = [_text(o, _MAX_TEXT) for o in unlettered]
        clean = [o for o in options if o is not None]
        if len(clean) != 4 or len({_plain(o) for o in clean}) != 4:
            return None
        if not isinstance(correct, int) or isinstance(correct, bool) or not 0 <= correct < 4:
            return None
        return QuestionContent(question, explanation, options=tuple(clean), correct=correct)
    if kind == "tf":
        statement = _text(data.get("afirmacion"), _MAX_TEXT)
        value = data.get("verdadera")
        if isinstance(value, str):
            value = {"true": True, "verdadera": True, "false": False, "falsa": False}.get(
                _plain(value)
            )
        if statement is None or not isinstance(value, bool):
            return None
        if truth is not None and value != truth:
            return None  # asked for a true (or false) statement and got the other
        return QuestionContent(statement, explanation, truth=value)
    if kind in ("open_self", "open_model"):
        question = _text(data.get("pregunta"), _MAX_TEXT)
        reference = _text(data.get("respuesta"), _MAX_EXPLANATION)
        if question is None or reference is None:
            return None
        return QuestionContent(question, explanation, reference=reference)
    raise ValueError(f"unknown question kind {kind!r}")


def generate_question(
    llm: LLMClient, config: Config, section: Section, kind: str, rng: random.Random
) -> QuestionContent | None:
    system = load_prompt("practice_system")
    fields = {"citation": section.citation, "section": section.text}
    truth = None
    if kind == "mc":
        # Small models put the right answer first. The position is chosen here instead of
        # shuffling afterwards, so an explanation that names options by letter stays true.
        user = load_prompt("practice_mc").format(**fields, letter=rng.choice("ABCD"))
    elif kind == "tf":
        truth = rng.random() < 0.5
        hint = (
            ""
            if truth
            else " Que el error sea concreto (un concepto, una relación o un dato cambiado), "
            "no un matiz discutible."
        )
        user = load_prompt("practice_tf").format(
            **fields,
            truth="verdadera" if truth else "falsa",
            truth_json="true" if truth else "false",
            false_hint=hint,
        )
    else:
        user = load_prompt("practice_open").format(**fields)
    for _ in range(2):  # one retry, then the section is skipped
        reply = _complete(llm, system, user, config.practice.question_max_tokens)
        content = parse_question(reply, kind, truth)
        if content is not None:
            return content
    return None


# --- grading open answers -----------------------------------------------------------------------


@dataclass(frozen=True)
class Grade:
    result: str  # correct | partial | wrong
    reason: str


def parse_grade(reply: str) -> Grade | None:
    data = _json_object(reply)
    if data is None:
        return None
    verdict = _VERDICTS.get(_plain(str(data.get("veredicto", ""))))
    reason = _text(data.get("razon", ""), _MAX_EXPLANATION) or ""
    return Grade(verdict, reason) if verdict else None


def grade_answer(
    llm: LLMClient, config: Config, question: StoredQuestion, answer: str
) -> Grade | None:
    """The model's grade of an open answer (experimental, ADR-020); None if none was usable."""
    user = load_prompt("practice_grade_user").format(
        citation=question.section.citation,
        section=question.section.text,
        question=question.content.text,
        reference=question.content.reference,
        answer=answer,
    )
    system = load_prompt("practice_grade_system")
    for _ in range(2):
        grade = parse_grade(_complete(llm, system, user, config.practice.grade_max_tokens))
        if grade is not None:
            return grade
    return None
