"""«Diccionario» (ADR-034): concepts, functions and formulas of a topic or a guide, each checked
against the literal text of the fragment it comes from.

One LLM call per fragment returns three lists. A concept needs its term and a supporting quote in
the fragment; a function its signature (which contains its name); a formula its LaTeX, spacing
ignored. Entries repeated across fragments are merged and keep every source.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace

from biblio.config import DictionaryConfig
from biblio.generate.ask import load_prompt
from biblio.generate.context import Fragment
from biblio.generate.llm import LLMClient
from biblio.maps.build import fold
from biblio.practice.feynman import contains
from biblio.practice.quiz import _complete, _json_object

KINDS = ("concept", "function", "formula")
_MAX_TERM = 120
_MAX_DETAIL = 600


@dataclass(frozen=True)
class Entry:
    kind: str  # concept | function | formula
    term: str  # the concept, the function's name or the formula's name
    text: str  # the definition, what the function does, what the formula computes
    detail: str  # the quote (concept), the signature (function) or the LaTeX (formula)
    sources: tuple[int, ...]  # Fragment.number of every fragment it was found in


@dataclass(frozen=True)
class Progress:
    processed: int  # fragments read so far
    skipped: int  # fragments whose reply was unusable twice
    entries: tuple[Entry, ...]


def squash(text: str) -> str:
    """Without any whitespace: how code and LaTeX are compared."""
    return "".join(text.split())


def _clean(value: object, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split()).strip(" «»\"'")
    return text if 0 < len(text) <= limit else None


def _raw(value: object, limit: int) -> str | None:
    """Code or LaTeX: kept as written, outer spaces and `$` removed."""
    if not isinstance(value, str):
        return None
    text = value.strip().strip("$").strip()
    return text if 0 < len(text) <= limit else None


def _named_in(text: str, term: str) -> bool:
    return f" {fold(term)} " in f" {fold(text)} " and bool(fold(term))


def parse_entries(
    reply: str, fragment: Fragment, per_list: int, definition_max: int
) -> list[Entry] | None:
    """The model's three lists for one fragment, checked against it; None if unusable."""
    data = _json_object(reply)
    if data is None:
        return None
    out: list[Entry] = []
    source = (fragment.number,)
    concepts = data.get("conceptos")
    for item in (concepts if isinstance(concepts, list) else [])[: per_list * 2]:
        if not isinstance(item, dict):
            continue
        term = _clean(item.get("termino"), _MAX_TERM)
        text = _clean(item.get("definicion"), definition_max)
        quote = _clean(item.get("cita"), _MAX_DETAIL)
        if (
            term
            and text
            and quote
            and _named_in(fragment.text, term)
            and contains(fragment.text, quote)
        ):
            out.append(Entry("concept", term, text, quote, source))
    functions = data.get("funciones")
    code = squash(fragment.text)
    for item in (functions if isinstance(functions, list) else [])[: per_list * 2]:
        if not isinstance(item, dict):
            continue
        name = _clean(item.get("nombre"), _MAX_TERM)
        signature = _raw(item.get("firma"), _MAX_DETAIL)
        text = _clean(item.get("descripcion"), definition_max)
        if not (name and signature and text):
            continue
        bare = name.split("(", 1)[0].strip()
        if bare and bare in signature and squash(signature) in code:
            out.append(Entry("function", bare, text, signature, source))
    formulas = data.get("formulas")
    for item in (formulas if isinstance(formulas, list) else [])[: per_list * 2]:
        if not isinstance(item, dict):
            continue
        latex = _raw(item.get("formula"), _MAX_DETAIL)
        name = _clean(item.get("nombre"), _MAX_TERM)
        text = _clean(item.get("descripcion"), definition_max)
        if latex and name and text and len(squash(latex)) >= 3 and squash(latex) in code:
            out.append(Entry("formula", name, text, latex, source))
    per_kind: dict[str, int] = {}
    kept = []
    for entry in out:
        per_kind[entry.kind] = per_kind.get(entry.kind, 0) + 1
        if per_kind[entry.kind] <= per_list:
            kept.append(entry)
    return kept


def entry_key(entry: Entry) -> tuple[str, str]:
    if entry.kind == "formula":
        return entry.kind, squash(entry.detail)
    return entry.kind, fold(entry.term)


def merge_entries(entries: Sequence[Entry], new: Sequence[Entry], limit: int) -> list[Entry]:
    """Add `new` to `entries`: a known entry gains the new sources; the first wording is kept.
    `limit` applies to each list (concepts, functions, formulas) on its own, so a long guide's
    concepts never crowd out the functions or formulas of its later sections."""
    merged = list(entries)
    index = {entry_key(e): i for i, e in enumerate(merged)}
    per_kind = {k: sum(1 for e in merged if e.kind == k) for k in KINDS}
    for entry in new:
        key = entry_key(entry)
        if key in index:
            old = merged[index[key]]
            sources = tuple(dict.fromkeys(old.sources + entry.sources))
            merged[index[key]] = replace(old, sources=sources)
        elif per_kind[entry.kind] < limit:
            index[key] = len(merged)
            merged.append(entry)
            per_kind[entry.kind] += 1
    return merged


def extract_entries(
    llm: LLMClient, cfg: DictionaryConfig, focus: str, fragment: Fragment
) -> list[Entry] | None:
    user = load_prompt("dictionary_user").format(
        focus=focus,
        citation=fragment.citation,
        section=fragment.text[: cfg.max_fragment_chars],
        max_entries=cfg.entries_per_fragment,
    )
    system = load_prompt("dictionary_system")
    for _ in range(2):  # one retry, then the fragment is skipped
        reply = _complete(llm, system, user, cfg.extract_max_tokens)
        entries = parse_entries(reply, fragment, cfg.entries_per_fragment, cfg.definition_max_chars)
        if entries is not None:
            return entries
    return None


def build_dictionary(
    llm: LLMClient,
    cfg: DictionaryConfig,
    focus: str,
    fragments: Sequence[Fragment],
    llm_turn: Callable[[], AbstractContextManager[object]] = nullcontext,
) -> Iterator[Progress]:
    """Read the fragments one by one and yield the dictionary as it grows."""
    entries: list[Entry] = []
    skipped = 0
    for i, fragment in enumerate(fragments, start=1):
        with llm_turn():
            found = extract_entries(llm, cfg, focus, fragment)
        if found is None:
            skipped += 1
        else:
            entries = merge_entries(entries, found, cfg.max_entries)
        yield Progress(i, skipped, tuple(entries))


# --- exports --------------------------------------------------------------------------------------

_HEADINGS = {"concept": "Conceptos", "function": "Funciones", "formula": "Fórmulas"}


def to_markdown(
    title: str, entries: Sequence[Entry], fragments: Sequence[Fragment], today: str
) -> str:
    by_number = {f.number: f for f in fragments}
    lines = [
        f"# Diccionario: {title}",
        "",
        f"> Generado por Biblioteca el {today} a partir de tus notas. Cada entrada cita con",
        "> [n] el fragmento de donde sale; la definición es del modelo y la cita es literal.",
        "",
    ]
    for kind in KINDS:
        chosen = [e for e in entries if e.kind == kind]
        if not chosen:
            continue
        lines += [f"## {_HEADINGS[kind]}", ""]
        for e in sorted(chosen, key=lambda e: fold(e.term)):
            cites = "".join(f"[{n}]" for n in e.sources)
            if kind == "concept":
                lines += [f"**{e.term}**: {e.text} {cites}", f"> «{e.detail}»", ""]
            elif kind == "function":
                lines += [f"**`{e.term}`**: {e.text} {cites}", "```", e.detail, "```", ""]
            else:
                lines += [f"**{e.term}**: {e.text} {cites}", "", f"$${e.detail}$$", ""]
    lines += ["## Fuentes", ""]
    used = sorted({n for e in entries for n in e.sources})
    for n in used:
        f = by_number.get(n)
        if f is not None:
            lines.append(f"{n}. {f.citation} (líneas {f.start_line}-{f.end_line})")
    return "\n".join(lines).rstrip() + "\n"


def _anki(text: str) -> str:
    return " ".join(text.replace("\t", " ").split())


def to_anki(title: str, entries: Sequence[Entry], fragments: Sequence[Fragment]) -> str:
    """Anki's plain-text import: tab-separated front and back, one card per line."""
    by_number = {f.number: f for f in fragments}
    rows = ["#separator:tab", "#html:true", f"#tags:biblioteca {_anki(title).replace(' ', '_')}"]
    for e in entries:
        where = "; ".join(by_number[n].citation for n in e.sources if n in by_number)
        if e.kind == "concept":
            front, back = e.term, f"{e.text}<br><i>«{e.detail}»</i>"
        elif e.kind == "function":
            front, back = f"<code>{e.detail}</code>", e.text
        else:
            front, back = f"{e.term}<br>\\({e.detail}\\)", e.text
        rows.append(f"{_anki(front)}\t{_anki(back)}<br><small>{_anki(where)}</small>")
    return "\n".join(rows) + "\n"
