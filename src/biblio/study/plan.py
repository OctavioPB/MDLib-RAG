"""Study plan per folder (ADR-028): an order of its guides proposed by the LLM, and their states.

The model only ever names guides by number; the reply is checked so the plan holds every guide of
the folder exactly once. Folders bigger than `plan.max_guides` are ordered by module (first-level
subfolder) and then inside each module.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass

from biblio.config import Config, PlanConfig
from biblio.generate.ask import load_prompt
from biblio.generate.llm import LLMClient
from biblio.practice.quiz import _complete, _json_object, candidate_sections, is_content_heading
from biblio.retrieve.store import scope_prefix

STATES = ("unread", "read", "practiced", "mastered")


@dataclass(frozen=True)
class Guide:
    doc_path: str
    title: str
    headings: tuple[str, ...]  # its section titles, in order
    links: frozenset[str] = frozenset()  # file names it links to


@dataclass(frozen=True)
class PlanItem:
    doc_path: str
    reason: str
    module: str  # subfolder the guide was ordered in; "" for a folder ordered in one go
    placed: bool  # False: the model left it out and it was appended


@dataclass(frozen=True)
class PlanProgress:
    done: int  # orderings finished
    total: int
    label: str  # what was just ordered


def folder_guides(conn: sqlite3.Connection, folder: str, headings: int) -> list[Guide]:
    """Every indexed guide under `folder`, at any depth, by title, with its first section titles."""
    prefix = scope_prefix(folder)
    docs = conn.execute(
        "SELECT id, path, title FROM documents WHERE substr(path, 1, ?) = ? ORDER BY title, path",
        (len(prefix), prefix),
    ).fetchall()
    guides = []
    for doc_id, path, title in docs:
        seen: list[str] = []
        links: set[str] = set()
        for heading_json, links_json in conn.execute(
            "SELECT heading_path, links FROM chunks WHERE document_id = ? ORDER BY ordinal",
            (doc_id,),
        ):
            links |= {link.rsplit("/", 1)[-1] for link in json.loads(links_json)}
            path_parts = json.loads(heading_json)
            heading = path_parts[1] if len(path_parts) >= 2 else ""
            if heading and is_content_heading(heading) and heading not in seen:
                seen.append(heading)
        guides.append(Guide(str(path), str(title), tuple(seen[:headings]), frozenset(links)))
    return guides


# --- ordering -----------------------------------------------------------------------------------


def parse_order(reply: str, count: int, reason_max: int) -> list[tuple[int, str]] | None:
    """[(0-based index, reason)] in the model's order: unknown numbers and repeats dropped.
    None if the reply is unusable."""
    data = _json_object(reply)
    raw = data.get("orden") if data is not None else None
    if not isinstance(raw, list):
        return None
    order: list[tuple[int, str]] = []
    seen: set[int] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        number = item.get("guia")
        if isinstance(number, str) and number.strip().isdigit():
            number = int(number.strip())
        if not isinstance(number, int) or isinstance(number, bool) or not 1 <= number <= count:
            continue
        if number - 1 in seen:
            continue
        reason = item.get("razon", "")
        text = " ".join(reason.split())[:reason_max] if isinstance(reason, str) else ""
        seen.add(number - 1)
        order.append((number - 1, text))
    return order or None


def complete_order(
    order: Sequence[tuple[int, str]], titles: Sequence[str]
) -> list[tuple[int, str, bool]]:
    """The model's order, then what it left out by title: (index, reason, placed by the model)."""
    placed = {i for i, _ in order}
    missing = sorted((i for i in range(len(titles)) if i not in placed), key=lambda i: titles[i])
    return [(i, r, True) for i, r in order] + [(i, "", False) for i in missing]


def order_entries(
    llm: LLMClient, cfg: PlanConfig, folder: str, entries: Sequence[tuple[str, Sequence[str]]]
) -> list[tuple[int, str, bool]]:
    """Order (title, details) entries: guides with their sections, or modules with their guides."""
    if len(entries) < 2:
        return [(i, "", True) for i in range(len(entries))]
    lines = []
    for n, (title, details) in enumerate(entries, start=1):
        shown = "; ".join(details) if details else "sin secciones"
        lines.append(f"{n}. {title} — {shown}")
    user = load_prompt("plan_order_user").format(
        folder=folder, guides="\n".join(lines), count=len(entries)
    )
    system = load_prompt("plan_order_system")
    best: list[tuple[int, str]] = []
    for _ in range(2):  # one retry when the reply is unusable or leaves guides out
        order = parse_order(
            _complete(llm, system, user, cfg.order_max_tokens), len(entries), cfg.reason_max_chars
        )
        if order and len(order) > len(best):
            best = order
        if len(best) == len(entries):
            break
    return complete_order(best, [title for title, _ in entries])


def _module_of(doc_path: str, folder: str) -> str:
    rest = doc_path[len(scope_prefix(folder)) :]
    return rest.split("/", 1)[0] if "/" in rest else ""


def _groups(cfg: PlanConfig, folder: str, guides: Sequence[Guide]) -> list[tuple[str, list[Guide]]]:
    """One group when the folder is small enough for one prompt; else one per first-level
    subfolder, with the guides directly in the folder as module ""."""
    if len(guides) <= cfg.max_guides:
        return [("", list(guides))]
    modules: dict[str, list[Guide]] = {}
    for g in guides:
        modules.setdefault(_module_of(g.doc_path, folder), []).append(g)
    return sorted(modules.items())


def _orderings(cfg: PlanConfig, folder: str, guides: Sequence[Guide]) -> int:
    groups = _groups(cfg, folder, guides)
    if len(groups) == 1:
        name, members = groups[0]
        if name:  # everything is in one subfolder: go down
            return _orderings(cfg, f"{folder}/{name}", members)
        return 1 if len(members) <= cfg.max_guides else 0
    return 1 + sum(
        _orderings(cfg, f"{folder}/{name}", members)
        if name
        else int(len(members) <= cfg.max_guides)
        for name, members in groups
    )


def plan_folder(
    llm: LLMClient, cfg: PlanConfig, folder: str, guides: Sequence[Guide]
) -> Iterator[PlanProgress | list[PlanItem]]:
    """Yields progress after each LLM ordering, and the finished plan last."""
    counter = [0, max(_orderings(cfg, folder, guides), 1)]
    items = yield from _plan(llm, cfg, folder, guides, "", counter)
    yield items


def _plan(
    llm: LLMClient,
    cfg: PlanConfig,
    folder: str,
    guides: Sequence[Guide],
    module: str,
    counter: list[int],
) -> Generator[PlanProgress, None, list[PlanItem]]:
    groups = _groups(cfg, folder, guides)
    if len(groups) == 1 and groups[0][0]:  # everything in one subfolder: go down
        name, members = groups[0]
        return (
            yield from _plan(llm, cfg, f"{folder}/{name}", members, _join(module, name), counter)
        )
    if len(groups) == 1:
        return (yield from _flat(llm, cfg, folder, groups[0][1], module, counter))
    entries = [
        (
            name or f"{folder.rsplit('/', 1)[-1]} (guías sueltas)",
            [g.title for g in members][: cfg.headings_per_guide],
        )
        for name, members in groups
    ]
    ranked = order_entries(llm, cfg, f"{folder} (módulos)", entries)
    counter[0] += 1
    yield PlanProgress(counter[0], counter[1], "módulos")
    items: list[PlanItem] = []
    for i, _, _ in ranked:
        name, members = groups[i]
        if name:
            items += yield from _plan(
                llm, cfg, f"{folder}/{name}", members, _join(module, name), counter
            )
        else:
            items += yield from _flat(llm, cfg, folder, members, module, counter)
    return items


def _flat(
    llm: LLMClient,
    cfg: PlanConfig,
    folder: str,
    guides: Sequence[Guide],
    module: str,
    counter: list[int],
) -> Generator[PlanProgress, None, list[PlanItem]]:
    """One ordering of the guides themselves; too many for one prompt keep their title order."""
    if len(guides) > cfg.max_guides:
        return [PlanItem(g.doc_path, "", module, False) for g in guides]
    ranked = order_entries(
        llm, cfg, folder, [(_entry_title(g, guides), g.headings) for g in guides]
    )
    counter[0] += 1
    yield PlanProgress(counter[0], counter[1], module or folder)
    return [PlanItem(guides[i].doc_path, reason, module, placed) for i, reason, placed in ranked]


def is_overview(guide: Guide, guides: Sequence[Guide]) -> bool:
    """A guide that links to at least half of the others (and two or more): the folder's index."""
    others = {g.doc_path.rsplit("/", 1)[-1] for g in guides if g is not guide}
    linked = len(others & guide.links)
    return linked >= 2 and 2 * linked >= len(others)


def _entry_title(guide: Guide, guides: Sequence[Guide]) -> str:
    return (
        f"{guide.title} [guía general: enlaza a las demás]"
        if is_overview(guide, guides)
        else guide.title
    )


def _join(module: str, name: str) -> str:
    return f"{module}/{name}" if module else name


# --- states ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GuideState:
    state: str  # unread | read | practiced | mastered
    sections: int  # content sections
    practiced: int  # of them, with an answer in the practice history
    recall: float | None  # mean recall today of the practised ones


def guide_states(
    conn: sqlite3.Connection,
    config: Config,
    doc_paths: Sequence[str],
    reads: set[str],
    recall: Mapping[tuple[str, str], float],
) -> dict[str, GuideState]:
    """The state of each guide from its reading mark and the practice schedule (ADR-028)."""
    practised_docs = {path for path, _ in recall}
    out = {}
    for path in doc_paths:
        keys = {c.key for c in candidate_sections(conn, config, "guide", path)}
        done = [recall[k] for k in keys if k in recall]
        mean = sum(done) / len(done) if done else None
        if (
            keys
            and mean is not None
            and len(done) / len(keys) >= config.plan.mastered_coverage
            and mean >= config.practice.mastered_at
        ):
            state = "mastered"
        elif path in practised_docs:
            state = "practiced"
        elif path in reads:
            state = "read"
        else:
            state = "unread"
        out[path] = GuideState(
            state, len(keys), len(done), round(mean, 2) if mean is not None else None
        )
    return out


def next_guide(order: Sequence[str], states: Mapping[str, GuideState]) -> tuple[str, str] | None:
    """(doc_path, why): the first guide not practised yet, else the first not mastered."""
    for path in order:
        if states[path].state in ("unread", "read"):
            return path, "next"
    for path in order:
        if states[path].state != "mastered":
            return path, "review"
    return None
