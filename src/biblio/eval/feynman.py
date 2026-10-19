"""Offline evaluation of «Explícamelo tú» (ADR-027): does it report the gaps it should, and only
those?

`eval/feynman.jsonl` is written by the user, one explanation per line:

    {"id": "f01", "concept": "análisis PESTEL", "explanation": "…",
     "missing": ["ideas of the notes the explanation leaves out"],
     "unsupported": ["claims of the explanation the notes do not back or contradict"]}

A reported point counts as one of the expected ones when their embeddings are close
(`feynman.eval_match_similarity`), or, for claims, when one is written inside the other.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from biblio.config import Config
from biblio.eval.runner import _git_state
from biblio.generate.llm import LLMClient
from biblio.maps.export import numbered_fragments
from biblio.practice.feynman import check_claims, check_coverage, contains, extract_ideas
from biblio.retrieve.types import Retriever

RUNS_FILE = "feynman_runs.jsonl"
Embed = Callable[[Sequence[str]], np.ndarray]


class FeynmanSetError(ValueError):
    pass


@dataclass(frozen=True)
class Item:
    id: str
    concept: str
    explanation: str
    missing: tuple[str, ...]
    unsupported: tuple[str, ...]


def load_feynman_set(path: Path) -> list[Item]:
    if not path.is_file():
        raise FeynmanSetError(
            f"No evaluation set at {path}. Write it first (see eval/feynman.example.jsonl)."
        )
    items: list[Item] = []
    seen: set[str] = set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            item = Item(
                str(raw["id"]),
                str(raw["concept"]),
                str(raw["explanation"]),
                tuple(str(x) for x in raw.get("missing", [])),
                tuple(str(x) for x in raw.get("unsupported", [])),
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise FeynmanSetError(f"{path}:{n}: {exc}") from exc
        if item.id in seen:
            raise FeynmanSetError(f"{path}:{n}: repeated id {item.id!r}")
        seen.add(item.id)
        items.append(item)
    if not items:
        raise FeynmanSetError(f"{path} is empty")
    return items


def match_points(
    expected: Sequence[str],
    reported: Sequence[str],
    embed: Embed,
    threshold: float,
    literal: bool = False,
) -> list[tuple[int, int]]:
    """Pairs (expected, reported), each used once, most similar first. With `literal`, a point
    written inside the other also matches (claims are quoted from the same explanation)."""
    if not expected or not reported:
        return []
    a = np.asarray(embed(list(expected)), dtype=np.float32)
    b = np.asarray(embed(list(reported)), dtype=np.float32)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    sim = a @ b.T
    if literal:
        for i, e in enumerate(expected):
            for j, r in enumerate(reported):
                if contains(e, r) or contains(r, e):
                    sim[i, j] = max(sim[i, j], 1.0)
    pairs: list[tuple[int, int]] = []
    used_e: set[int] = set()
    used_r: set[int] = set()
    order = sorted(
        ((float(sim[i, j]), i, j) for i in range(len(expected)) for j in range(len(reported))),
        key=lambda p: (-p[0], p[1], p[2]),  # ties: the earliest first
    )
    for score, i, j in order:
        if score < threshold:
            break
        if i not in used_e and j not in used_r:
            pairs.append((i, j))
            used_e.add(i)
            used_r.add(j)
    return pairs


def _rate(part: int, whole: int) -> float | None:
    return round(part / whole, 3) if whole else None


def run_feynman_eval(
    config: Config,
    retriever: Retriever,
    llm: LLMClient,
    embed: Embed,
    items: Sequence[Item],
    tag: str,
    progress: Callable[[Item, dict[str, Any]], None] = lambda _i, _r: None,
) -> dict[str, Any]:
    cfg = config.feynman
    rows: list[dict[str, Any]] = []
    for item in items:
        t0 = time.perf_counter()
        hits = retriever.search(item.concept, cfg.fragments * 2)
        fragments = numbered_fragments(hits, cfg.fragments, cfg.max_fragment_chars)
        ideas = extract_ideas(llm, cfg, item.concept, fragments) if fragments else []
        coverage = (
            check_coverage(llm, cfg, item.concept, ideas, item.explanation) if ideas else None
        )
        claims = (
            check_claims(llm, cfg, item.concept, fragments, item.explanation) if ideas else None
        )
        elapsed = time.perf_counter() - t0
        gaps = [
            i.text for i, c in zip(ideas, coverage or [], strict=False) if c.verdict == "missing"
        ]
        flagged = [c.text for c in claims or [] if c.verdict != "supported"]
        found_gaps = match_points(item.missing, gaps, embed, cfg.eval_match_similarity)
        found_claims = match_points(
            item.unsupported, flagged, embed, cfg.eval_match_similarity, literal=True
        )
        row = {
            "id": item.id,
            "seconds": round(elapsed, 1),
            "ideas": len(ideas),
            "failed": coverage is None,
            "expected_missing": len(item.missing),
            "reported_missing": gaps,
            "found_missing": len(found_gaps),
            "expected_unsupported": len(item.unsupported),
            "reported_unsupported": flagged,
            "found_unsupported": len(found_claims),
            "not_found": [
                item.missing[i]
                for i in range(len(item.missing))
                if i not in {p[0] for p in found_gaps}
            ]
            + [
                item.unsupported[i]
                for i in range(len(item.unsupported))
                if i not in {p[0] for p in found_claims}
            ],
        }
        rows.append(row)
        progress(item, row)

    def total(key: str) -> int:
        return sum(len(r[key]) if isinstance(r[key], list) else int(r[key]) for r in rows)

    seconds = sorted(r["seconds"] for r in rows)
    commit, dirty = _git_state(config.root)
    return {
        "tag": tag,
        "date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": commit,
        "dirty": dirty,
        "model": config.llm.model,
        "n_items": len(rows),
        "metrics": {
            # of the gaps the user expected, how many were reported
            "missing_recall": _rate(total("found_missing"), total("expected_missing")),
            # of the gaps reported, how many the user did not expect (false alarms)
            "missing_extra": _rate(
                total("reported_missing") - total("found_missing"), total("reported_missing")
            ),
            "unsupported_recall": _rate(total("found_unsupported"), total("expected_unsupported")),
            "unsupported_extra": _rate(
                total("reported_unsupported") - total("found_unsupported"),
                total("reported_unsupported"),
            ),
            "failed": sum(1 for r in rows if r["failed"]),
        },
        "latency_s": {
            "p50": round(statistics.median(seconds), 1),
            "p95": round(seconds[min(len(seconds) - 1, int(0.95 * len(seconds)))], 1),
        },
        "items": rows,
    }


def append_feynman_run(results_dir: Path, record: dict[str, Any]) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / RUNS_FILE
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path
