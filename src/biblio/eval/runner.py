"""`biblio eval`: run the question set through a retriever and append one JSON line per run.

Format and protocol: `retrieval-eval` skill. Each run stores per-question ranks so two runs can be
compared question by question (`biblio compare`).
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from biblio.config import Config
from biblio.eval.metrics import (
    QuestionScore,
    all_found_at,
    matches_location,
    ndcg_at,
    normalize,
    recall_at,
    reciprocal_rank,
    score_question,
)
from biblio.eval.questions import Location, Question
from biblio.index.schema import read_meta
from biblio.retrieve.store import load_hits
from biblio.retrieve.types import Retriever

RUNS_FILE = "runs.jsonl"
GRANULARITIES = ("section", "file")


@dataclass(frozen=True)
class ExpectedCheck:
    problems: tuple[str, ...]  # block the run: the question set points to nothing
    warnings: tuple[str, ...]  # measurable, but worth knowing


_Docs = dict[str, tuple[int, str, str]]  # normalized path -> (id, path, sha256)


def _documents(conn: sqlite3.Connection) -> tuple[_Docs, dict[str, list[str]]]:
    rows = conn.execute("SELECT id, path, sha256 FROM documents").fetchall()
    docs = {normalize(path): (doc_id, path, sha) for doc_id, path, sha in rows}
    copies: dict[str, list[str]] = {}
    for _, path, sha in rows:
        copies.setdefault(sha, []).append(path)
    return docs, copies


def check_expected(conn: sqlite3.Connection, questions: Sequence[Question]) -> ExpectedCheck:
    """Validate every expected and alternative section against the index (skill: fix eval first)."""
    docs, _ = _documents(conn)
    problems: list[str] = []
    for q in questions:
        for exp in q.expected:
            for option in exp.options:
                doc = docs.get(normalize(option.path))
                if doc is None:
                    problems.append(f"{q.id}: file not in the index: {option.path}")
                    continue
                rows = conn.execute("SELECT id FROM chunks WHERE document_id = ?", (doc[0],))
                hits = load_hits(conn, [(cid, 0.0) for (cid,) in rows], "check")
                if not any(matches_location(h, option) for h in hits):
                    problems.append(f"{q.id}: heading {option.heading!r} not found in {doc[1]}")
    return ExpectedCheck(tuple(problems), ())


def with_identical_copies(
    conn: sqlite3.Connection, questions: Sequence[Question]
) -> list[Question]:
    """Accept byte-identical copies of an expected file as alternatives: same text, same answer."""
    docs, copies = _documents(conn)
    out = []
    for q in questions:
        expected = []
        for exp in q.expected:
            known = {(normalize(o.path), normalize(o.heading)) for o in exp.options}
            extra = []
            for option in exp.options:
                doc = docs.get(normalize(option.path))
                for twin in copies[doc[2]] if doc else []:
                    key = (normalize(twin), normalize(option.heading))
                    if key not in known:
                        known.add(key)
                        extra.append(Location(twin, option.heading))
            expected.append(replace(exp, alternatives=(*exp.alternatives, *extra)))
        out.append(replace(q, expected=tuple(expected)))
    return out


@dataclass(frozen=True)
class _Scored:
    question: Question
    section: QuestionScore
    file: QuestionScore
    latency_ms: float


def run_eval(
    config: Config,
    retriever: Retriever,
    conn: sqlite3.Connection,
    questions: Sequence[Question],
    tag: str,
) -> dict[str, Any]:
    ks = config.eval.ks
    depth = max(max(ks), config.retrieval.final_k)
    questions = with_identical_copies(conn, questions)
    scored: list[_Scored] = []
    for q in questions:
        if q.type == "absent":
            continue
        t0 = time.perf_counter()
        hits = retriever.search(q.question, depth)
        latency = (time.perf_counter() - t0) * 1000
        scored.append(
            _Scored(
                q,
                score_question(hits, q.expected, "section"),
                score_question(hits, q.expected, "file"),
                latency,
            )
        )
    if not scored:
        raise ValueError("No scorable questions: every question is of type 'absent'.")

    final_k = config.retrieval.final_k
    by_type: dict[str, Any] = {}
    for qtype in sorted({s.question.type for s in scored}):
        group = [s for s in scored if s.question.type == qtype]
        by_type[qtype] = {"n": len(group), **_aggregate(group, ks, final_k)}

    latencies = [s.latency_ms for s in scored]
    meta = read_meta(conn)
    commit, dirty = _git_state(config.root)
    return {
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "tag": tag,
        "commit": commit,
        "dirty": dirty,
        "mode": retriever.name,
        "config": {
            "dense_top_n": config.retrieval.dense_top_n,
            "bm25_top_n": config.retrieval.bm25_top_n,
            "rrf_k": config.retrieval.rrf_k,
            "rerank_candidates": getattr(retriever, "rerank_candidates", None),
            "reranker_device": getattr(retriever, "reranker_device", None),
            "rerank_max_tokens": config.reranker.max_tokens,
            "final_k": final_k,
            "identical_copies_accepted": True,
            **{
                k: meta.get(k)
                for k in (
                    "embedding_model",
                    "embedding_revision",
                    "index_dtype",
                    "max_seq_length",
                    "chunker_version",
                    "chunking",
                )
            },
        },
        "n_questions": len(scored),
        "n_absent": sum(1 for q in questions if q.type == "absent"),
        "metrics": _aggregate(scored, ks, final_k),
        "by_type": by_type,
        "failures": [
            s.question.id
            for s in scored
            if s.section.first_rank is None or s.section.first_rank > final_k
        ],
        "latency_ms": {
            "p50": round(statistics.median(latencies), 1),
            "p95": round(_percentile(latencies, 95), 1),
        },
        "per_question": {
            s.question.id: {
                "type": s.question.type,
                "section_ranks": list(s.section.expected_ranks),
                "file_ranks": list(s.file.expected_ranks),
            }
            for s in scored
        },
    }


def _aggregate(group: Sequence[_Scored], ks: Sequence[int], ndcg_k: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for gran in GRANULARITIES:
        scores = [s.section if gran == "section" else s.file for s in group]
        m: dict[str, float] = {}
        for k in ks:
            m[f"recall@{k}"] = _mean(recall_at(sc, k) for sc in scores)
        m["mrr"] = _mean(reciprocal_rank(sc) for sc in scores)
        m[f"ndcg@{ndcg_k}"] = _mean(ndcg_at(sc, ndcg_k) for sc in scores)
        multi = [sc for s, sc in zip(group, scores, strict=True) if s.question.type == "synthesis"]
        if multi:
            for k in ks:
                m[f"all@{k}"] = _mean(all_found_at(sc, k) for sc in multi)
        out[gran] = m
    return out


def _mean(values: Any) -> float:
    items = list(values)
    return round(sum(items) / len(items), 4) if items else 0.0


def _percentile(values: Sequence[float], pct: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(pct / 100 * len(ordered) + 0.5) - 1))
    return ordered[index]


def _git_state(root: Path) -> tuple[str | None, bool | None]:
    """Commit hash and whether the tree has uncommitted changes; (None, None) outside git."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, bool(status.strip())


def append_run(results_dir: Path, record: dict[str, Any]) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / RUNS_FILE
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def load_run(results_dir: Path, tag: str) -> dict[str, Any]:
    """The most recent run with `tag`."""
    path = results_dir / RUNS_FILE
    runs = []
    if path.is_file():
        runs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    matching: list[dict[str, Any]] = [r for r in runs if r.get("tag") == tag]
    if not matching:
        raise KeyError(f"No run tagged {tag!r} in {path}")
    return matching[-1]


@dataclass(frozen=True)
class Flip:
    question_id: str
    qtype: str
    before: int | None  # best section rank in the first run
    after: int | None


def flips(before: dict[str, Any], after: dict[str, Any], k: int) -> list[Flip]:
    """Questions whose first expected section entered or left the top-k between two runs."""

    def first(ranks: list[int | None]) -> int | None:
        found = [r for r in ranks if r is not None]
        return min(found) if found else None

    out = []
    a, b = before["per_question"], after["per_question"]
    for qid in sorted(set(a) & set(b)):
        ra, rb = first(a[qid]["section_ranks"]), first(b[qid]["section_ranks"])
        hit_a, hit_b = ra is not None and ra <= k, rb is not None and rb <= k
        if hit_a != hit_b:
            out.append(Flip(qid, b[qid]["type"], ra, rb))
    return out
