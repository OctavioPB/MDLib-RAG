"""Answer evaluation with human review (`rag-prompting` skill: faithfulness, coverage, abstention).

`run` answers every question and writes, per run tag:
- `eval/review/answers-<tag>.json`: the answers and their automatic signals;
- `eval/review/answers-<tag>.md`: a review sheet with the cited fragments and grade lines.
The user fills `fidelidad: X de Y` (supported claims of total claims) and `cobertura: X de N`
(key points present). `score` reads both files and appends to `eval/results/answers.jsonl`.
Abstention is measured automatically: the model either said the fixed sentence or did not.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from biblio.eval.questions import Question
from biblio.generate.ask import Answer, Tutor

RESULTS_FILE = "answers.jsonl"
# "- fidelidad: 3 de 4", also "- **fidelidad: 3 de 4** — nota" (bold and trailing text tolerated).
_GRADE_RE = re.compile(
    r"^\s*-\s*\**\s*(fidelidad|cobertura)\s*:\s*(\?|\d+)\s+de\s+(\?|\d+)\b.*$", re.I
)
# Grades belong to "### Calificación <id>" or, in a hand-made sheet, to "## <id> (type) — …".
_SECTION_RE = re.compile(r"^(?:###\s+Calificación\s+(\S+)|##\s+(\S+)\s+\()")
_EXCERPT_CHARS = 400


def review_paths(review_dir: Path, tag: str) -> tuple[Path, Path]:
    return review_dir / f"answers-{tag}.json", review_dir / f"answers-{tag}.md"


def run_answers(
    tutor: Tutor,
    questions: Sequence[Question],
    on_question: Callable[[Question, Answer, float], None] | None = None,
) -> list[dict[str, Any]]:
    records = []
    for q in questions:
        t0 = time.perf_counter()
        answer = tutor.ask(q.question)
        seconds = time.perf_counter() - t0
        v = answer.validation
        records.append(
            {
                "id": q.id,
                "type": q.type,
                "question": q.question,
                "expected": [{"path": e.path, "heading": e.heading} for e in q.expected],
                "key_points": list(q.key_points),
                "answer": answer.text,
                "strategy": answer.strategy,
                "abstained": v.abstained,
                "grounded": v.grounded,
                "cited": list(v.cited),
                "invalid": list(v.invalid),
                "placeholders": v.placeholders,
                "fragments": [
                    {
                        "number": f.number,
                        "citation": f.citation,
                        "lines": [f.start_line, f.end_line],
                        "text": f.text,
                    }
                    for f in answer.context.fragments
                ],
                "notes": [{"guide": n.guide, "text": n.text} for n in answer.notes],
                "seconds": round(seconds, 2),
            }
        )
        if on_question:
            on_question(q, answer, seconds)
    return records


def review_markdown(records: Sequence[dict[str, Any]], tag: str) -> str:
    lines = [
        f"# Revisión de respuestas — `{tag}`",
        "",
        "Para cada respuesta, compara cada afirmación con el fragmento que cita y rellena:",
        "",
        "- `fidelidad: X de Y` — Y afirmaciones en la respuesta, X respaldadas por su cita.",
        "- `cobertura: X de N` — cuántos de los N puntos clave aparecen en la respuesta.",
        "",
        "Deja `?` lo que no califiques. La abstención se mide sola. Luego ejecuta",
        f"`uv run biblio ask-eval score {tag}`.",
        "",
    ]
    for r in records:
        status = (
            "se abstuvo"
            if r["abstained"]
            else ("con citas válidas" if r["grounded"] else "⚠ sin citas válidas")
        )
        lines += [
            f"## {r['id']} ({r['type']}) — {r['question']}",
            "",
            f"*{r['strategy']} · {len(r['fragments'])} fragmentos · {r['seconds']} s · {status}*",
            "",
        ]
        if r["expected"]:
            lines += [
                "Esperado: " + "; ".join(f"`{e['path']}` › {e['heading']}" for e in r["expected"]),
                "",
            ]
        lines += ["### Respuesta", ""] + [
            f"> {ln}" if ln else ">" for ln in r["answer"].splitlines()
        ]
        lines.append("")
        if r["invalid"]:
            lines += [f"⚠ Cita fragmentos inexistentes: {r['invalid']}", ""]
        if r.get("placeholders"):
            lines += [
                f"⚠ {r['placeholders']} citas sin número (como `[n]`): "
                "esas afirmaciones no citan nada.",
                "",
            ]
        cited = [f for f in r["fragments"] if f["number"] in r["cited"]]
        if cited:
            lines += ["### Fragmentos citados", ""]
            for f in cited:
                # The whole fragment: a claim can only be judged against everything it cites.
                excerpt = " ".join(f["text"].split())[:_EXCERPT_CHARS]
                fence = "`" * max(4, _longest_backtick_run(f["text"]) + 1)
                lines += [
                    f"**[{f['number']}] {f['citation']}** (líneas {f['lines'][0]}-{f['lines'][1]})",
                    "",
                    f"{excerpt}{'…' if len(excerpt) < len(' '.join(f['text'].split())) else ''}",
                    "",
                    "<details><summary>Texto completo del fragmento</summary>",
                    "",
                    f"{fence}text",
                    f["text"],
                    fence,
                    "",
                    "</details>",
                    "",
                ]
        if r["notes"]:
            lines += ["<details><summary>Notas del paso map</summary>", ""]
            for n in r["notes"]:
                lines += [f"**{n['guide']}**", "", n["text"], ""]
            lines += ["</details>", ""]
        lines += [f"### Calificación {r['id']}", "", "- fidelidad: ? de ?"]
        if r["key_points"]:
            lines.append(f"- cobertura: ? de {len(r['key_points'])}")
            lines += [f"  - [ ] {k}" for k in r["key_points"]]
        lines.append("")
    return "\n".join(lines) + "\n"


def _longest_backtick_run(text: str) -> int:
    return max((len(m.group(0)) for m in re.finditer(r"`+", text)), default=0)


def _grade_blocks(markdown: str) -> dict[str, list[str]]:
    """Lines of each "### Calificación <id>" block, up to the next question."""
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for line in markdown.splitlines():
        m = re.match(r"^###\s+Calificación\s+(\S+)\s*$", line)
        if m:
            current = m.group(1)
            blocks[current] = []
            continue
        if line.startswith("## "):
            current = None
        if current is not None:
            blocks[current].append(line)
    return blocks


def carry_over_grades(new_sheet: str, old_sheet: str) -> str:
    """Copy grades, notes and checked key points from an old sheet into a regenerated one."""
    old = _grade_blocks(old_sheet)
    out: list[str] = []
    skipping = False
    for line in new_sheet.splitlines():
        m = re.match(r"^###\s+Calificación\s+(\S+)\s*$", line)
        if m and m.group(1) in old:
            out.append(line)
            out.extend(old[m.group(1)])
            skipping = True
            continue
        if skipping and line.startswith("## "):
            skipping = False
        if not skipping:
            out.append(line)
    return "\n".join(out) + "\n"


def parse_grades(markdown: str) -> dict[str, dict[str, tuple[int, int]]]:
    """{question id: {"fidelidad": (x, y), "cobertura": (x, n)}}; `?` entries are left out."""
    grades: dict[str, dict[str, tuple[int, int]]] = {}
    current: str | None = None
    for line in markdown.splitlines():
        if m := _SECTION_RE.match(line):
            current = m.group(1) or m.group(2)
            continue
        if line.startswith("## "):
            current = None
        if current and (m := _GRADE_RE.match(line)):
            name, x, y = m.group(1).lower(), m.group(2), m.group(3)
            if x != "?" and y != "?" and int(y) > 0 and int(x) <= int(y):
                grades.setdefault(current, {})[name] = (int(x), int(y))
    return grades


def score(
    records: Sequence[dict[str, Any]], grades: dict[str, dict[str, tuple[int, int]]]
) -> dict[str, Any]:
    def summarize(group: Sequence[dict[str, Any]]) -> dict[str, Any]:
        answerable = [r for r in group if r["type"] != "absent"]
        absent = [r for r in group if r["type"] == "absent"]
        faith = [
            grades[r["id"]]["fidelidad"] for r in group if "fidelidad" in grades.get(r["id"], {})
        ]
        cover = [
            grades[r["id"]]["cobertura"]
            for r in answerable
            if "cobertura" in grades.get(r["id"], {})
        ]
        out: dict[str, Any] = {"n": len(group)}
        if faith:
            out["faithfulness"] = round(sum(x for x, _ in faith) / sum(y for _, y in faith), 3)
            out["fully_faithful"] = round(sum(x == y for x, y in faith) / len(faith), 3)
            out["graded_faithfulness"] = len(faith)
        if cover:
            out["coverage"] = round(sum(x / y for x, y in cover) / len(cover), 3)
            out["graded_coverage"] = len(cover)
        if absent:
            out["abstained_when_absent"] = round(
                sum(r["abstained"] for r in absent) / len(absent), 3
            )
        if answerable:
            out["false_abstention"] = round(
                sum(r["abstained"] for r in answerable) / len(answerable), 3
            )
            out["grounded"] = round(sum(r["grounded"] for r in answerable) / len(answerable), 3)
        out["invalid_citations"] = sum(len(r["invalid"]) for r in group)
        out["placeholder_citations"] = sum(r.get("placeholders", 0) for r in group)
        out["mean_seconds"] = round(sum(r["seconds"] for r in group) / len(group), 2)
        out["map_reduce_share"] = round(
            sum(r["strategy"] == "map-reduce" for r in group) / len(group), 3
        )
        return out

    by_type = {
        t: summarize([r for r in records if r["type"] == t])
        for t in sorted({r["type"] for r in records})
    }
    return {"metrics": summarize(records), "by_type": by_type}


def save_run(
    review_dir: Path, tag: str, meta: dict[str, Any], records: Sequence[dict[str, Any]]
) -> tuple[Path, Path]:
    review_dir.mkdir(parents=True, exist_ok=True)
    data_path, sheet_path = review_paths(review_dir, tag)
    data_path.write_text(
        json.dumps({"tag": tag, **meta, "records": list(records)}, ensure_ascii=False, indent=1),
        encoding="utf-8",
        newline="\n",
    )
    sheet_path.write_text(review_markdown(records, tag), encoding="utf-8", newline="\n")
    return data_path, sheet_path


def append_score(results_dir: Path, tag: str, meta: dict[str, Any], result: dict[str, Any]) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / RESULTS_FILE
    line = {
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "tag": tag,
        **meta,
        **result,
    }
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    return path
