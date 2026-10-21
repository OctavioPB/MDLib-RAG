from __future__ import annotations

import itertools
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from biblio.config import Config
from biblio.eval.metrics import score_question
from biblio.eval.questions import Expected, Location, Question
from biblio.eval.runner import (
    append_run,
    check_expected,
    flips,
    load_run,
    run_eval,
    with_identical_copies,
)
from biblio.index.indexer import index_library
from biblio.retrieve.bm25 import BM25Retriever, fts_query
from biblio.retrieve.dense import DenseRetriever
from biblio.retrieve.fusion import HybridRetriever, rrf
from biblio.retrieve.pipeline import build_retriever
from biblio.retrieve.rerank import RerankingRetriever, RerankPlan, passage, resolve_rerank_plan
from biblio.retrieve.store import IndexNotFoundError, QueryEmbedderMismatchError, open_index
from biblio.retrieve.types import Hit

from fakes import FakeEmbedder, WordCounter

GUIDES = {
    "estadistica/regresion.md": "# Regresión\n\n## Mínimos cuadrados\n\n"
    + "La regresión lineal estima coeficientes minimizando la suma de residuos al cuadrado. " * 3
    + "\n\n## Diagnóstico\n\n"
    + "Los residuos estudentizados detectan observaciones influyentes y heterocedasticidad. " * 3,
    "estadistica/bayes.md": "# Bayes\n\n## Teorema\n\n"
    + "El teorema de Bayes invierte probabilidades condicionadas usando la marginal. " * 3,
    "algebra/svd.md": "# SVD\n\n## Definición\n\n"
    + "La descomposición en valores singulares factoriza cualquier matriz real. " * 3,
    "copia/svd.md": "# SVD\n\n## Definición\n\n"
    + "La descomposición en valores singulares factoriza cualquier matriz real. " * 3,
}


@pytest.fixture
def indexed(config: Config, tmp_path: Path) -> Config:
    library = tmp_path / "library"
    for rel, text in GUIDES.items():
        (library / rel).parent.mkdir(parents=True, exist_ok=True)
        (library / rel).write_text(text, encoding="utf-8")
    cfg = replace(
        config,
        chunking=replace(config.chunking, min_tokens=5, target_tokens=40, max_tokens=60),
        paths=replace(config.paths, library=library, index_db=tmp_path / "data" / "index.db"),
        eval=replace(config.eval, results_dir=tmp_path / "results"),
    )
    index_library(
        cfg,
        library,
        full=True,
        dtype="float32",
        embedder_factory=FakeEmbedder,
        counter_factory=WordCounter,
    )
    return cfg


def _fake(_config: Config) -> FakeEmbedder:
    return FakeEmbedder()


QUESTIONS = [
    Question(
        "q1",
        "lookup",
        "regresion minimos cuadrados",
        (Expected("estadistica/regresion.md", "Mínimos cuadrados"),),
    ),
    Question("q2", "exact_term", "Bayes", (Expected("estadistica/bayes.md", "Teorema"),)),
    Question(
        "q3",
        "synthesis",
        "residuos y valores singulares",
        (
            Expected("estadistica/regresion.md", "Diagnóstico"),
            Expected("algebra/svd.md", "Definición"),
        ),
    ),
    Question("q4", "absent", "kubernetes en bare metal", ()),
]


def test_fts_query_quotes_and_deduplicates_words() -> None:
    assert fts_query("¿Qué es RRF k=60? ¿RRF?") == '"qué" OR "es" OR "rrf" OR "k" OR "60"'
    assert fts_query("¿?¡!") is None


def test_bm25_folds_accents_and_ranks(indexed: Config) -> None:
    retriever = BM25Retriever(open_index(indexed))
    hits = retriever.search("regresion", 5)
    assert hits[0].doc_path == "estadistica/regresion.md"
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert all(a.score >= b.score for a, b in itertools.pairwise(hits))
    assert retriever.search("¿?", 5) == []


def test_dense_finds_the_identical_chunk(indexed: Config) -> None:
    conn = open_index(indexed)
    breadcrumb, text = conn.execute(
        "SELECT breadcrumb, text FROM chunks c JOIN documents d ON d.id = c.document_id "
        "WHERE d.path = 'estadistica/bayes.md'"
    ).fetchone()
    hits = DenseRetriever(conn, FakeEmbedder()).search(f"{breadcrumb}\n\n{text}", 3)
    assert hits[0].doc_path == "estadistica/bayes.md"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert hits[0].citation == "estadistica/bayes.md › Teorema"


# --- folder scope (ADR-019) ---------------------------------------------------------------------


def test_scope_limits_dense_and_bm25_to_one_folder(indexed: Config) -> None:
    conn = open_index(indexed)
    query = "descomposición en valores singulares"
    unscoped = {h.doc_path for h in BM25Retriever(conn).search(query, 10)}
    assert {"algebra/svd.md", "copia/svd.md"} <= unscoped
    scoped = (BM25Retriever(conn, "algebra"), DenseRetriever(conn, FakeEmbedder(), "algebra"))
    for retriever in scoped:
        hits = retriever.search(query, 10)
        assert hits and {h.doc_path for h in hits} == {"algebra/svd.md"}, retriever.name
    assert DenseRetriever(conn, FakeEmbedder(), "nada").search(query, 10) == []
    # a folder name is never a prefix match on a sibling ("estadistic" is not "estadistica")
    assert BM25Retriever(conn, "estadistic").search("regresión", 10) == []


def test_library_folders_counts_every_level(indexed: Config, tmp_path: Path) -> None:
    from biblio.retrieve.store import library_folders

    folders = dict(library_folders(open_index(indexed)))
    assert folders == {"algebra": 1, "copia": 1, "estadistica": 2}


def test_missing_index_and_embedder_mismatch(indexed: Config, tmp_path: Path) -> None:
    missing = replace(indexed, paths=replace(indexed.paths, index_db=tmp_path / "none.db"))
    with pytest.raises(IndexNotFoundError):
        build_retriever(missing, "bm25")
    other = replace(indexed, embedding=replace(indexed.embedding, revision="0" * 40))
    with pytest.raises(QueryEmbedderMismatchError):
        build_retriever(other, "dense", _fake)
    build_retriever(other, "bm25")  # lexical search does not depend on the embedder
    with pytest.raises(ValueError, match="Unknown retrieval mode"):
        build_retriever(indexed, "bogus")


def test_check_expected(indexed: Config) -> None:
    bad = [
        Question("x1", "lookup", "q", (Expected("no/existe.md", "A"),)),
        Question("x2", "lookup", "q", (Expected("algebra/svd.md", "No existe"),)),
    ]
    checked = check_expected(open_index(indexed), [*QUESTIONS, *bad])
    assert checked.problems == (
        "x1: file not in the index: no/existe.md",
        "x2: heading 'No existe' not found in algebra/svd.md",
    )
    alt = Question(
        "x3",
        "lookup",
        "q",
        (Expected("algebra/svd.md", "Definición", (Location("estadistica/bayes.md", "Nope"),)),),
    )
    assert check_expected(open_index(indexed), [alt]).problems == (
        "x3: heading 'Nope' not found in estadistica/bayes.md",
    )


def test_run_eval_record(indexed: Config) -> None:
    conn = open_index(indexed)
    record = run_eval(indexed, BM25Retriever(conn), conn, QUESTIONS, "baseline-bm25")
    assert record["mode"] == "bm25"
    assert record["n_questions"] == 3 and record["n_absent"] == 1
    assert record["config"]["chunker_version"] == "1"
    assert set(record["by_type"]) == {"lookup", "exact_term", "synthesis"}
    assert record["metrics"]["section"]["recall@1"] >= 2 / 3  # q1 and q2 are rank 1
    assert record["metrics"]["file"]["recall@20"] == 1.0
    assert "all@20" in record["metrics"]["section"]
    assert record["per_question"]["q1"]["section_ranks"] == [1]
    assert record["latency_ms"]["p50"] >= 0
    json.dumps(record)  # serializable


def test_runs_round_trip_and_flips(indexed: Config) -> None:
    conn = open_index(indexed)
    a = run_eval(indexed, BM25Retriever(conn), conn, QUESTIONS, "a")
    b = json.loads(json.dumps(a))
    b["tag"] = "b"
    b["per_question"]["q2"]["section_ranks"] = [None]
    results = indexed.eval.results_dir
    append_run(results, a)
    append_run(results, b)
    assert load_run(results, "a")["tag"] == "a"
    (flip,) = flips(load_run(results, "a"), load_run(results, "b"), 8)
    assert (flip.question_id, flip.before, flip.after) == ("q2", 1, None)
    with pytest.raises(KeyError):
        load_run(results, "nope")


# --- CLI ----------------------------------------------------------------------------------------


def _questions_file(tmp_path: Path, rows: list[Question]) -> Path:
    path = tmp_path / "questions.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": q.id,
                    "type": q.type,
                    "question": q.question,
                    "expected": [{"path": e.path, "heading": e.heading} for e in q.expected],
                },
                ensure_ascii=False,
            )
            for q in rows
        ),
        encoding="utf-8",
    )
    return path


def test_cli_search_eval_compare(
    indexed: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import biblio.cli as cli

    monkeypatch.setattr(cli, "load_config", lambda: indexed)
    runner = CliRunner()
    questions = str(_questions_file(tmp_path, QUESTIONS))

    found = runner.invoke(cli.app, ["search", "teorema de bayes", "--mode", "bm25", "-k", "3"])
    assert found.exit_code == 0, found.output
    assert "estadistica/bayes.md" in found.output

    for tag in ("before", "after"):
        run = runner.invoke(
            cli.app, ["eval", "--mode", "bm25", "--tag", tag, "--questions", questions]
        )
        assert run.exit_code == 0, run.output
    assert len((indexed.eval.results_dir / "runs.jsonl").read_text("utf-8").splitlines()) == 2

    diff = runner.invoke(cli.app, ["compare", "before", "after"])
    assert diff.exit_code == 0, diff.output
    assert "No question entered or left" in diff.output


def test_cli_eval_refuses_broken_question_set(
    indexed: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import biblio.cli as cli

    monkeypatch.setattr(cli, "load_config", lambda: indexed)
    bad = _questions_file(tmp_path, [Question("x", "lookup", "q", (Expected("no.md", "A"),))])
    result = CliRunner().invoke(cli.app, ["eval", "--mode", "bm25", "--questions", str(bad)])
    assert result.exit_code == 2
    assert "file not in the index: no.md" in result.output
    assert not (indexed.eval.results_dir / "runs.jsonl").exists()


# --- Phase 4: fusion and reranking ------------------------------------------------------------


def _hit(chunk_id: int, rank: int, source: str = "x") -> Hit:
    return Hit(chunk_id, f"d{chunk_id}.md", ("G", f"S{chunk_id}"), "t", 1, 1, rank, 0.0, source)


class _Fixed:
    """A retriever that returns a fixed ranking and records the depth it was asked for."""

    def __init__(self, name: str, ids: list[int]) -> None:
        self.name = name
        self._ids = ids
        self.asked: list[int] = []
        self.closed = False

    def search(self, query: str, k: int) -> list[Hit]:
        self.asked.append(k)
        return [_hit(cid, r, self.name) for r, cid in enumerate(self._ids[:k], start=1)]

    def close(self) -> None:
        self.closed = True


def test_rrf_by_hand() -> None:
    a = [_hit(1, 1), _hit(2, 2), _hit(3, 3)]
    b = [_hit(3, 1), _hit(4, 2)]
    fused = rrf([a, b], k=60)
    assert [h.chunk_id for h in fused] == [3, 1, 2, 4]  # 2 and 4 tie: first seen wins
    assert fused[0].score == pytest.approx(1 / 63 + 1 / 61)
    assert [h.rank for h in fused] == [1, 2, 3, 4]
    assert {h.source for h in fused} == {"hybrid"}


def test_hybrid_asks_each_retriever_for_its_top_n(config: Config) -> None:
    dense, bm25 = _Fixed("dense", [1, 2, 3]), _Fixed("bm25", [3, 4])
    retriever = HybridRetriever(dense, bm25, config.retrieval)
    hits = retriever.search("q", 2)
    assert [h.chunk_id for h in hits] == [3, 1]
    assert dense.asked == [config.retrieval.dense_top_n]
    assert bm25.asked == [config.retrieval.bm25_top_n]
    retriever.close()
    assert dense.closed and bm25.closed


class _KeywordReranker:
    """Scores a passage by whether it contains the chunk id the test wants on top."""

    def __init__(self, favorite: str) -> None:
        self.favorite = favorite
        self.seen = 0
        self.closed = False

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.seen += len(passages)
        return [1.0 if self.favorite in p else 0.0 for p in passages]

    def close(self) -> None:
        self.closed = True


def test_reranker_reorders_the_candidates() -> None:
    base = _Fixed("hybrid", [1, 2, 3, 4, 5])
    reranker = _KeywordReranker("S4")
    retriever = RerankingRetriever(base, reranker, candidates=4)
    hits = retriever.search("q", 2)
    assert retriever.name == "hybrid-rerank"
    assert [h.chunk_id for h in hits] == [4, 1]
    assert [h.rank for h in hits] == [1, 2]
    assert base.asked == [4] and reranker.seen == 4  # only the candidates are scored
    assert passage(hits[0]) == "G › S4\n\nt"
    retriever.close()
    assert reranker.closed


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid", "dense-rerank", "hybrid-rerank"])
def test_every_mode_builds_and_searches(indexed: Config, mode: str) -> None:
    retriever = build_retriever(
        indexed,
        mode,
        _fake,
        lambda _c, _d: _KeywordReranker("Teorema"),
        lambda c: RerankPlan("cpu", c.reranker.cpu_candidates),
    )
    hits = retriever.search("teorema de bayes", 3)
    assert retriever.name == mode
    assert hits and hits[0].doc_path == "estadistica/bayes.md"
    retriever.close()


@pytest.mark.models
def test_real_reranker_prefers_the_relevant_passage(config: Config) -> None:
    from biblio.config import load_config
    from biblio.retrieve.rerank import CrossEncoderReranker

    reranker = CrossEncoderReranker(load_config(), "cpu")
    relevant, unrelated = reranker.score(
        "¿Qué es la fusión de rankings recíproca?",
        [
            "RRF suma 1/(k + rango) de cada lista ordenada para combinarlas.",
            "La receta de pan requiere harina, agua, sal y levadura.",
        ],
    )
    assert relevant > unrelated


@pytest.mark.parametrize(
    ("device", "fallback", "cuda", "llm", "expected"),
    [
        ("cuda", "skip", True, False, ("cuda", 40, "")),
        ("cuda", "skip", False, False, (None, 0, "CUDA is not available")),
        ("cuda", "skip", True, True, (None, 0, "an LLM server is running")),
        ("cuda", "cpu", True, True, ("cpu", 10, "an LLM server is running")),
        ("cpu", "skip", True, False, ("cpu", 10, "")),
    ],
)
def test_rerank_plan_prefers_a_free_gpu(
    config: Config,
    device: str,
    fallback: str,
    cuda: bool,
    llm: bool,
    expected: tuple[str | None, int, str],
) -> None:
    cfg = replace(
        config,
        reranker=replace(config.reranker, device=device, fallback=fallback, cpu_candidates=10),
        retrieval=replace(config.retrieval, rerank_candidates=40),
    )
    plan = resolve_rerank_plan(cfg, cuda_available=lambda: cuda, llm_running=lambda: llm)
    assert (plan.device, plan.candidates) == expected[:2]
    assert plan.fallback_reason.startswith(expected[2])


def test_rerank_plan_probes_the_real_llm_port(config: Config) -> None:
    import socket

    with socket.create_server(("127.0.0.1", 0)) as server:
        port = server.getsockname()[1]
        cfg = replace(config, llm=replace(config.llm, base_url=f"http://127.0.0.1:{port}/v1"))
        plan = resolve_rerank_plan(cfg, cuda_available=lambda: True)
    assert plan.device is None and "LLM server" in plan.fallback_reason


def test_eval_records_the_reranker_device(indexed: Config) -> None:
    retriever = build_retriever(
        indexed,
        "hybrid-rerank",
        _fake,
        lambda _c, _d: _KeywordReranker("Teorema"),
        lambda c: RerankPlan("cpu", 7, "test"),
    )
    record = run_eval(indexed, retriever, open_index(indexed), QUESTIONS, "t")
    assert record["config"]["reranker_device"] == "cpu"
    assert record["config"]["rerank_candidates"] == 7


def test_reranker_leaves_hits_beyond_the_candidates_in_first_stage_order() -> None:
    base = _Fixed("hybrid", [1, 2, 3, 4, 5, 6])
    reranker = _KeywordReranker("S3")
    hits = RerankingRetriever(base, reranker, candidates=3).search("q", 5)
    assert [h.chunk_id for h in hits] == [3, 1, 2, 4, 5]
    assert reranker.seen == 3
    assert [h.source for h in hits] == ["hybrid-rerank"] * 3 + ["hybrid"] * 2


def test_skipped_reranker_returns_the_plain_first_stage(indexed: Config) -> None:
    retriever = build_retriever(
        indexed,
        "hybrid-rerank",
        _fake,
        lambda _c, _d: pytest.fail("the reranker must not be loaded"),
        lambda _c: RerankPlan(None, 0, "LLM running"),
    )
    assert retriever.name == "hybrid"
    assert retriever.search("teorema de bayes", 3)[0].doc_path == "estadistica/bayes.md"


def test_unknown_fallback_is_a_config_error(project_copy: Path) -> None:
    from biblio.config import ConfigError, load_config

    (project_copy / "config" / "local.toml").write_text('[reranker]\nfallback = "gpu"\n', "utf-8")
    with pytest.raises(ConfigError, match="fallback"):
        load_config(project_copy)


def test_identical_copies_count_as_hits(indexed: Config) -> None:
    conn = open_index(indexed)
    (q,) = with_identical_copies(
        conn, [Question("c", "lookup", "q", (Expected("algebra/svd.md", "Definición"),))]
    )
    assert q.expected[0].alternatives == (Location("copia/svd.md", "Definición"),)
    # Only the copy is retrieved: still a hit.
    copy_only = [
        h
        for h in BM25Retriever(conn).search("valores singulares", 10)
        if h.doc_path == "copia/svd.md"
    ]
    assert score_question(copy_only, q.expected, "section").first_rank is not None
    record = run_eval(indexed, BM25Retriever(conn), conn, QUESTIONS, "copies")
    assert record["config"]["identical_copies_accepted"] is True


def test_cli_candidates_writes_a_review_file(
    indexed: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import biblio.cli as cli

    missing = Question(
        "m1", "lookup", "valores singulares", (Expected("estadistica/bayes.md", "Teorema"),)
    )
    questions = _questions_file(tmp_path, [*QUESTIONS, missing])
    cfg = replace(indexed, eval=replace(indexed.eval, questions=questions))
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    runner = CliRunner()
    assert runner.invoke(cli.app, ["eval", "--mode", "bm25", "--tag", "r"]).exit_code == 0

    result = runner.invoke(cli.app, ["candidates", "r", "--top", "2"])
    assert result.exit_code == 0, result.output
    review = (cfg.eval.results_dir.parent / "review" / "r.md").read_text(encoding="utf-8")
    assert "## m1 (lookup)" in review
    assert '{"path": "algebra/svd.md", "heading": "Definición"}' in review
    assert questions.read_text(encoding="utf-8").count("alternatives") == 0  # never edited


# --- tables of contents and the heading weight (ADR-044) ----------------------------------------


def _nav_hit(rank: int, *heading: str) -> Hit:
    return Hit(rank, "g.md", ("Guía", *heading), "texto", 1, 2, rank, 1.0 / rank, "fake")


class _NavListed:
    name = "fake"

    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.asked: list[int] = []

    def search(self, query: str, k: int) -> list[Hit]:
        self.asked.append(k)
        return self.hits[:k]

    def close(self) -> None:
        pass


def test_tables_of_contents_are_not_results() -> None:
    from biblio.retrieve.navigation import SkipNavigation, is_navigation

    assert is_navigation(("Guía", "0. Tabla de Contenidos"))
    assert is_navigation(("Guide", "Table of contents"))
    assert is_navigation(("Guía", "Índice"))
    assert not is_navigation(("Índice",))  # a note titled so is content, not its own index
    assert not is_navigation(("Guía", "Índice de masa corporal"))
    base = _NavListed(
        [
            _nav_hit(1, "Índice"),
            _nav_hit(2, "Sección A"),
            _nav_hit(3, "Sección B"),
            _nav_hit(4, "C"),
        ]
    )
    hits = SkipNavigation(base, margin=2).search("q", 2)
    assert [h.heading_path[-1] for h in hits] == ["Sección A", "Sección B"]
    assert [h.rank for h in hits] == [1, 2] and base.asked == [4]  # k + margin asked for


def test_the_heading_weight_is_used_and_validated(indexed: Config) -> None:
    from biblio.config import ConfigError, _validate

    conn = open_index(indexed)
    for weight in (0.5, 1.0, 2.0):
        hits = BM25Retriever(conn, heading_weight=weight).search("regresion", 5)
        assert hits[0].doc_path == "estadistica/regresion.md"
    with pytest.raises(ConfigError):
        _validate(replace(indexed, retrieval=replace(indexed.retrieval, bm25_heading_weight=0)))


def test_hits_are_grouped_by_note_in_rank_order() -> None:
    from biblio.retrieve.grouping import group_by_note

    def hit(rank: int, path: str) -> Hit:
        return Hit(rank, path, ("G", f"s{rank}"), "t", 1, 2, rank, 1.0 / rank, "x")

    hits = [hit(1, "a.md"), hit(2, "b.md"), hit(3, "a.md"), hit(4, "c.md"), hit(5, "b.md")]
    groups = group_by_note(hits, 2)
    assert [[h.rank for h in g] for g in groups] == [[1, 3], [2, 5]]  # c.md: over the 2 notes
    assert group_by_note([], 3) == []
