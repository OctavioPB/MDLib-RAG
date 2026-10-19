# PLAN.md — biblio-rag

States: `[ ]` pending · `[~]` in progress · `[x]` done · `[-]` dropped (with an ADR or a note).

**Active phase: 7**

Ordering principle: first a search engine without an LLM, measured. Generation is built on top of a
retrieval layer that has already proven itself (ADR-009).

---

## Phase 0 — Environment and guarantees

Goal: a repository that installs, downloads models once and proves it runs without network.

- [x] `pyproject.toml` with dependencies, working `uv sync`, `ruff` + `mypy` + `pytest` configured.
- [x] `config.py`: loads `config/default.toml` and merges `config/local.toml` over it if present; typed dataclasses.
- [x] `biblio models pull`: downloads embeddings and reranker into `models/` at a pinned revision (HF commit hash).
- [x] `biblio doctor`: reports CPU, RAM, GPU/VRAM (or Apple Silicon), model presence,
      `enable_load_extension` support, `sqlite-vec` loading and offline variables.
- [x] `conftest.py` with `pytest-socket` (`--disable-socket --allow-hosts=127.0.0.1,localhost`).
- [x] Complete ADR-011 (real hardware profile) with the output of `biblio doctor`.

**Exit criterion:** with the network disabled, `biblio doctor` finishes green.

## Phase 1 — Ingestion and chunking

Goal: turn every `.md` into chunks faithful to its structure. Use the `markdown-chunking` skill.

- [x] File discovery (configurable glob, exclusions, symlinks ignored).
- [x] Parser: YAML frontmatter → metadata; `markdown-it-py` tokens with start and end lines.
- [x] Heading-based chunker with `heading_path`, never splitting code blocks, tables, `$$…$$` formulas
      or ASCII diagrams.
- [x] Splitting of long sections by paragraphs and merging of tiny sections (thresholds in config).
- [x] Extraction of internal links (`[[wikilink]]` and relative `.md` links) stored as metadata (used in Phase 7).
- [x] `CHUNKER_VERSION` as a constant; changing it invalidates the index.
- [x] Fixtures: at least 8 real files representative of the library, plus pathological cases
      (no headings, empty heading, code containing `#`, setext headings, embedded HTML).
      10 real guides in `tests/fixtures/library/` (one per main folder, chosen by features) and
      19 pathological files in `tests/fixtures/pathological/`.
- [x] `biblio chunks <file>`: prints the chunks of a file for manual inspection.

**Exit criterion:** manual review of the chunks of 10 random files with no incorrect cuts. ✔ 2026-10-17 (Claude review + user sign-off).

## Phase 2 — Indexing

- [x] SQLite schema: `documents`, `chunks`, `chunks_fts` (FTS5, external content), `vec_chunks` (vec0, float[1024]), `index_meta`.
- [x] `Embedder` (Protocol) + bge-m3 implementation in batches, normalized embeddings: `index_device`/`index_dtype`
      for indexing (GPU fp32 by default, CPU fallback with a warning), `device` (CPU) for queries (ADR-012).
- [x] Refuse GPU indexing while the LLM server answers on `llm.base_url` (loopback probe), unless `--ignore-llm` (ADR-012).
- [x] Incremental indexing: sha256 per file; insert, update or delete in one transaction per document.
- [x] Version guards: if `embedding_model`, `embedding_revision`, `index_dtype` or `chunker_version` differ from `index_meta`, require `--full`.
      Also store the `[chunking]` thresholds: a `local.toml` override changes the chunks without bumping `CHUNKER_VERSION`.
- [x] Progress bar, total time and chunks/s reported at the end.

**Exit criterion:** full indexing without errors; a second `biblio index` with no changes takes seconds. ✔ 2026-10-17: 2,252 docs / 61,753 chunks in 38 min on GPU; integrity checks clean; no-change run 0.6 s.

## Phase 3 — Evaluation and baseline search

Evaluation comes before optimization. Use the `retrieval-eval` skill.

- [~] The user writes `eval/questions.jsonl`: 40–60 questions with the expected section.
      At least 30% of type `synthesis` (answer spread over 2+ guides) and 20% with exact terms.
      Include 3–5 questions whose answer is a term deep inside a large glossary table (e.g. the
      "Glosario" sections of `Data Engineering/Data-Engineering.md` or `GCP-Networking.md`), so the
      oversized-table question below can be measured.
- [x] Metrics: Recall@k (k = 1, 5, 8, 20), MRR, nDCG@8. File-level and section-level matching, reported separately.
- [x] `biblio eval` stores one JSONL line in `eval/results/` with date, commit, relevant config and metrics per question type.
- [x] Baseline 1: dense only. Baseline 2: BM25 only.
- [x] `biblio search` with readable output (rank, score, heading path, excerpt).

**Exit criterion:** baselines recorded. The numeric targets for Phase 4 are set from them (not before).

## Phase 4 — Hybrid retrieval and reranking

- [x] RRF fusion (k_rrf = 60) of dense top-N and BM25 top-N.
- [x] `Reranker` (Protocol) + bge-reranker-v2-m3 on CPU over the best `rerank_candidates`.
- [x] Measure each piece separately: dense → +BM25 → +RRF → +reranker. Record the four runs.
- [x] Measure p50/p95 latency of `search` and tune `rerank_candidates` if p95 > the config target.
- [~] Error analysis: list the failed questions and classify the cause (chunking, vocabulary, ambiguity, badly written eval).
- [ ] Oversized atomic units (decided 2026-10-17 to keep whole until measured): 77 chunks exceed the
      1,024-token embedding window, mostly glossary tables (up to 6,279 tokens), so dense search only
      sees their first 1,024 tokens. If the glossary questions fail, try splitting tables over
      `max_tokens` by row groups that repeat the header row; that supersedes part of ADR-003, so it
      needs a new ADR with before/after eval runs (rule 3) and a `CHUNKER_VERSION` bump.

**Targets (agreed with the user on 2026-10-17, section level, default mode `hybrid-rerank`):**
recall@8 ≥ 0.75, MRR ≥ 0.55, p95 latency ≤ 3 s (`retrieval.latency_p95_target_ms`).
Met by `v2-hybrid-rerank` (64 questions): recall@8 0.770, MRR 0.597, p95 1.85 s.

**Exit criterion:** Phase 3 targets reached or the deviation justified in an ADR. **At this point the search engine is done.**

## Phase 5 — Tutor (`biblio ask`)

Use the `memory-budget` and `rag-prompting` skills.

- [x] `LLMClient` (Protocol) with an HTTP client against an OpenAI-compatible API on `localhost`.
- [x] Choice of the concrete model according to ADR-011 and `memory-budget`; record an ADR.
- [x] Context assembly with a token budget (`context_budget_tokens`), fragments numbered `[1]..[n]`.
- [x] Prompt with mandatory citations and explicit abstention.
- [x] Post-validation: every cited `[n]` exists; no citations → answer flagged as ungrounded.
- [x] Map-reduce mode for `synthesis` questions when the evidence does not fit the budget.
      Manual opt-in only, off by default (ADR-017: measured faithfulness 0.58 vs 0.87 for the direct path).
- [x] Answer evaluation set (10–15 questions): faithfulness to sources, coverage and correct abstention, with human review.
      `eval/questionsv3.jsonl` (14 questions, user-graded `v3-off` and `v3-mr`). Open: a13 and a14 are labeled absent but the library covers them.
- [x] Streaming of the answer in the CLI.

## Phase 6 — Writing assistant (`biblio draft`)

- [x] Two-step flow: the model proposes an outline of sections from the evidence; the user approves or edits it; then each section is written in turn.
- [x] Per-section retrieval (query = section title + topic).
- [x] Retrieval of "sibling guides" to imitate structure and conventions.
- [x] Output to a `.md` file with a final block listing the sources used.
- [x] Explicit expectation: a draft to edit, not a finished guide (ADR-008).

## Phase 7 — Optional (each one requires an ADR before starting)

- [ ] Expansion through the internal link graph (neighbors of the top hits as extra candidates).
- [ ] Style LoRA on the local model for `draft` (only if Phase 6 shows a formatting problem, not a content one).
- [x] Local web interface or editor integration. (ADR-016: `biblio serve`)
- [x] Web UI v2, page **Conversar** (ADR-019): one thread, "Explicar" / "Solo ubicar", sources panel, settings, folder scope.
- [x] Web UI v2, page **Guías** (ADR-019): the `draft` flow in the browser.
- [x] Web UI v2, page **Practicar** (ADR-020): quizzes from library sections, local history.
- [x] Conversation history (ADR-023): stored in `data/conversations.db`; list, reopen, search, rename, delete, export.
- [x] One-step launcher (ADR-021): `Biblioteca.cmd` / `scripts/start.sh` → `biblio start`.
- [x] File watcher for automatic reindexing.
      `biblio watch`: stdlib polling of (mtime, size) every 30 s, 10 s settle; GPU if free, else CPU (ADR-018).


## Phase 8 — Concept maps (page `Mapas`)

Goal: a web page that, given a topic, draws a mind map of the related concepts in the user's library:
the topic in the centre, the main concepts around it, their sub-concepts, and labelled relations
between them. Every node and every relation cites the section it comes from (rule 2); nothing is
invented to fill the map. Requires ADR-022 before any code (Phase 7 rule).

Design to confirm in ADR-022:

1. **Evidence.** The default retrieval for the topic (the best 8–12 sections, as `ask` uses), plus the
   heading structure of the guides they come from. The internal link graph is too sparse to be the
   backbone (565 of 61,753 chunks have links, 2,816 links in total), but it can add links between guides.
2. **Extraction.** One LLM call per fragment returns JSON: the concepts it defines or uses, and
   relations between them (`{from, to, label}`, e.g. «es un tipo de», «se usa para», «depende de»).
   Checked in code: a concept must appear in its fragment's text (ignoring accents and case), and both
   ends of a relation must appear in the same fragment; anything else is dropped. One retry, then the
   fragment is skipped (the Practicar pattern, ADR-020).
3. **Merging.** Concepts found in several fragments are merged by normalized name and, for
   near-synonyms, by embedding similarity with the query embedder that is already loaded (threshold
   in config). A merged node keeps all its sources.
4. **Hierarchy, built in code.** Level 1: the concepts most related to the topic and present in the
   most fragments. Level 2: the rest, under the level-1 concept they relate to. Relations across branches
   are drawn as cross-links. The global structure is not left to the LLM: a 4B model is unreliable at
   it (ADR-017).
5. **Drawing.** Own SVG with a radial tree layout in plain JavaScript, no library (ADR-016). Pan and
   zoom; node size by number of sources; colour by guide; light and dark themes. A node opens its
   sources in the side panel; **Expandir** builds a sub-map of that concept; **Explicar** and
   **Practicar** go to the other pages with the concept as the topic. A nested text outline of the same
   map is the accessible alternative to the drawing.
6. **Export.** Markdown outline with citations, and an Obsidian Canvas file (`.canvas`, the open JSON
   Canvas format) saved to `drafts/maps/` (outside the library, never indexed), plus SVG download.

Tasks:

- [x] ADR-022 (Accepted: LLM extraction per fragment, checked against the text): how maps are built (headings only, no LLM · LLM extraction per fragment checked against
      the text · one LLM call for the whole map), the checking rules and the export formats. The user
      chooses.
- [ ] Evaluation set written by the user: 8–12 topics, each with the key concepts a good map must
      contain (`eval/maps.jsonl`), and a review sheet like `ask-eval` to grade whether each node and
      relation is supported by its source.
- [x] `biblio.maps`: Spanish extraction prompt, JSON validation against the fragment text, merging
      (names and embeddings), hierarchy, typed `ConceptMap` (nodes, edges, sources).
- [x] `biblio map "tema"`: prints the map as an outline (for debugging and for the evaluation runner).
- [x] API: `/api/maps` streamed (nodes appear as each fragment is processed), `/api/maps/expand`,
      `/api/maps/save`. One generation at a time, like drafts and quizzes.
- [x] Page **Mapas**: radial SVG layout, pan and zoom, keyboard navigation between nodes, sources
      panel, expand, export, the text outline; checked with headless Edge screenshots.
- [ ] Measure on the set: concept coverage, share of nodes and relations supported by their sources
      (user-graded), time to the first node and to the full map. Targets agreed with the user before
      the page is adopted.
- [x] Docs: README, CLAUDE.md, the log below.

**Exit criterion:** on the evaluation set, the user-graded share of supported nodes and relations
meets the agreed target, and a map from 10 fragments completes within the agreed time (estimate:
~3 s per fragment as measured for Practicar, so ~30 s in all, with the first nodes after ~3 s).

**Risks.** Extraction quality of Qwen3-4B: checking against the text removes invented concepts but
may also drop synonyms and abstractions. Readability past ~40 nodes: a cap per level (config).
Few cross-guide links, because the link graph is sparse.

---

## Log

| Date | Phase | Change |
|---|---|---|
| 2026-10-05 | — | Project initialized: CLAUDE.md, PLAN.md, DECISIONS.md, skills. |
| 2026-10-17 | — | All project documentation translated to English; convention updated in CLAUDE.md. |
| 2026-10-17 | 0 | uv env (Python 3.12, CPU torch), typed config with local.toml merge, offline env guard and loopback check, `models pull` with pinned commits + manifest (bge-m3 MIT, reranker Apache-2.0), `doctor` (+ `--load-models`), 51 tests. |
| 2026-10-17 | 0 | Exit criterion met: `doctor --load-models` green with every non-loopback socket and DNS lookup blocked in-process; 0 network attempts. ADR-011 closed. Phase 1 active. |
| 2026-10-17 | 1 | Ingest layer: discovery, parser (markdown-it + dollarmath, frontmatter), structural chunker v1 (line-range partition, merge/split, atomic units), link extraction, bge-m3 token counter, `biblio chunks`. 17 pathological fixtures; coverage and atomicity property tests. Real fixtures and the 10-file manual review are pending (need the library path). |
| 2026-10-17 | 1 | Library set (an Obsidian vault): `.obsidian/` and `.trash/` excluded by default. 2,252 files → 61.8k chunks, 18.7M bge-m3 tokens (p5/p50/p95 = 81/258/646), 0 invariant violations library-wide. |
| 2026-10-17 | 1 | Review of 10 random files (Claude): fixed lead-ins split from their formula/table/code (colon-ending boundaries 275 → 77, the rest are before headings), no sub-`min_tokens` fragments from splits (189 → 10, all single-chunk notes), unclosed-fence warning (1 in the library: Q-Learning.md:158). Pending: user sign-off of the review; decisions on oversized tables and indexing time. |
| 2026-10-17 | 1 | ADR-012 (supersedes ADR-008): GPU fp32 for batch indexing, CPU for queries. PyTorch switched to the CUDA 13.0 build (torch 2.14.1+cu130). Measured: CPU 516 tok/s (~11.5 h full index), GPU fp32 10,299 tok/s (~35 min, 2.71 GiB VRAM, vectors identical to CPU), GPU fp16 35,195 tok/s. New config keys `embedding.index_device`/`index_dtype`; doctor checks CUDA; network-blocked doctor still green. |
| 2026-10-17 | 1 | Decision (user): oversized glossary tables and code blocks stay whole (ADR-003 unchanged) until the Phase 3 eval can measure splitting them; glossary questions added to the eval-set requirements and a measurement task added to Phase 4. |
| 2026-10-17 | 1 | Phase 1 closed: user signed off the chunk review. Phase 2 active. Q-Learning.md has an unclosed fence and unformatted sections 8–10 (note content, not a chunker bug). |
| 2026-10-17 | 2 | Index layer: SQLite schema (documents, chunks, external-content FTS5 via triggers, vec0 cosine 1024-d, index_meta), `Embedder` Protocol + bge-m3 (GPU fp32 for indexing, CPU for queries), incremental indexing by sha256 with one transaction per document, `--full` built in a temp file and swapped on success, version guards (model, revision, dtype, max_seq_length, chunker, thresholds, library), LLM-server probe before GPU use, single-run lock, progress bar. 19 new tests. |
| 2026-10-17 | 2 | Full index of the library: 2,252 docs, 61,753 chunks, 419 MB, 2,300 s total (2,107 s embedding, 29.3 chunks/s on the RTX 5060). Second run with no changes: 0.6 s. Two notes have stray non-UTF-8 bytes (GCP-BigQuery.md:463, Estadistica-Basica.md:1755). Phase 2 closed; Phase 3 active. |
| 2026-10-17 | 3 | Retrieval baselines and eval harness: `Hit`/`Retriever` interfaces, dense (sqlite-vec cosine, CPU query embedding, guarded against embedder mismatch) and BM25 (FTS5, quoted OR terms) retrievers; metrics (Recall@k, MRR, nDCG@8, all@k for synthesis; file and section level; per type); `biblio eval` (pre-checks expected paths/headings, warns on identical copies, appends to `eval/results/runs.jsonl` with per-question ranks and latency), `biblio compare`, `biblio search`. 26 new tests. Observed on the real index: BM25 surfaces author-header and table-of-contents chunks; 13 groups of byte-identical notes (14 copies). Waiting for `eval/questions.jsonl` (user) to record the baselines. |
| 2026-10-17 | 3 | User question set v1: 48 questions (25 lookup, 13 exact_term, 6 synthesis, 4 absent), all on 2 guides (`Data Analyst/business-intelligence.md`, `Data Analyst/Data Straategy/Data-Lifecycle.md`); paths fixed by Claude (folders missing). Short of the plan: synthesis 14% (< 30%), no glossary questions, 2 of 2,252 guides. v1 baselines recorded (`v1-baseline-dense`, `v1-baseline-bm25`, `v1-baseline-hybrid`); section recall@8 0.498 / 0.625 / 0.612. |
| 2026-10-17 | 4 | Hybrid (RRF k=60) and cross-encoder reranking implemented as opt-in modes (`hybrid`, `dense-rerank`, `hybrid-rerank`); `default_mode` unchanged (dense). Latency p95 on 20 queries: dense 354 ms, bm25 67 ms, hybrid 425 ms, hybrid-rerank on CPU 33.5 s (40 cand.) / 18.1 s (20) / 9.2 s (10), on GPU 1.9 s (40). v1 eval with the reranker on GPU (quality only): hybrid-rerank section recall@8 0.756 (+0.144 vs hybrid), MRR 0.591 (+0.163), nDCG@8 0.614 (+0.171), exact_term +0.31, no type regresses; net +5 questions in the top-8 (8 found, 3 lost). Error analysis v1: most misses are ambiguity (other guides have sections dedicated to the same topic). Pending: ADR on reranker device; eval set v2 (broader + alternative answers). |
| 2026-10-17 | 4 | ADR-013 (supersedes ADR-012, user's choice): reranker on the GPU for `search` while no LLM server runs, CPU fallback with `reranker.cpu_candidates` (10) otherwise. `default_mode` → `hybrid-rerank` (v1: section recall@8 0.612 → 0.756, no type regresses; to be re-confirmed on question set v2). Reranker now reranks only its candidates (deeper eval hits keep first-stage order). `v1-hybrid-rerank` via CLI: device cuda, 40 candidates, p95 1,776 ms. Doctor reports where the reranker would run. |
| 2026-10-17 | 4 | CPU fallback measured (`v1-hybrid-rerank-cpu10`): section recall@8 0.634, MRR 0.565, nDCG@8 0.548, p95 8.4 s. vs GPU (40 cand.): −0.121 recall@8, exact_term −0.31. vs plain hybrid: recall@8 +0.022 (1 question, noise), MRR +0.137, 20x slower. Open question for the user: keep the CPU fallback or fall back to plain `hybrid` while an LLM server runs (would amend ADR-013). |
| 2026-10-17 | 4 | ADR-014 (supersedes ADR-013, user's choice): with the GPU busy or missing, `search` skips the reranker and returns plain `hybrid` (p95 0.43 s) instead of reranking 10 candidates on CPU (p95 8.4 s, same recall within noise). `reranker.fallback = "skip"` (default) or `"cpu"`. |
| 2026-10-17 | 3 | Question set v2 = v1 (48) + user's 22 new questions (ids `v2-*`, 3 more guides: GBM, Factor Graphs, Embeddings); paths fixed and v2-q019 set to lookup (1 expected section) by Claude, as agreed. 70 questions, 64 scored, 5 guides, synthesis 16%, exact_term 33%. v1 kept as `eval/questions.v1.jsonl`. Baselines `v2-dense` / `v2-bm25` / `v2-hybrid`: section recall@8 0.546 / 0.648 / 0.647, MRR 0.459 / 0.439 / 0.467. Still missing: other areas (Math, Software, Cloud, Economía), cross-guide synthesis, glossary questions. |
| 2026-10-17 | 4 | `v2-hybrid-rerank` (GPU, 40 cand.): section recall@8 0.769 (+0.122 vs hybrid), MRR 0.597 (+0.129), nDCG@8 0.618 (+0.138), p95 1.85 s; exact_term +0.238, synthesis +0.183, lookup +0.030; net +6 questions in the top-8 (9 found, 3 lost). Confirms the `hybrid-rerank` default on v2. |
| 2026-10-17 | 3 | Eval format: optional `alternatives` per expected section (any option satisfies it, counted once); byte-identical copies accepted automatically (`identical_copies_accepted` in each run); `biblio candidates <tag>` writes `eval/review/<tag>.md` with the sections returned for missed questions, for the user to judge. Generated `eval/review/v2-hybrid-rerank.md` (12 missed questions). |
| 2026-10-17 | 4 | Moved on to Phase 5 at the user's request with Phase 4 partly open: numeric targets not yet agreed (proposed: section recall@8 ≥ 0.75 overall and ≥ 0.65 per type, MRR ≥ 0.55, p95 ≤ 3 s), alternatives review (`eval/review/v2-hybrid-rerank.md`) and the oversized-table measurement pending. `hybrid-rerank` stays the default. |
| 2026-10-17 | 5 | ADR-015 (supersedes ADR-007, user's choice): Qwen3-4B-Instruct-2507 Q4_K_M on Ollama 0.18.2, KV q8_0, 8k context, loopback only (`scripts/serve-llm.ps1`); pinned by manifest digest, checked by doctor. Measured: KV 612 MiB, 3.4 GiB total, 100% GPU, 1.2 GiB headroom, prompt 3,715 tok/s, generation 90 tok/s. Qwen3 tokenizer pinned and pulled to count the context budget offline. |
| 2026-10-17 | 5 | `biblio ask`: `LLMClient` (OpenAI-compatible, loopback-only, streaming), context assembly (numbered fragments, contiguous hits joined, whole-fragment budget in Qwen3 tokens, max 6), Spanish prompts in `generate/prompts/`, citation validation (invalid numbers, ungrounded answers, abstention), streaming CLI with cited sources. Smoke test: Data Warehouse vs Data Mart answered with citations from 3 guides (retrieval 0.6 s, answer 2.5 s); Kubernetes bare metal → abstained. 11 new tests. Pending: map-reduce, answer evaluation set (user), reranker inside `ask`. |
| 2026-10-17 | 6 | `biblio draft`: step 1 proposes an outline from topic evidence + the H2/H3 structure of the 2 closest guides, written to `drafts/<slug>.outline.md` for the user to edit; step 2 (`--outline`) writes each section from its own retrieval (topic + section title), cites `[n]` renumbered to a global `## Fuentes` list, marks sections without evidence or without citations, and opens with the ADR-008 disclaimer. Smoke test on «Validación cruzada en machine learning»: outline 16 s (after dropping a title and an «Índice» section the model added), draft 54 s, 8 sections all cited, 15 sources. 9 new tests. Phase 5 still has map-reduce and the answer evaluation set pending. |
| 2026-10-17 | 5 | Answer evaluation: question set v3 (`eval/questionsv3.jsonl`, user: 5 lookup, 6 synthesis of which 3 span 3 guides, 3 absent, key_points; paths fixed by Claude). `biblio ask-eval run/score` (review sheet with cited fragments and grade lines; faithfulness, coverage, abstention). Map-reduce implemented (per-guide map notes keeping global [n], then reduce; auto trigger: ≥3 guides or budget overflow). Runs `v3-off` / `v3-auto`: mean ~3 s vs ~15 s; auto fired on 12/14 questions; off abstained on the cross-guide a06/a08 where map-reduce answered (a08 without valid citations); a13 and a14, labeled absent, are covered by the library (a14 answered in both runs). Default set to `map_reduce = "off"` pending the user's grades. |
| 2026-10-17 | 5 | Defect found in the review sheet: map/reduce prompts showed the placeholder `[n]`, and the model copied it (70 in map notes, 5 answers of `v3-auto`). Fixed with concrete examples in the map/reduce prompts (direct prompt unchanged); placeholder citations now count against `grounded` and are reported. Re-run `v3-mr` (map-reduce on for all 14): 0 placeholders; a06/a08 (cross-guide) answered with citations; but a12 (kubeadm, truly absent) answered with steps from the model's own knowledge cited to unrelated fragments — automatic `grounded` cannot catch this, only human grading. Pending: user grades `v3-off` vs `v3-mr`; a13/a14 labels. |
| 2026-10-17 | 7 | Link-graph expansion analyzed before any ADR: 2,690 resolved links, a tree of index notes (each guide links mainly to its index); on the 12 questions `hybrid-rerank` misses, the expected guide is 1 hop from the top-8 in 1 case, while expansion would add ~24 guides per query. Upper bound +1/64 (noise). User chose to leave the task open, no ADR recorded. |
| 2026-10-17 | 7 | ADR-016 (user's choice): `biblio serve`, stdlib `ThreadingHTTPServer` + own HTML/CSS/JS, no new dependency. Loopback-only bind, Host-header check (DNS rebinding), JSON-only POSTs, no CORS, strict CSP, text inserted with textContent; a test fails if any asset references an external URL. Query embedder kept warm; reranker plan resolved per request and released before every ask (ADR-014). Views: search (Obsidian links) and ask (SSE streaming, citations linked, validation flags). Real run: search 452 ms, ask 4.2 s grounded; found and fixed a dropped connection on non-UTF-8 bodies (now 400). 13 new tests. |
| 2026-10-17 | 5 | User graded `v3-off` (transcribed by Claude from the chat into the sheet). Direct strategy: lookup faithfulness 1.00 (5/5 fully faithful), coverage 0.60; synthesis faithfulness 0.53 (10/19 claims), coverage 0.32, false abstention 2/6 (a06, a08); absent 2/3 abstained (a14 mislabeled). Bug found: a stray backspace () replaced  in the grade regex, so no grade was ever parsed; fixed, repo scanned for control characters (none left), bogus empty score line removed. Parser now also accepts bold/hand-made sheets. Pending: grades for `v3-mr` (a06-a12 first) to decide map-reduce; a13/a14 labels. |
| 2026-10-17 | 5 | Audit of the v3-off grades (pasted by the user) found claims "not visible in the excerpt": the review sheet cut every cited fragment at 400 characters, so faithfulness could not be verified (tool defect). Sheets now include the full text of each cited fragment (collapsible, fenced); new `biblio ask-eval sheet <tag>` regenerates a sheet from the saved run keeping grades, notes and checked key points (v3-off grades verified identical after regeneration). Disputed fidelity grades (a03, a05, a07, a09, a10, a11) to be re-checked against the full fragments before scoring. |
| 2026-10-17 | 5 | User graded `v3-mr` and re-checked the six disputed `v3-off` grades against full fragments (a07 2->4/5, a09 1->0/4, a11 3->6/6; a03, a05, a10 confirmed); transcribed by Claude. Scores (`eval/results/answers.jsonl`): direct vs map-reduce — faithfulness 0.87 vs 0.58, fully faithful 80% vs 29%, lookup coverage 0.60 vs 0.13, synthesis faithfulness 0.74 vs 0.45 (coverage 0.32 vs 0.38; false abstentions 2/6 vs 0/6), absent abstention 2/3 vs 0/3 (map-reduce hallucinated a12/a13), mean 3.1 s vs 13.3 s. Map-reduce answers more synthesis questions but with citations that do not support the claims. Pending: ADR on map-reduce. |
| 2026-10-17 | 5 | ADR-017 (user's choice): map-reduce kept as a manual opt-in, off by default; `ask` warns when it is used. |
| 2026-10-17 | 7 | ADR-018 (user's choice): `biblio watch` polls the library's (mtime, size) every 30 s with the stdlib, waits 10 s for the change to settle and runs the usual incremental index; embeddings on the GPU when free, on CPU while the LLM server listens (a float16 index is deferred instead). Locked index retried next check; version mismatch stops it. Real library: snapshot 0.37-0.39 s; no-change catch-up 1.4 s warm (14.6 s with a cold disk cache). 11 new tests (306 pass). |
| 2026-10-17 | 4 | Phase 4 targets agreed with the user: section recall@8 ≥ 0.75, MRR ≥ 0.55, p95 ≤ 3 s (the proposed per-type floor was not adopted). `v2-hybrid-rerank` meets them: 0.770 / 0.597 / 1.85 s; by type recall@8 lookup 0.82, synthesis 0.73, exact_term 0.71 (MRR 0.43). The plain-hybrid fallback used while the LLM server runs (ADR-014) is below them (0.647 / 0.467), as accepted in that ADR. Error analysis and the oversized-table measurement remain open. |
| 2026-10-17 | 5 | `eval/questionsv3.jsonl` a14 relabeled by the user from absent to synthesis: expected `ML-CLass-NBayes-Algorithm.md` § 2 and § 4 (§ 3 accepted for § 4), 4 key points (Bayes rule, conditional independence, argmax without P(x), named terms). a13 still absent but covered by the library; the proposed DPO replacement was rejected because `DataScience-LLMs-DPO.md` covers it; waiting for another topic. |
| 2026-10-17 | 5 | `eval/questionsv3.jsonl` a13 replaced by the user with an out-of-domain absent question (roux proportions); checked: no cooking content in the library, top search hits unrelated. v3 is now 5 lookup, 7 synthesis, 2 absent. |
| 2026-10-17 | 7 | ADR-019/020 accepted (user approved the canvas designs). Web UI v2, page Conversar: one thread with "Explicar"/"Solo ubicar", citations open the full fragment in a side panel, settings (precision, k, folder, strategy), `/ubicar` `/explicar` `/en <carpeta>` shortcuts, thread and settings in localStorage; `/api/folders`, `scope`/`precision`/`strategy` parameters, full fragment text in responses. Folder scope: dense via sqlite-vec KNN restricted to the folder's chunk ids (0.15 s for 10k chunks); BM25 ranks every match then keeps the folder's (a `rowid IN` constraint took 26 s, now 0.08 s). Rule 3: `v2-hybrid-adr019` identical to `v2-hybrid` (no question entered or left the top-8). Live: scoped search 0.36–0.49 s, scoped ask 3.0 s. 312 tests. |
| 2026-10-17 | 7 | Web UI v2, page Guías (ADR-019): `/api/drafts` (list, detail), `/api/drafts/outline` and `/api/drafts/write` (server-sent events), same files as `biblio draft`; outline editing (rename, reorder, notes, add/remove), live section-by-section writing, sources with full fragments, copy/download. One generation at a time (409 otherwise); drafts take the engine lock only per search, so the other pages keep answering; closing the page stops the generation. Minimal Markdown renderer (paragraphs, headings, lists, code, tables with citations) for drafts and Conversar answers, built with DOM nodes only. Headless Edge screenshots used to check the layout: found and fixed `[hidden]` being overridden by `display` rules (both pages showed at once). Live: outline 6.8 s, two sections 13.2 s with valid citations. 317 tests. |
| 2026-10-17 | 7 | ADR-021 (user request): one-step launcher. `Biblioteca.cmd` (Windows) and `scripts/start.sh` install uv if missing (after asking), run `uv sync`, then `biblio start`: missing models (asks), library path saved to `config/local.toml` (asked once), incremental index before the LLM so it gets the GPU (first full index asked for), Ollama started with the settings read from `scripts/serve-llm.ps1` and stopped on exit, pinned model pulled if missing (asks) and its digest checked, web UI + browser, watcher thread. Already running → just opens the browser. `index`/`watch`/`serve` refactored to share `_run_index`, `_make_watcher`, `_open_web`. Live on this machine, through `Biblioteca.cmd`: deps checked in 6 ms, index unchanged in 13 s on GPU, Ollama started, ask answered with a valid citation; first run found that Ollama opens its port before its API answers (now retried). 8 new tests (325). |
| 2026-10-17 | 7 | Web UI v2, page Practicar (ADR-020): `biblio.practice` (store in `data/practice.db`: quizzes, questions, attempts; section choice: failed first, then new, then oldest, flagged sections excluded, navigation/reference sections and sections outside `[practice]` token bounds skipped; generation per section with validated JSON and one retry; open-answer grading by the model with user review and agreement tracking), `/api/practice/*` (answers graded and recorded on the server, never sent before answering), `practicar.js`. Questions stream in while the first is answered (only the header updates, so typing is not interrupted); «Terminar» aborts generation. Live (Qwen3-4B): 4 questions of the 4 kinds in 9.7 s, model grade in 0.9 s; 8 multiple-choice from a topic in 29 s. Found live: explanations named options by position, wrong after shuffling; now the right answer's letter is chosen at random in the prompt and explanations use letters (0 of 8 numeric references after the fix). Driven end to end in headless Edge through the DevTools protocol with no page errors. Test history removed from `data/practice.db`. 13 new tests (338). |
| 2026-10-17 | 7 | Web UI: light / system / dark theme switch in the top bar, saved in localStorage; `theme.js` (not deferred) applies it before the first paint, so the page never flashes the other theme. Checked in headless Edge with the OS in dark and light: each choice wins over the OS, survives a reload, and "system" follows the OS again. |
| 2026-10-17 | 8 | Phase 8 (concept maps, page `Mapas`) planned at the user's request: design, tasks, exit criterion and risks. Not started: it needs ADR-022 and an evaluation set written by the user first. |
| 2026-10-18 | 8 | ADR-022 accepted (user's choice): concept maps by LLM extraction per fragment, checked against the fragment text, merged and laid out in code. Next: the evaluation set of topics and key concepts (written by the user). |
| 2026-10-18 | 8 | Concept maps built (ADR-022): `biblio.maps` (Spanish extraction prompt; concepts kept only if written in their fragment, relations only between kept concepts; merge by name and by query-embedding similarity with a cache; hierarchy in code: level 1 by fragments, relations and closeness to the topic, children need a stated relation or two shared fragments, weak one-word concepts dropped; radial layout computed in Python and shared by page and Canvas), `biblio map "tema"`, `/api/maps` streamed and `/api/maps/save` (Markdown + Obsidian Canvas in `drafts/maps/`), page Mapas (SVG, pan/zoom, keyboard focus on nodes, outline view, panel with relations and source fragments, Expandir/Explicar/Practicar, SVG download). Live: «Naive Bayes» 14 concepts from 10 fragments, first nodes 4.9 s, full map 34 s; «validación cruzada» 13 concepts, 32.6 s. Found live: children attached by one shared fragment were unrelated (now need a relation or two shared fragments); relation direction was ambiguous in the outline (now full sentences); the map was fitted only to its first nodes (now re-fitted until the user moves it). Pending: the user's evaluation set and the measurements against it. 8 new tests (346). |
| 2026-10-19 | 7 | ADR-023 (user's choice): Conversar history on the server, `data/conversations.db` (conversations, messages with their full sources as page JSON, FTS5 over questions and answers with accent folding, AND search with snippets). `/api/search` and `/api/ask` take `conversation_id` and store each message when complete (errors too, with the id in the error event); `/api/conversations` list/search/detail, rename, delete, export to `drafts/conversations/`. Page: history panel with search, reopen, rename/export/delete bar; only the open conversation id stays in localStorage. 6 new tests (352). |
| 2026-10-19 | 7 | Web UI: «Actualizar» button next to the notes count runs the incremental reindex from the page (`/api/reindex`, streamed progress; GPU if free, else CPU, as the watcher; 409 while another index run holds the lock; LLM not required). Engine: `_threaded` generalizes the threaded-event helper. Live: unchanged library checked in 1.0 s. 3 new tests (355). |
| 2026-10-19 | 7 | Web UI: «Historial» button in the top bar (Conversar only) shows or hides the conversations panel; remembered in localStorage, `aria-expanded` kept in step, focus moves to the history search when opened. Checked in headless Edge. |
| 2026-10-19 | 8 | ADR-024: map history in `data/maps.db` (map as sent to the page plus its source fragments; FTS5 over topic and concept names). Every map is stored when complete (`map_id` in the done event); `/api/maps/history[?q=]`, `/api/maps/history/<id>`, `/api/maps/delete`; «Guardar» now works on any stored map (domain objects rebuilt from the stored JSON). Page: history panel on Mapas, reopen without the LLM, «Borrar»; the «Historial» button now serves Conversar and Mapas, each remembering its own state. Checked in headless Edge. 1 new test (356). |
| 2026-10-19 | 8 | Fix reported by the user (map of «quimica» off-topic): retrieval was right (10 fragments from `Cosas/Quimica`), but the topic rule merged every concept containing the topic word into the centre, so 21 branches («química orgánica», «bioquímica»…) vanished and their relations were attributed to the topic («quimica estudia carbono»). Now only the topic itself (give or take an article or plural) is the centre (regression test). Also: level 1 ranked by fragments more than by relations (a method's 5-step list no longer outranks the branches), `level1_max` 7→9, children need a relation stated in a fragment (shared fragments alone nested sibling branches under each other), prompt asks for relations to the topic and for reading direction. Live: «quimica» now 9 branches around the topic, children by stated relations, 40 s. |
| 2026-10-19 | 8 | Mapas: nodes can be dragged with the mouse (a click still selects; the background still pans) or moved with the arrow keys; lines and labels follow without a redraw, relations unchanged. Moves are stored apart from the automatic layout (`positions` column, migrated into existing `maps.db`; `/api/maps/positions`, `{}` resets), applied when a map is reopened and used by «Guardar» (Canvas). «Reorganizar» restores the automatic layout. Checked in headless Edge with real mouse events. 2 new tests (359). |
| 2026-10-19 | 8 | Mapas: «Concepto» button shows or hides the concept panel (remembered; the map re-fits to the space). The drawing area has its own theme tokens: soft mid-grey background in the light theme (`#dfe2e6`) with darker lines and labels, so white and green nodes and the relations stand out; a slightly lighter grey in the dark theme. Checked in headless Edge in both themes. |
