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

**Exit criterion:** Phase 3 targets reached or the deviation justified in an ADR. **At this point the search engine is done.**

## Phase 5 — Tutor (`biblio ask`)

Use the `memory-budget` and `rag-prompting` skills.

- [x] `LLMClient` (Protocol) with an HTTP client against an OpenAI-compatible API on `localhost`.
- [x] Choice of the concrete model according to ADR-011 and `memory-budget`; record an ADR.
- [x] Context assembly with a token budget (`context_budget_tokens`), fragments numbered `[1]..[n]`.
- [x] Prompt with mandatory citations and explicit abstention.
- [x] Post-validation: every cited `[n]` exists; no citations → answer flagged as ungrounded.
- [~] Map-reduce mode for `synthesis` questions when the evidence does not fit the budget.
      Implemented (`--map-reduce auto|on|off`, default off); adoption waits for the user's grades of `v3-mr` vs `v3-off` and an ADR.
- [x] Answer evaluation set (10–15 questions): faithfulness to sources, coverage and correct abstention, with human review.
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
- [ ] File watcher for automatic reindexing.

---

## Log

| Date | Phase | Change |
|---|---|---|
| 2026-10-05 | — | Project initialized: CLAUDE.md, PLAN.md, DECISIONS.md, skills. |
| 2026-10-17 | — | All project documentation translated to English; convention updated in CLAUDE.md. |
| 2026-10-17 | 0 | uv env (Python 3.12, CPU torch), typed config with local.toml merge, offline env guard and loopback check, `models pull` with pinned commits + manifest (bge-m3 MIT, reranker Apache-2.0), `doctor` (+ `--load-models`), 51 tests. |
| 2026-10-17 | 0 | Exit criterion met: `doctor --load-models` green with every non-loopback socket and DNS lookup blocked in-process; 0 network attempts. ADR-011 closed. Phase 1 active. |
| 2026-10-17 | 1 | Ingest layer: discovery, parser (markdown-it + dollarmath, frontmatter), structural chunker v1 (line-range partition, merge/split, atomic units), link extraction, bge-m3 token counter, `biblio chunks`. 17 pathological fixtures; coverage and atomicity property tests. Real fixtures and the 10-file manual review are pending (need the library path). |
| 2026-10-17 | 1 | Library set (`C:/Users/USER/iCloudDrive/AI Engineer`, Obsidian vault): `.obsidian/` and `.trash/` excluded by default. 2,252 files → 61.8k chunks, 18.7M bge-m3 tokens (p5/p50/p95 = 81/258/646), 0 invariant violations library-wide. |
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
| 2026-10-20 | 9 | Notes in English in Mapas and the Diccionario (ADR-042, user request; the user chose the original name with a Spanish one beneath). Measured first on 9 English sections and 6 Spanish ones with the real model (`eval/results/english_notes.jsonl`): maps kept 56 % of concepts in English vs 74 % in Spanish (17 dropped as translations, three sections empty), the dictionary 73 % vs 86 %; diagrams (causes 100 %, flow 85 %) and «Explícamelo tú» (ideas 93 %, claims 85 %) lost nothing to translation, so they were left as they were. Changed: maps ask for `{nombre, es}` per concept and the topic as the fragment names it (`tema`, made the centre), the dictionary for `traduccion`; checks stay on the original; English fragments get a direct note with an example (the general rule alone still let one section be fully translated); translated concepts merge with the Spanish notes' ones; the page shows the Spanish name in italics beneath, exports write «name (translation)», the map history searches translations; an English fragment naming the topic means it is not absent (gap detector). After: maps 86 % in English, dictionary 91 %, 0 translated; Spanish maps 76 % with the old and the new prompt on 16 sections. Live: «costos del burnout en los empleados» mixes «replacement cost / costo de reemplazo» with Spanish notes' concepts. 455 tests pass. |
| 2026-10-20 | 9 | Ajustes › Modelo de lenguaje (ADR-043, user request): «Apagar el modelo» / «Encender el modelo». `launcher.LLMSwitch` keeps the Ollama server the app started (`biblio start` hands it over; `biblio serve` gets an empty one) and stops only that one; an Ollama opened outside the app gets its model unloaded (`keep_alive: 0`) and keeps running, which the page explains. Starting uses the settings of `scripts/serve-llm.ps1` and waits for the API. `/api/llm/stop`, `/api/llm/start` (this PC only; refused while generating); `/api/status` gains `llm_control`. Checked live in headless Edge with the real Ollama: outside server → model unloaded, process kept; after closing it, «Encender» started the app's own server (owned, «Lista» in the top bar) and «Apagar» closed it (port closed, «Modelo apagado»). 459 tests pass. |
| 2026-10-20 | 9 | Search heuristics for big notes (ADR-044, user request; plan accepted). Analysis of plain hybrid's 21 misses on `eval/questions.jsonl`: the right section loses mostly to other notes on the same subject (137 hits above it), not to its own note's sections (5) or tables of contents (3); one note fills 3+ of the top 8 in 30 of 64 questions; oversized chunks barely appear. The set has no question on a note over 60 fragments (125 such notes): `eval/review/big-notes-candidates.md` (gitignored) lists 47 deep sections of 15 big notes for the user to write questions on (step 0). Done: tables of contents (2,172 sections, by heading only) are never results nor reranker candidates (`retrieve/navigation.py`); BM25 weighs the heading path, which carries the note's title into every chunk, at 0.5 (measured 0.1–4; raising it hurt). Hybrid R@1 0.255 → 0.286, R@8 0.647 → 0.663, exact-term R@1 0.048 → 0.095; 12 questions better, none worse; hybrid-rerank unchanged. No reindex. Next: H3 (group search results by note) and H4 (search inside a note in the reader), and a separate title column once the big-note questions exist. 461 tests pass. |
| 2026-10-20 | 9 | H3 and H4 of the big-notes plan (ADR-045, user request). «Solo ubicar» grouped by note: `/api/search` with `group: true` reads k × 3 hits and returns k notes (`retrieve/grouping.py`); the page shows each note's best section, the next two beneath, the rest folded. Measured on `eval/questions.jsonl` before building the page (`eval/results/grouping.jsonl`): distinct notes on screen 5.9 → 8.0; the right section stays in view as often as before (hybrid 0.688 → 0.719, hybrid-rerank 0.781 = 0.781), but only with the next two sections in view (best only: 0.672 / 0.703). Reader: «Buscar en esta nota» (`/api/notes/search`, scope = one `.md`, not kept in the history), a sticky bar with the note's sections ranked, a click jumps below the results. Live: a 143-fragment note took 14 of 24 hits for «frames y slots…», now one line with 11 folded; searching inside it found «Estructura de un frame» first and jumped to «Frames vs. clases orientadas a objetos». The test conversation was deleted. 463 tests pass. |
| 2026-10-21 | 9 | English interface (ADR-046, user request; the user chose level 1: the interface only). `static/i18n.js` (`tx`, `txServer`, `i18nLocale`, the static HTML translated on load) and `static/i18n-en.js` (789 entries, Spanish text → English). 595 literals in 11 page scripts wrapped by a scanner that parses strings, nested templates, comments and regular expressions (templates become `tx("… {0} …", expr)`), then a residual sweep by hand (plurals, a page-name list, a comparison with a now-translated label, two internal ids reverted). Server error messages translated on the page (exact or with `{0}` patterns); dates and numbers by locale. Ajustes › Apariencia › Idioma, kept per browser; generated content stays Spanish. Tests: every `tx` key and HTML text has an English entry with the same placeholders; no accented literal outside `tx`; the files are served. Checked on a second instance in headless Edge: the five pages, Ajustes and the reader in English with no Spanish interface text left (only the user's own section names), and back to Spanish unchanged; no page errors. 466 tests pass. |
| 2026-10-21 | 9 | «Gap Spotter» in Guías (ADR-047, user request; the user chose both sources marked, cards + pending topics, and Guías). `study/spotter.py`: the topic's sections (16) → terms mentioned without explaining (quotes checked) → the model's syllabus (16 subtopics, half concrete methods) → each candidate searched in the whole library and judged («cubierto»/«parcial»/«no», credited only with a quote of 6+ consecutive words in a section read) → 5–10 guide cards (each gap once, questions, related notes). Runs in `data/practice.db` (`spotter_runs`); `/api/spotter` (stream), history, delete, «Apuntar en Temas pendientes» (gap source `spotter`). Page: tile and screen in Guías, progress, coverage lists with sources and quotes, guide cards with «Copiar ficha» (Markdown) and «Apuntar…»; all text through `tx` with English entries. Live: four issues found and fixed (cut JSON, LaTeX escapes, exact-quote false gaps, a too basic syllabus); «aprendizaje por refuerzo» 57 s with real, deep gaps; «regresión logística» 64 s. Test data deleted. 473 tests pass. |
