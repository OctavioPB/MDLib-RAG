# DECISIONS.md — biblio-rag

Architecture decision record (ADR). Format and rules: `decision-record` skill.
States: **Proposed** · **Accepted** · **Superseded by ADR-NNN** · **Rejected**.

| ADR | Title | State |
|---|---|---|
| 001 | RAG instead of training or fine-tuning as the knowledge substrate | Accepted |
| 002 | Full operational independence; origin dependency accepted | Accepted |
| 003 | Structural chunking by headings with a hierarchical path | Accepted |
| 004 | BGE-M3 as the embedding model, versioned in the index | Accepted |
| 005 | SQLite + sqlite-vec + FTS5 with exact search | Accepted |
| 006 | Hybrid retrieval with RRF and a cross-encoder reranker | Accepted |
| 007 | LLM behind a local OpenAI-compatible server | Superseded by ADR-015 |
| 008 | Memory budget < 8 GB and an assumed quality ceiling | Superseded by ADR-012 |
| 009 | Search engine without an LLM first; evaluation before optimization | Accepted |
| 010 | Python 3.11+, uv, typer | Accepted |
| 011 | Reference hardware: Windows 11, Ryzen 7 5700X, 32 GB RAM, RTX 5060 8 GB (CUDA) | Accepted |
| 012 | Memory budget < 8 GB; GPU for the LLM at query time and for batch embedding while no LLM runs | Superseded by ADR-013 |
| 013 | Memory budget < 8 GB; the GPU serves whichever workload is active, the LLM has priority | Superseded by ADR-014 |
| 014 | As ADR-013, but `search` skips the reranker (plain hybrid) when the GPU is not free | Accepted |
| 015 | Qwen3-4B-Instruct-2507 (Q4_K_M, KV q8_0, 8k context) on Ollama, behind `LLMClient` | Accepted |
| 016 | Local web UI for `search` and `ask`: stdlib HTTP server + own HTML/JS, loopback only | Accepted |
| 017 | Map-reduce is a manual opt-in for `ask`, off by default | Accepted |
| 018 | Automatic reindexing by polling file metadata (stdlib), GPU if free else CPU | Accepted |
| 019 | Web UI v2: Conversar, Guías and Practicar pages on the stdlib server | Accepted |
| 020 | Practice quizzes generated from library sections, with local history | Accepted |
| 021 | One-step launcher: setup steps that download only when missing, after asking | Accepted |
| 022 | Concept maps by LLM extraction per fragment, checked against the text | Accepted |
| 023 | Conversation history in a local SQLite file on the server | Accepted |
| 024 | Concept-map history in a local SQLite file on the server, as ADR-023 | Accepted |

---

## ADR-001 — RAG instead of training or fine-tuning as the knowledge substrate

**Date:** 2026-10-05 · **State:** Accepted

**Context.** The goal is a system that answers based on a personal library of 500–2,000 Markdown guides that grows and changes.

**Options.**
1. Train an LLM from scratch. Requires trillions of tokens; the library provides on the order of millions. Expected result: superficial imitation of the style with hallucinated content.
2. Fine-tune (LoRA) an open model on the guides. Teaches form and style well, and faithful factual recall poorly. It cannot cite, and every change to the library means retraining.
3. RAG: a generic local model that answers over fragments retrieved at query time.

**Decision.** Option 3. Knowledge lives in the documents, not in the weights.

**Consequences.** (+) Verifiable citations, immediate updates on reindexing, less hallucination. (−) Quality depends on retrieval, which becomes the critical piece (see ADR-009). Fine-tuning remains an optional style layer (Phase 7), never a source of knowledge.

---

## ADR-002 — Full operational independence; origin dependency accepted

**Date:** 2026-10-05 · **State:** Accepted

**Context.** The requirement is "absolute independence from external models".

**Decision.** It is interpreted as operational independence: at runtime there are no network calls outside `localhost`, no APIs or telemetry, and no data leaves the machine. The origin dependency is accepted: the weights of open models were trained by third parties, but once downloaded and pinned by revision they cannot be revoked or changed.

**Consequences.** `biblio models pull` is the only networked command. Models are pinned by revision hash. Tests block sockets. The license of every model must be reviewed before pinning it, especially for commercial use. Development may rely on external assistants; the requirement applies to the running system.

---

## ADR-003 — Structural chunking by headings with a hierarchical path

**Date:** 2026-10-05 · **State:** Accepted

**Context.** The guides have a rich heading hierarchy, code blocks, formulas and ASCII diagrams. Fixed-size cutting breaks semantic units and leaves ambiguous fragments ("stability" means different things depending on the guide).

**Decision.** A chunk is a section (or subsection) delimited by headings. Long sections are split by paragraphs and tiny ones are merged. Atomic units (code, tables, `$$…$$`, diagrams) are never split. The embedded text is prefixed with the path `Guide › Section › Subsection`. Parser: `markdown-it-py`, for its tokens with line numbers; regexes are ruled out.

**Consequences.** Section-level citations with a line range. `CHUNKER_VERSION` must be maintained, and changing it forces a reindex. Operational details in the `markdown-chunking` skill.

---

## ADR-004 — BGE-M3 as the embedding model, versioned in the index

**Date:** 2026-10-05 · **State:** Accepted

**Context.** Queries and corpus are in Spanish with technical terminology in English. A multilingual, high-quality model that runs on CPU is needed.

**Options.** BGE-M3 (568M params, 1024 dims, multilingual, long context); multilingual-E5 base or large; monolingual English models (ruled out because of the language).

**Decision.** BGE-M3, dense vectors only at this stage, on CPU, normalized (cosine similarity = dot product). Revision pinned by hash.

**Consequences.** ~2.3 GB of RAM in fp32 during indexing; the initial indexing on CPU can take from tens of minutes to a couple of hours. Vectors from different models are incomparable: `index_meta` stores model and revision, and the system refuses to mix them. Revisit if FTS5 lexical search proves insufficient; in that case, evaluate BGE-M3's sparse vectors in a new ADR.

---

## ADR-005 — SQLite + sqlite-vec + FTS5 with exact search

**Date:** 2026-10-05 · **State:** Accepted

**Context.** Tens of thousands of chunks at most. Minimal infrastructure and a single artifact are sought.

**Decision.** One `data/index.db` file with relational tables, FTS5 (`unicode61 remove_diacritics 2`) and a `sqlite-vec` `vec0` table. Exact (brute-force) vector search, no ANN indexes.

**Consequences.** At this scale exact search takes milliseconds and removes the approximation error of an ANN. A single file makes backup and deletion trivial. Risk: Python builds without `enable_load_extension` (especially macOS), mitigated with the `uv`-managed Python and a check in `biblio doctor`. Revisit if the corpus exceeds ~200k chunks.

---

## ADR-006 — Hybrid retrieval with RRF and a cross-encoder reranker

**Date:** 2026-10-05 · **State:** Accepted

**Context.** Dense search fails on exact terms (acronyms, theorem names, identifiers). BM25 fails on paraphrases.

**Decision.** Dense top-N and BM25 top-N, fused with Reciprocal Rank Fusion (k = 60), which needs no calibration of heterogeneous scores. Then reranking with `bge-reranker-v2-m3` on CPU over the best `rerank_candidates`.

**Consequences.** The reranker dominates latency on CPU; it is controlled with `rerank_candidates` and truncation. Each piece is measured separately in Phase 4; a piece that does not improve the metrics is removed.

---

## ADR-007 — LLM behind a local OpenAI-compatible server

**Date:** 2026-10-05 · **State:** Superseded by ADR-015

**Context.** It must be possible to change runtime (Ollama, llama.cpp's `llama-server`) and model without touching the rest of the system.

**Decision.** Our own `LLMClient` interface. Initial implementation: an HTTP client against an OpenAI-compatible endpoint on `localhost`. The concrete runtime and model are fixed in Phase 5, after closing ADR-011.

**Consequences.** Full decoupling from the runtime. The server is a separate process, which makes it easier to free GPU memory. Provisional until the model is chosen.

---

## ADR-008 — Memory budget < 8 GB and an assumed quality ceiling

**Date:** 2026-10-05 · **State:** Superseded by ADR-012

**Context.** Less than 8 GB is available for models. An LLM's memory is weights + KV cache (proportional to context) + buffers.

**Decision.** The GPU is reserved for the LLM; embeddings and reranker run on CPU. By default, a 3–4B-class model at Q4_K_M; a 7–8B one only if `memory-budget` confirms the headroom. KV cache quantized to q8 when the runtime allows it. Context is limited by `context_budget_tokens`.

**Consequences.** Search engine: no quality loss. Tutor: functional, with synthesis limited to a few sources per call (map-reduce for broad questions). Writing: drafts useful as a skeleton, with quality noticeably below that of large models; explicitly assumed. Future ways out: partial offload to system RAM (slow) or new hardware, each with its own ADR.

---

## ADR-009 — Search engine without an LLM first; evaluation before optimization

**Date:** 2026-10-05 · **State:** Accepted

**Decision.** Phases 1–4 deliver a complete search engine without generation. The evaluation set and the baselines exist before any retrieval optimization. Numeric targets are set after the baseline, not before.

**Consequences.** Useful value early. Retrieval decisions are made with data. If retrieval is bad, no LLM makes up for it; if it is good, a small model performs above its size.

---

## ADR-010 — Python 3.11+, uv, typer

**Date:** 2026-10-05 · **State:** Accepted

**Decision.** Python ≥ 3.11 (native `tomllib`), `uv` for the environment and the managed Python (solves SQLite extension loading), CLI with `typer` + `rich`, quality with `ruff`, `mypy --strict` and `pytest`.

---

## ADR-011 — Reference hardware: Windows 11, Ryzen 7 5700X, 32 GB RAM, RTX 5060 8 GB (CUDA)

**Date:** 2026-10-05 (opened) · 2026-10-17 (closed) · **State:** Accepted

**Context.** Known when opened: < 8 GB of memory for the model. Unknown: operating system, accelerator type (NVIDIA GPU, Apple Silicon, CPU only), total system RAM and CPU model. The LLM runtime (CUDA, Metal or CPU), the model size of ADR-008 and the viability of offloading to RAM depend on this.

**Measured profile** (`biblio doctor`, 2026-10-17):

| Item | Value |
|---|---|
| OS | Windows 11 (10.0.26200, AMD64) |
| CPU | AMD Ryzen 7 5700X, 8 physical / 16 logical cores (Zen 3, AVX2, no AVX-512) |
| RAM | 31.9 GiB total; ~9.7 GiB available with the usual desktop load |
| GPU | NVIDIA GeForce RTX 5060, 8.0 GiB VRAM, driver 610.88; **~4.8 GiB free at idle** (desktop and apps hold ~3.2 GiB) |
| Python / SQLite | uv-managed CPython 3.12.13; SQLite 3.50.4 with extension loading, sqlite-vec v0.1.9 and FTS5 working |

**Decision.** This machine is the reference profile. The LLM runs on the NVIDIA GPU through CUDA; embeddings and reranker stay on CPU (ADR-008). The concrete runtime and model are still chosen in Phase 5 (ADR-007) with the `memory-budget` skill.

**Consequences.** (+) CUDA is available, so a GPU runtime (Ollama or `llama-server` with CUDA) is viable. (+) 32 GB of system RAM makes partial offload to RAM possible as a fallback, and comfortably holds bge-m3 + reranker (~4.6 GB fp32) next to an offloaded LLM. (−) The usable VRAM budget is what is free at run time, not 8 GB: with ~4.8 GiB free at idle, the `memory-budget` calculation must start from the measured free VRAM (re-measure with the usual apps open before choosing), which points to the 3–4B class of ADR-008 unless other GPU consumers are closed. (−) Available system RAM (~10 GiB under normal load) is the practical limit for indexing plus offload, not the 32 GB total. Revisit if the hardware or the usual background load changes.

---

## ADR-012 — Memory budget < 8 GB; GPU for the LLM at query time and for batch embedding while no LLM runs

**Date:** 2026-10-17 · **State:** Superseded by ADR-013 · **Supersedes:** ADR-008

**Context.** ADR-008 put embeddings and reranker on CPU so the GPU stays free for the LLM. Phase 1 measured the real library (2,252 notes → 61,753 chunks, 18.7M bge-m3 tokens), far more text than ADR-004 assumed ("tens of minutes to a couple of hours" of indexing). bge-m3 on the reference CPU (ADR-011) embeds 516 tokens/s, so a full index takes ~11.5 h. A full reindex is required whenever the embedding model or the chunker changes (CLAUDE.md, rule 6), which will happen during Phases 3–4. Indexing is a batch job that does not need the LLM.

**Options.**
1. Keep everything on CPU (ADR-008 as is). No new dependency; ~11.5 h per full reindex, which makes chunker and model experiments impractical.
2. Embed on the GPU during batch indexing only, with the LLM server stopped; queries keep embedding on CPU. Needs the CUDA build of PyTorch (~2.5 GB more on disk).
3. Move all embedding (indexing and queries) to the GPU. Faster queries, but then bge-m3 competes with the LLM for the ~4.8 GiB of free VRAM at query time, which is exactly what ADR-008 avoided.

**Measurements** (128 real chunks, 44,393 tokens, batch 16, `max_seq_length` 1024, 2026-10-17):

| Device / dtype | tokens/s | Full library | Peak VRAM | vs CPU fp32 vectors |
|---|---|---|---|---|
| CPU fp32 | 516 | ~11.5 h | — | — |
| GPU fp32 | 10,299 | ~35 min | 2.71 GiB | cosine 1.000000 (identical) |
| GPU fp16 | 35,195 | ~10 min | 1.38 GiB | cosine ≥ 0.99976; same top-10 on a probe query |

**Decision.** Option 2, in fp32. The memory budget, LLM sizing and quality ceiling stay as in ADR-008: < 8 GB for models; by default a 3–4B-class LLM at Q4_K_M, a 7–8B one only if `memory-budget` confirms the headroom; KV cache at q8 when the runtime allows it; context limited by `context_budget_tokens`; the reranker and query embeddings run on CPU. New: `biblio index` embeds on `embedding.index_device` (default `cuda`, `index_dtype = "float32"`) and falls back to CPU with a warning if CUDA is unavailable. GPU fp32 vectors are identical to CPU fp32 vectors, so an index built on the GPU is queried from the CPU without mixing incompatible vectors. The indexer refuses to use the GPU while the LLM server answers on `llm.base_url` (loopback probe), unless forced. `index_meta` records the dtype; fp16 is a separate configuration that requires a full reindex and an evaluation run (CLAUDE.md, rule 3) before adoption.

**Consequences.** (+) Full reindex in ~35 min instead of ~11.5 h, which makes chunker/model experiments in Phases 3–4 feasible. (+) Query path and memory coexistence unchanged from ADR-008. (−) ~2.5 GB extra disk for the CUDA build of PyTorch (cu130 wheels, NVIDIA driver ≥ 580); machines without CUDA fall back to the slow CPU path. (−) The LLM server has to be stopped during GPU indexing. Revisit if the hardware changes, or if Phase 3 shows fp16 retrieval is indistinguishable from fp32 (then switch the default to fp16 with its own measurement).

---

## ADR-013 — Memory budget < 8 GB; the GPU serves whichever workload is active, the LLM has priority

**Date:** 2026-10-17 · **State:** Superseded by ADR-014 · **Supersedes:** ADR-012

**Context.** ADR-012 kept the reranker on CPU at query time. Phase 4 measured it on the real index (20 timing queries, `final_k` 8; target p95 3,000 ms): `hybrid-rerank` on CPU has p95 33.5 s with 40 candidates, 18.1 s with 20 and 9.2 s with 10, so the CPU path misses the target by 3–11×. On the RTX 5060 (fp32, 40 candidates) p95 is 1.9 s. The reranker is worth having: on question set v1 (`eval/results/runs.jsonl`, tags `v1-baseline-hybrid` vs `v1-hybrid-rerank-gpu`) it raises section recall@8 from 0.612 to 0.756, MRR from 0.428 to 0.591 and nDCG@8 from 0.443 to 0.614, with no question type regressing (exact_term +0.31, synthesis +0.22, lookup +0.04); net +5 questions in the top-8 out of 44. v1 covers only 2 guides, so the absolute numbers are provisional; the comparison between modes is not.

**Options.**
1. Reranker on CPU with 10 candidates (ADR-012 as is): p95 9.2 s, still 3× over target, and only part of the gain.
2. Reranker on the GPU for `search` when no LLM server is running, CPU fallback with fewer candidates when one is. Meets the target in the common case; keeps the LLM's priority on the GPU.
3. No reranker (`hybrid`): p95 0.43 s, but −14 points of recall@8.
4. A smaller reranker on CPU: unmeasured; would need a new model and its own evaluation.

**Decision.** Option 2 (chosen by the user). Everything else in ADR-012 stays: < 8 GB for models; LLM sizing as in ADR-008; batch indexing embeds on the GPU in fp32 while no LLM server runs; query embeddings on CPU. New: the reranker prefers `reranker.device` (default `cuda`, fp32, `rerank_candidates` = 40). It falls back to CPU with `reranker.cpu_candidates` (default 10) when CUDA is unavailable or an LLM server answers on `llm.base_url` (loopback probe), and logs the fallback. Every eval run records the device and candidate count actually used. `retrieval.default_mode` becomes `hybrid-rerank`. Whether the reranker can share the GPU with the LLM during `ask` is decided in Phase 5 with the `memory-budget` skill (bge-reranker-v2-m3 is ~2.3 GB in fp32, ~1.1 GB in fp16).

**Consequences.** (+) `search` meets the latency target with the best measured ranking. (+) The LLM keeps the GPU when it runs; nothing changes for the query embeddings. (−) With an LLM server running, `search` is slower (CPU, p95 ≈ 9 s) and less precise (10 candidates); the size of that quality loss is measured separately. (−) VRAM: ~2.3 GB for the reranker while it is loaded. Revisit when Phase 5 sizes the LLM, if fp16 reranking proves equivalent in the eval, or when question set v2 changes the picture.

---

## ADR-014 — As ADR-013, but `search` skips the reranker (plain hybrid) when the GPU is not free

**Date:** 2026-10-17 · **State:** Accepted · **Supersedes:** ADR-013

**Context.** ADR-013 made the reranker fall back to CPU with 10 candidates whenever the GPU is not usable (no CUDA, or an LLM server running). The fallback was then measured on question set v1 (`eval/results/runs.jsonl`): `v1-hybrid-rerank-cpu10` has section recall@8 0.634, MRR 0.565, nDCG@8 0.548 and p95 8.4 s. Plain `hybrid` (`v1-baseline-hybrid`) has recall@8 0.612, MRR 0.428, nDCG@8 0.443 and p95 0.43 s. The CPU fallback finds about the same sections as plain hybrid (+1 question of 44, within noise) and ranks the first hit higher (MRR +0.137), but is 20× slower and still 3× over the latency target. With only 10 candidates the right section often never reaches the reranker (exact_term recall@8 0.462 vs 0.769 on the GPU).

**Options.**
1. Keep the CPU fallback (ADR-013): better first hit, 8.4 s per search.
2. Fall back to plain `hybrid`: same recall within noise, 0.4 s per search, lower MRR.

**Decision.** Option 2 (chosen by the user). Everything in ADR-013 stays except the fallback: when CUDA is unavailable or an LLM server answers on `llm.base_url`, `search` runs `hybrid` without the reranker and logs why. The CPU path remains available as `reranker.fallback = "cpu"` (with `reranker.cpu_candidates`) for whoever prefers the better first hit; the default is `"skip"`. Eval runs record the mode and device actually used.

**Consequences.** (+) `search` always answers within the latency target (0.4 s while an LLM runs, ~1.8 s otherwise). (−) While an LLM server runs (Phase 5 and later), ranking quality drops to plain hybrid (v1: −0.144 recall@8 and −0.163 MRR vs the GPU reranker). Revisit when Phase 5 decides whether the reranker can share the GPU with the LLM (memory-budget skill); if it can, the fallback rarely triggers.

---

## ADR-015 — Qwen3-4B-Instruct-2507 (Q4_K_M, KV q8_0, 8k context) on Ollama, behind `LLMClient`

**Date:** 2026-10-17 · **State:** Accepted · **Supersedes:** ADR-007

**Context.** ADR-007 put the LLM behind our own `LLMClient` and an OpenAI-compatible server on `localhost`, leaving runtime and model open until the hardware was known. ADR-011 measured the hardware: RTX 5060 8 GB with ~4.2 GiB free under the usual desktop load. The memory rules are those of ADR-013/014 (< 8 GB, the LLM has priority on the GPU). Installed runtimes: Ollama 0.18.2 (no models) and LM Studio (Gemma 3 4B Q4_K_M already downloaded); llama.cpp's `llama-server` is not installed. Candidates were checked against their current model cards and `config.json` (2026-10-17), not against training memory.

**Options** (budget at `context_budget_tokens` 6,000 + `max_answer_tokens` 1,024 → 8,192-token runtime context, `memory-budget` skill):
1. **Qwen3-4B-Instruct-2507**, Apache-2.0, text only, no thinking mode. 36 layers, 8 KV heads × 128: KV 144 KiB/token in f16 → 1.1 GiB f16 / 0.56 GiB q8_0 at 8k; weights ~2.4 GB Q4_K_M; total ~3.6 GB with buffers.
2. **Qwen3.5-4B** (Feb 2026), Apache-2.0, hybrid attention (8 of 32 layers keep a KV cache, 4 heads × 256): KV only 0.25 GiB f16 at 8k, but the Ollama build carries vision weights (3.3–4.0 GB) and thinking is on by default (latency, and reasoning outside the evidence unless disabled per call). ~4.2 GB: at the limit of the free VRAM.
3. **Gemma 3 4B** (already local, LM Studio): Gemma Terms of Use (use restrictions), not Apache.

**Decision.** Option 1, chosen by the user, served by Ollama on `127.0.0.1:11434` with `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`, `OLLAMA_CONTEXT_LENGTH=8192` and `OLLAMA_HOST=127.0.0.1`. Tag `qwen3:4b-instruct-2507-q4_K_M`, pinned by manifest digest `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0` (GGUF blob `sha256-85e4a5b7…18b9`); `biblio doctor` checks the digest. Context assembly counts tokens with the model's own tokenizer, pinned like the other models: `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f17c01a7cc42f958dc650907174af0554` (tokenizer files only, downloaded by `biblio models pull`), because Ollama exposes no tokenize endpoint.

**Measured** (2026-10-17, 6,230-token prompt of real chunks, 256 generated tokens): Ollama reports KV cache 612 MiB (q8_0, 8k) and 3.4 GiB total, 37/37 layers on GPU (`ollama ps`: 3.6 GB, 100% GPU). GPU memory 3,604 → 6,950 MiB of 8,151: 1.2 GiB (15%) headroom. Prompt processing 3,715 tok/s, generation 90 tok/s; the first load from disk took ~46 s, later calls start immediately while the model stays loaded.

**Consequences.** (+) Fits fully on the GPU with ≥ 10% margin; ~2–4 s per answer once loaded. (+) Apache-2.0; no thinking mode to disable. (−) No room for the reranker on the GPU while the LLM is loaded: `ask` retrieves with plain `hybrid` (ADR-014); whether a CPU reranker is worth it inside `ask` is measured with the answer evaluation set. (−) The Ollama server must be started with the settings above (`scripts/serve-llm.ps1`); a server started without them uses an f16 cache and may not fit. Revisit if the desktop's VRAM use changes, if Qwen3.5-4B can be run text-only with thinking off, or if answer quality is insufficient (then a 7–8B model with partial offload, with its own ADR).

---

## ADR-016 — Local web UI for `search` and `ask`: stdlib HTTP server + own HTML/JS, loopback only

**Date:** 2026-10-17 · **State:** Accepted

**Context.** Phase 7 lists a local web interface as optional; the user asked for one view to search and one to ask. It must keep the project's guarantees: no network at runtime (ADR-002: no CDN, web fonts or external scripts), never reachable from the LAN, and the GPU rules of ADR-014/015 (the reranker and the LLM never share the GPU). A server also keeps models loaded between queries, which the CLI cannot.

**Options.**
1. Python standard library (`ThreadingHTTPServer`) + own static HTML/CSS/JS shipped in the package: no new dependency, nothing to load from outside; more code for request handling and streaming.
2. FastAPI + uvicorn: convenient routing and streaming; 2–4 new dependencies to review with `offline-guarantee`.
3. Gradio or Streamlit: fastest to build, but Streamlit sends usage statistics by default and Gradio has analytics and external assets; heavy dependency trees. Rejected under ADR-002.

**Decision.** Option 1 (chosen by the user). `biblio serve` binds only to a loopback address (refused otherwise) and rejects requests whose `Host` header is not the loopback address it serves (DNS-rebinding defense); API calls must be JSON `POST`s and no CORS headers are sent, so other sites in the browser cannot call it. All assets are served from the package; a test fails if any of them references an external URL. Library and model text is always inserted as text, never as HTML. The query embedder stays loaded on CPU; the reranker plan is resolved per request exactly as in the CLI (GPU while no LLM server listens, released otherwise: ADR-014), so the UI never holds the reranker on the GPU while the LLM may need it. `ask` streams the answer with server-sent events and reports the same validation as the CLI.

**Consequences.** (+) No new dependency; offline by construction; queries take ~0.5 s with warm models instead of reloading per command. (+) Citations link to the source note (`obsidian://` local URI). (−) Hand-written request handling and a small JavaScript client to maintain. (−) Single user: requests that use models are serialized. Revisit if the UI must grow (e.g. drafts, eval review), where option 2 becomes worth its dependencies.

---

## ADR-017 — Map-reduce is a manual opt-in for `ask`, off by default

**Date:** 2026-10-17 · **State:** Accepted

**Context.** PLAN Phase 5 asked for a map-reduce mode (`rag-prompting` skill) for synthesis questions whose evidence does not fit the budget. Two findings changed the picture. First, the budget is never the problem with Qwen3-4B (ADR-015): with `max_fragments` 6 and a 6,000-token budget, answers use about 2,200 tokens. The failing synthesis questions fail because the small model does not relate 3 guides in one pass. Second, the user graded both strategies on question set v3 (`eval/questionsv3.jsonl`; 5 lookup, 6 synthesis, 3 absent), with grades checked against the full cited fragments (`eval/review/answers-v3-off.md`, `answers-v3-mr.md`; scores in `eval/results/answers.jsonl`):

| | direct (`v3-off`) | map-reduce (`v3-mr`) |
|---|---|---|
| faithfulness (supported claims) | 0.87 | 0.58 |
| fully faithful answers | 80% | 29% |
| lookup coverage | 0.60 | 0.13 |
| synthesis faithfulness / coverage | 0.74 / 0.32 | 0.45 / 0.38 |
| synthesis false abstentions | 2 of 6 | 0 of 6 |
| absent questions abstained | 2 of 3 | 0 of 3 (hallucinated a12, a13) |
| mean latency | 3.1 s | 13.3 s |

Map-reduce answers the cross-guide syntheses the direct path gives up on and covers slightly more key points there, but its citations often do not support its claims, it stops abstaining, and it is 4× slower.

**Options.**
1. Reject map-reduce and keep the code disabled (experimental).
2. Reject it and remove the code.
3. Keep it as a manual opt-in, without rejecting it.

**Decision.** Option 3 (chosen by the user). `llm.map_reduce = "off"` stays the default; `--map-reduce on|auto` on `ask` and `ask-eval run` enables it for a single call, and the CLI warns that its measured faithfulness is lower. The `"auto"` trigger (3+ guides or budget overflow) is kept but not recommended: on v3 it fired on 12 of 14 questions.

**Consequences.** (+) The default path keeps the better measured faithfulness, abstention and latency. (+) Map-reduce remains available for questions the user knows span several guides. (−) When used, answers need checking against their sources (v3: 45% supported claims on synthesis). (−) The code is kept and maintained. Revisit with a larger LLM, a different map/reduce prompt, or a new answer evaluation set.

---

## ADR-018 — Automatic reindexing by polling file metadata (stdlib), GPU if free else CPU

**Date:** 2026-10-17 · **State:** Accepted

**Context.** Phase 7 lists a file watcher for automatic reindexing. The library is an Obsidian vault in iCloud Drive (2,252 notes), edited while Obsidian autosaves every few seconds. `biblio index` is already incremental (0.6 s when nothing changed) but has to be run by hand, and with the LLM server running it refuses to use the GPU (ADR-012). Measured on 2026-10-17: listing the library and reading the modification time and size of every note takes ~360 ms.

**Options.**
1. Poll with the standard library: compare a (mtime, size) snapshot every N seconds and run the incremental indexer when it changed and then stayed stable. No new dependency; ~360 ms per check; reacts within one interval plus the settle time.
2. Native file-system events (`watchdog`): immediate, almost free when idle; one more dependency to review under `offline-guarantee`, and cloud-synced folders produce bursts of noisy events that still need debouncing.
3. No watcher: schedule `biblio index` with the Windows Task Scheduler. No code, but it would refuse while the LLM server runs.

**Decision.** Option 1 (chosen by the user). `biblio watch` runs in the foreground: an initial catch-up index, then a snapshot every `watch.interval_s` (30 s); after a change it waits until the snapshot has been stable for `watch.settle_s` (10 s) so a note is not reindexed mid-edit, and then runs the usual incremental indexer (same lock, same version guards). Embeddings use the GPU when it is free and fall back to CPU when an LLM server listens — the LLM keeps priority (ADR-014) without blocking indexing; an index built in float16 is not updated on CPU (the vectors would differ), and the update is deferred. A run that finds the index locked by another `biblio index` retries on the next cycle; a version mismatch stops the watcher (it needs `--full`).

**Consequences.** (+) Edits in Obsidian reach search within about 30–40 s, with no new dependency and nothing outside the machine. (+) Robust to cloud sync: only metadata is compared, and content hashes still decide what is re-embedded. (−) ~1% of one CPU core while idle (360 ms every 30 s). (−) Each run loads the embedder again (~10 s on GPU, more on CPU) and a CPU fallback re-embeds at ~1.5 chunks/s (one edited note ≈ 20 s). Revisit if the library grows enough that a snapshot takes seconds, or if keeping the embedder warm becomes worth its memory.

---

## ADR-019 — Web UI v2: Conversar, Guías and Practicar pages on the stdlib server

**Date:** 2026-10-17 · **State:** Accepted

**Context.** The UI of ADR-016 has two tabs, "Buscar" and "Preguntar", which make a new user choose an operation before typing, and it exposes no settings. The user asked for a UI that a beginner can use and an advanced user can tune. Three proposals were drawn (design canvas, page "Propuestas"): A, a single search box; B, a conversation with a sources panel; C, a three-pane reading desk. The user chose B and asked for two more pages: one to generate guides (the `draft` flow of Phase 6) and one to practice with quizzes (ADR-020). ADR-016 said to revisit its server choice if the UI grew to cover drafts.

**Options (server).**
1. Keep the standard-library server (ADR-016) and add routes: about ten JSON endpoints and three server-sent-event streams (`ask`, `draft`, quiz generation). No new dependency.
2. Move to FastAPI + uvicorn: easier routing and validation; 2–4 new dependencies to review under `offline-guarantee`.

**Decision.** Option 1, plus a single static app with three pages reached by hash routes (`#/conversar`, `#/guias`, `#/practicar`):
- **Conversar** (proposal B). One thread with one input box; a switch picks "Explicar" (tutor answer with citations) or "Solo ubicar" (search results). Each message is answered on its own, as the CLI does, and the page says so (user's choice: no conversational memory for now; adding it would change retrieval and prompts and needs its own evaluation set). Clicking a citation shows the full cited fragment in a side panel, with a link that opens the note in Obsidian. "Ajustes" holds the advanced settings: precision (`hybrid-rerank` or `hybrid`), number of results, folder scope, and answer strategy (map-reduce flagged experimental, ADR-017). Shortcuts `/ubicar`, `/explicar` and `/en <carpeta>` do the same from the keyboard. Settings and the thread live in the browser's `localStorage` only.
- **Guías.** The two-step `draft` flow in the browser: propose an outline from the library, edit it (rename, reorder, delete, add), then write the sections with streamed progress and their sources. The draft is saved to `drafts/` like the CLI's (ADR-008 banner kept); past drafts are listed.
- **Practicar.** See ADR-020.
- Fonts come from the system stack: nothing is loaded from outside (ADR-016).

**Retrieval impact.** Defaults do not change. The folder scope is a new optional filter on candidate paths, applied before fusion, so filtered queries have no measured baseline; the evaluation keeps measuring the unfiltered default, and `biblio eval` is run after the change to confirm the default numbers did not move (rule 3).

**Consequences.** (+) One input for every question; beginners do not need to know the difference between searching and asking. (+) Sources can be checked without leaving the page. (+) Still no new dependency. (−) More hand-written request handling and JavaScript to maintain; revisit option 2 if the endpoints keep growing. (−) A chat layout may suggest that follow-up questions remember earlier turns; the page states that they do not.

---

## ADR-020 — Practice quizzes generated from library sections, with local history

**Date:** 2026-10-17 · **State:** Accepted

**Context.** The user wants to practice the topics in the library (much of it is organized as syllabi, e.g. `Data Science/Syllabus/...`) with quizzes in the web UI. Generated questions are generated content, so rule 2 applies: each question and each explanation must come from a library section and show it. The local LLM is Qwen3-4B (ADR-015), whose measured answer faithfulness is 0.87 (ADR-017), so grading open answers with it is uncertain. The user chose four question types (multiple choice, true/false, open answer graded by the user, open answer graded by the model) and a local history to know what to review.

**Decision.**
- **Scope.** A quiz draws from one guide, one folder, or a free topic (resolved with the default retrieval). Only content sections are used (no titles, tables of contents or reference lists, as in `draft`). Sections the user failed before, and sections not asked recently, are chosen first.
- **Generation.** One LLM call per question, given the text of one section and the requested type, with a Spanish prompt in `generate/prompts/`. The reply is JSON: question, options and answer key for multiple choice (4 different options, one correct), statement and truth value for true/false, reference answer for open questions, and a short explanation. Replies that fail validation are generated once more and then skipped. Every question stores its source (path, heading path, lines) and is shown with it. Questions appear as they are generated, so the first one is ready after one call.
- **Grading.** Multiple choice and true/false are graded by the code. Open answers graded by the user show the reference answer and the source, and the user marks "La sabía", "A medias" or "No la sabía". Open answers graded by the model are experimental: the model compares the answer with the section and the reference, returns correct / partial / incorrect with a reason, and the user confirms or changes the grade. The agreement rate between model and user is stored and shown; below `practice.grader_min_agreement` (after `practice.grader_min_samples` graded answers) the page warns that the model's grades are unreliable.
- **Quality control.** "La pregunta está mal" marks a question as faulty: it is never asked again, and the share of faulty questions is shown as a measure of generation quality.
- **History.** `data/practice.db`, a SQLite file separate from `index.db`, so a `--full` reindex never deletes it. Tables: quizzes, questions and attempts. A question whose section no longer exists after a reindex is kept in the history but not asked again. Nothing leaves the machine.
- **Resources.** The same LLM server; no new model or dependency. Quiz generation, `ask` and `draft` take turns on the server's model lock.

**Consequences.** (+) Practice built from the user's own notes, each question traceable to its section. (+) The history points to the sections that need review. (+) The model's grading is measured continuously against the user's judgement instead of being trusted. (−) A 4B model will write some wrong or ambiguous questions; the "faulty" flag and the shown source limit the damage but do not prevent it. (−) Each question costs one LLM call (a few seconds), so a 10-question quiz takes on the order of half a minute to generate in full. Revisit the model-graded type if the measured agreement stays low, and the generation prompt if the faulty share is high.

---

## ADR-021 — One-step launcher: setup steps that download only when missing, after asking

**Date:** 2026-10-17 · **State:** Accepted

**Context.** Using the app took about eight manual steps in a fixed order: install uv, Python and the dependencies; `models pull`; set the library path in `config/local.toml`; index (before starting the LLM server, so the GPU is free, ADR-012); install Ollama, pull the model and start it with `scripts/serve-llm.ps1` (ADR-015); `biblio serve`; open the browser. The user asked for a single command that installs and starts everything. Rule 1 allows network access only to `biblio models pull`; dependency installation (`uv sync`) and `ollama pull` are also downloads, though outside `biblio` itself.

**Options.**
1. Document the steps better and keep them manual.
2. A launcher that runs every step on each start, downloading whatever is missing without asking.
3. A launcher that checks each step and downloads only what is missing, announcing the size and asking first; with everything in place it touches no network.

**Decision.** Option 3, as requested by the user. `Biblioteca.cmd` (double-click, Windows) and `scripts/start.sh` (macOS/Linux) make sure uv is installed (offering to install it), then run `uv sync` and `uv run biblio start`. `biblio start`:
1. downloads missing models (the same code as `models pull`);
2. asks for the library path once and saves it in `config/local.toml`;
3. updates the index incrementally (a first full index is announced with its duration and asked for), before the LLM server starts, so it can use the GPU;
4. starts Ollama with the settings of `scripts/serve-llm.ps1`, read from that file so they live in one place, and pulls the pinned model if it is missing, after asking;
5. opens the web UI in the browser;
6. keeps the index up to date with the watcher of ADR-018.
Each download asks first unless `--yes` is passed. Without Ollama, the app runs with search only and says how to add it. Ollama is stopped on exit if `start` started it. With everything already in place, a start uses no network.

**Consequences.** (+) One double-click from a fresh clone to a working app; the safe order (index, then LLM) is built in. (+) Rule 1 still holds for every normal start; downloads happen only at setup and only after consent. (−) The launcher depends on the Ollama executable and its HTTP API (`/api/tags`, `/api/pull` through the `ollama` command), which may change between versions. (−) An Ollama started outside `start` (for example its tray app, with default settings) is used as found; `biblio doctor` still reports a wrong model digest.

---

## ADR-022 — Concept maps by LLM extraction per fragment, checked against the text

**Date:** 2026-10-18 · **State:** Accepted

**Context.** The user asked for a page that, given a topic, draws a mind map of the related concepts in the library (PLAN Phase 8). Like every generated output, the map must show where each part comes from and must not invent content (rule 2). What is available: the default retrieval (the best 8–12 sections for a topic, as `ask` uses), the heading structure of every guide in the index, the internal link graph, and the local LLM, Qwen3-4B (ADR-015). Measured facts that weigh on the choice:
- The link graph is sparse: 565 of 61,753 chunks have internal links, 2,816 links in total. It can connect guides but cannot be the backbone of a map.
- Qwen3-4B is reliable on one fragment at a time and unreliable when it has to relate many: answer faithfulness 0.87 for direct answers vs 0.58 with map-reduce over several guides (ADR-017).
- Generating from one section with a validated JSON reply works: Practicar produces a question in ~3–4 s with one retry and validation (ADR-020); 4 of 4 and 8 of 8 questions came out valid in the live tests.

**Options.**
1. **Headings only, no LLM.** Nodes are the guides and sections that retrieval returns; the hierarchy comes from the heading paths (guide → section → subsection); links between guides come from the link graph. Relations have no labels beyond "contains" and "links to".
   - (+) Instant (under a second), no LLM needed, nothing can be invented: every node is a real heading.
   - (−) It maps the documents, not the concepts: nodes are section titles ("3. El Problema Base: Clasificación Probabilística") rather than ideas ("probabilidad posterior"), and two guides that explain the same concept stay apart.
2. **LLM extraction per fragment, checked against the text (recommended).** One LLM call per fragment returns the concepts it defines or uses and the relations between them (`{from, to, label}`). Code keeps a concept only if it appears in its fragment's text (ignoring accents and case), and a relation only if both ends appear in the same fragment; then merges concepts across fragments by name and by embedding similarity, and builds the hierarchy itself (most connected concepts on level 1).
   - (+) A real concept map with labelled relations, each node and relation tied to the fragment it came from; the model never has to see more than one fragment, which is where it is reliable; invented concepts are removed by the check.
   - (−) ~3 s per fragment, so ~30 s for 10 fragments (first nodes after ~3 s, streamed). The literal-text check can drop synonyms and abstractions the fragment implies but does not name. Needs an evaluation set before adoption.
3. **One LLM call for the whole map.** All the evidence goes into one prompt and the model returns the complete map (nodes, hierarchy and relations) as JSON.
   - (+) Simplest to build; one call, ~10–15 s; the model can name abstract concepts that connect several guides.
   - (−) The model relates many fragments at once, which is where it was measured to fail (ADR-017); relations cannot be checked against a single fragment, and in a long JSON reply a 4B model often breaks the format or the citations. The most likely to invent structure.

**Decision.** Option 2 (chosen by the user). One LLM call per retrieved fragment, with a Spanish prompt asking for the concepts the fragment defines or uses and the labelled relations between them, as JSON. Code keeps a concept only if it appears in its fragment's text (accents and case ignored) and a relation only if both ends appear in the same fragment; an invalid reply is asked once more and the fragment is then skipped (the ADR-020 pattern). Concepts are merged across fragments by normalized name and by embedding similarity (the warm query embedder; threshold in config), keeping every source. The hierarchy is built in code: the concepts most connected to the topic and present in the most fragments on level 1, the rest under the level-1 concept they relate to, other relations as cross-links. Nodes are streamed to the page as each fragment is processed. The map is evaluated on a set of 8–12 topics written by the user, with the key concepts each map must contain, and the share of nodes and relations supported by their sources is graded by the user before the page is adopted (PLAN Phase 8, exit criterion).

**Consequences.** (+) A map of ideas rather than of documents, with labelled relations, in which every node and relation leads to the fragment it came from; the model only ever reads one fragment, where it was measured to be reliable. (+) No new model or dependency. (−) About 3 s per fragment: ~30 s for a 10-fragment map, softened by streaming (first nodes after ~3 s). (−) The literal check drops concepts a fragment implies without naming them, and synonyms the embedding merge misses stay as separate nodes; the evaluation set measures how much this costs. (−) Readability falls past ~40 nodes: a cap per level in config. Revisit with a larger LLM, or if the evaluation shows the check removes too many good concepts.

---

## ADR-023 — Conversation history in a local SQLite file on the server

**Date:** 2026-10-19 · **State:** Accepted

**Context.** ADR-019 kept the Conversar thread in the browser's `localStorage`: one thread of at most 30 messages, erased by «Nueva conversación». The user asked to keep a history of conversations, and chose what to do with it: list and reopen them, search them, rename and delete them, and export one to Markdown.

**Options.**
1. A local SQLite file on the server, `data/conversations.db`, like the practice history (ADR-020): survives clearing the browser, is the same from any browser on the machine, has no practical size limit and can be searched with FTS5.
2. Several threads in `localStorage`: no server change, but lost with the browser's data, tied to one browser and limited to ~5 MB (a few hundred messages with their sources).
3. One Markdown note per conversation in `drafts/conversations/`: readable in Obsidian, but hard to list, search and reopen with the sources in the page.

**Decision.** Option 1 (chosen by the user). This replaces the storage clause of ADR-019; the rest of ADR-019 stands, and in particular each message is still answered on its own (the history is for reading, not memory for the model). The server records each message when its answer is complete (`/api/search` and `/api/ask` take an optional `conversation_id`; without one a new conversation starts), with everything the page shows: question, mode, folder, strategy, answer, citations and the full text of its sources, or the error. Conversations are titled with their first question and can be renamed; deleting one deletes its messages. Search uses an FTS5 table over questions and answers with the same accent folding as the index (`unicode61 remove_diacritics 2`). Export writes `drafts/conversations/<title>.md` (outside the library, never indexed) with each answer, its citations and its sources, and the page also offers it as a download. The browser keeps only the id of the open conversation. Nothing leaves the machine.

**Consequences.** (+) Questions and answers are kept and can be found again, with their sources. (+) Same pattern as the practice history: a separate file, so `index --full` never touches it. (−) The history grows without limit (about 10–15 KB per answered question with its sources); deleting is manual. (−) A thread kept in `localStorage` before this change is not imported.

---

## ADR-024 — Concept-map history in a local SQLite file on the server, as ADR-023

**Date:** 2026-10-19 · **State:** Accepted

**Context.** Maps (ADR-022) were kept only in the server's memory (the last 20, to be saved with «Guardar») and lost when the app stopped. The user asked for the same as the conversation history: a list of the maps generated, which they can reopen, and a button to show or hide it.

**Options.** The same three as ADR-023 (local SQLite on the server, the browser's storage, files in `drafts/`). The user chose the first for conversations, for reasons that hold here too: maps carry the full text of their sources, survive the browser's data being cleared, and are the same from any browser on the machine.

**Decision.** `data/maps.db`, apart from the index. Every map is stored when it is complete, with its topic, the map (nodes, edges, layout) and its source fragments, so reopening one shows it as it was without calling the LLM again; an expanded map («Expandir») is stored as a map of its own. The list is newest first and can be searched by topic or by the names of the concepts in the map (FTS5 with the index's accent folding); a map can be deleted. «Guardar» now writes any stored map to `drafts/maps/`, not only those still in memory. The page's «Historial» button shows or hides the list on Mapas as it does on Conversar, each page remembering its own choice in the browser.

**Consequences.** (+) Maps that took ~30 s to build are kept and reopen instantly. (+) The same pattern and code shape as the conversation history. (−) One more file that grows without limit (a map with its sources is ~50–100 KB); deleting is manual. (−) A stored map reflects the library when it was built; it is not refreshed when notes change.
