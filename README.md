# biblio-rag

A 100% local search engine, tutor and writing assistant over a personal library of Markdown notes
(built for an Obsidian vault of ~2,250 notes). No data leaves the machine and no external API is
called at runtime.

- **Search:** where in my notes is something explained?
- **Tutor:** explain a topic by combining several guides, citing every claim.
- **Writing assistant:** draft a new guide consistent with the existing ones.
- **Web interface:** the same in the browser, with a sources panel, plus practice quizzes generated
  from your notes.

Project documents: `CLAUDE.md` (rules and conventions), `PLAN.md` (roadmap and log), `DECISIONS.md`
(architecture decisions, referred to below as ADR-NNN).

## Requirements

- Python 3.11+ managed by [uv](https://docs.astral.sh/uv/) (the system Python on macOS cannot load
  SQLite extensions).
- Optional but recommended: an NVIDIA GPU. PyTorch is the CUDA 13.0 build; without CUDA everything
  runs on the CPU, but indexing is ~20× slower.
- For `ask`, `draft` and the tutor in the web UI: [Ollama](https://ollama.com) with the pinned model
  (see [Tutor](#tutor-biblio-ask)).
- Disk: ~4.3 GB for the models in `models/`; the index of 2,252 notes is ~400 MB.

Developed and measured on Windows 11 with a Ryzen 7 5700X, 32 GB of RAM and an RTX 5060 (8 GB).

## Quick start

**Windows:** double-click **`Biblioteca.cmd`**. **macOS / Linux:** run `scripts/start.sh`.

That is all. The first time it asks for the folder of your notes and installs what is missing; after
that it starts in a few seconds and opens the app in your browser. Keep its window open while you use
the app; close it (or press Ctrl+C) to stop everything.

What it does, in order (ADR-021), skipping whatever is already in place:

1. installs [uv](https://docs.astral.sh/uv/) if needed (it asks first), then Python and the
   packages (`uv sync`);
2. downloads the embedding model and the reranker (~4.3 GB, asks first);
3. asks for your library folder once and saves it in `config/local.toml`;
4. updates the index; the first full index is announced with its duration and asked for;
5. starts Ollama with the measured settings and downloads the LLM if missing (~2.5 GB, asks first);
   without Ollama installed, the app runs with search only and tells you how to add it;
6. opens the web interface and keeps the index up to date while you edit your notes.

Downloads happen only when something is missing and only after you agree (`--yes` agrees to all);
with everything installed, starting the app uses no network. The same from a terminal:

```bash
uv run biblio start              # --yes, --no-llm, --no-browser, --port N
```

Step by step instead, if you prefer:

```bash
uv sync
uv run biblio models pull        # the only biblio command that uses the network
uv run biblio doctor             # hardware, models, SQLite extensions, offline mode
uv run biblio index ~/path/to/library
uv run biblio search "intervalo de confianza bootstrap"
```

## Configuration

`config/default.toml` holds every default, with comments. Put machine-specific values (library path,
LLM endpoint) in `config/local.toml`, which is gitignored and merged over the defaults key by key:

```toml
[paths]
library = "D:/notes/library"
```

Relative paths in the config are resolved against the project root. With `paths.library` set, the
commands below need no path argument.

## Models

`biblio models pull` downloads the embedding model (`BAAI/bge-m3`) and the reranker
(`BAAI/bge-reranker-v2-m3`) into `models/` at the revisions pinned in the config, and writes
`models/manifest.json` (repo, revision, date, license). At runtime models are loaded only from that
directory, with Hugging Face offline mode forced. `biblio models status` shows what is present.

## Indexing

```bash
uv run biblio index            # incremental: only changed files are re-embedded
uv run biblio index --full     # rebuild from scratch (built aside, swapped in only on success)
uv run biblio watch            # keep the index up to date while you edit (Ctrl+C to stop)
```

A full index of 2,252 notes takes ~38 min on the GPU; an incremental run with no changes takes under
a second. Changing the embedding model, the chunker or its thresholds makes `index` refuse and ask for
`--full`, so vectors from different versions are never mixed.

`biblio watch` catches up on changes made since the last index, then checks the library every 30 s
(file dates and sizes only, ~0.4 s). After a change it waits until nothing has changed for 10 s, so a
note being edited in Obsidian is indexed once, and then runs the incremental index. It embeds on the
GPU when it is free and on the CPU while the LLM server runs (one edited note takes ~20 s on the CPU).
Interval and wait are in `[watch]` in the config (ADR-018).

`biblio chunks <file.md>` prints how one note is split, for checking by eye.

## Search (`biblio search`)

```bash
uv run biblio search "regularización L1 vs L2"
```

Returns the best sections with their location (`path/file.md › Section › Subsection`) and line
numbers. Retrieval combines semantic search and BM25 keyword search (fused with RRF), then reorders the
best 40 candidates with a cross-encoder reranker on the GPU (ADR-006). While the LLM server runs, the
GPU is the LLM's and search skips the reranker (ADR-014). `--mode` picks another retriever (`dense`,
`bm25`, `hybrid`, `hybrid-rerank`).

## Tutor (`biblio ask`)

The tutor needs a local LLM server. Pull the pinned model once and start the server with the settings
its memory budget was measured with (ADR-015):

```powershell
ollama pull qwen3:4b-instruct-2507-q4_K_M
powershell scripts/serve-llm.ps1
uv run biblio ask "¿Qué diferencia hay entre un Data Warehouse y un Data Mart?"
```

Every claim is cited as `[n]` and the cited sections are listed after the answer. If the library does
not contain the answer, the tutor says so instead of improvising, and answers whose citations do not
check out are flagged. `--show-context` lists every fragment the model received. `--map-reduce on`
summarizes each guide separately before answering; it is off by default because its measured
faithfulness is lower (ADR-017).

## Writing assistant (`biblio draft`)

Two steps, with you in between (ADR-008: the result is a skeleton to edit, not a finished guide):

```powershell
uv run biblio draft "Validación cruzada en machine learning"
# edit drafts/validacion-cruzada-en-machine-learning.outline.md: reorder, delete, add sections or notes
uv run biblio draft "Validación cruzada en machine learning" --outline drafts/validacion-cruzada-en-machine-learning.outline.md
```

The outline imitates the structure of your closest guides. Each section retrieves its own evidence,
cites it with `[n]`, and says so when the library has nothing for it; the draft ends with the list of
sources used. Drafts are written to `drafts/`, outside the library, so they are never indexed.

## Web interface (`biblio serve`)

```powershell
uv run biblio serve        # then open http://127.0.0.1:8765
```

**Conversar** is a single thread with one input box:

- **Explicar** answers with numbered citations (needs the LLM server).
- **Solo ubicar** lists the sections where a topic appears.
- Click a citation or a result to read the whole original passage in the side panel, with a link that
  opens the note in Obsidian.
- Each message is answered on its own: follow-up questions do not remember earlier turns (ADR-019).

**Ajustes** holds the advanced settings: precision (reranker on or off), number of results, the folder
to search in, and the answer strategy ("Por guía" is the experimental map-reduce). The same can be
typed at the start of a message:

| Shortcut | Effect |
|---|---|
| `/ubicar <texto>` | search only, no LLM |
| `/explicar <pregunta>` | tutor answer |
| `/en <carpeta> <texto>` | limit to one folder, e.g. `/en Data Science/Syllabus regularización` |

**Your conversations are kept** (ADR-023). Every question and its answer, with its sources, is saved
in `data/conversations.db` on this machine. The list on the left shows them newest first, titled with
their first question; click one to reopen it and keep asking in it. The search box finds conversations
by any words in their questions or answers (accents and case ignored). **Renombrar**, **Exportar**
(a Markdown note with the answers, citations and sources, saved to `drafts/conversations/` and
downloaded) and **Borrar** act on the open one; **Nueva conversación** starts another. **Historial**
in the top bar shows or hides the list. The settings stay in the browser.

**Guías** is the `draft` flow in the browser:

1. Write a topic and press **Proponer índice**: an outline is proposed from your notes, imitating
   your closest guides.
2. Edit it: rename, reorder, remove or add sections, and add notes on what each should cover.
3. **Escribir borrador** writes it section by section while you watch, with clickable citations and
   the full text of every source at the end. **Copiar Markdown** and **Descargar .md** take it out.

Guides are the same files `biblio draft` uses (`drafts/<tema>.outline.md` and `drafts/<tema>.md`), so
the list on the left shows drafts made from the CLI too, and either can continue the other. One draft
or quiz is generated at a time; the other pages keep working meanwhile. `#/guias/<tema>` links straight
to one guide.

**Mapas** draws a concept map of a topic from your notes (ADR-022):

- Write a topic and press **Crear mapa**. The ten most relevant sections are read one by one; the map
  grows as each is read (~3 s each, ~30 s in all).
- Each concept comes from a fragment where it is written, and each relation was stated in a fragment;
  anything the model names that is not in the text is dropped. The topic is in the centre, the most
  connected concepts around it, and their sub-concepts further out.
- Click a concept to see its relations and the fragments where it appears. **Expandir** makes a map of
  that concept (the trail above lets you go back), **Explicar** asks about it in Conversar, and
  **Practicar** makes a quiz on it.
- **Drag a concept** to move it; its lines and labels follow and the relations stay as they are
  (with a concept focused, the arrow keys move it too, Shift for bigger steps). The layout is kept
  with the map and used by **Guardar**; **Reorganizar** goes back to the automatic layout.
- **Concepto** shows or hides the panel of the selected concept, giving its width to the drawing.
- **Esquema** shows the same map as an indented list with each relation written as a sentence.
- **Every map is kept** (ADR-024) in `data/maps.db`: the list on the left shows them newest first,
  can be searched by topic or by the concepts in them, and reopens a map instantly, without the LLM.
  **Borrar** deletes the open map; **Historial** in the top bar shows or hides the list.
- **Guardar** writes it to `drafts/maps/` as Markdown and as an Obsidian Canvas file (`.canvas`):
  copy that file into your vault to open it in Obsidian, with each concept linked to its notes.
  **Descargar SVG** saves the drawing. `uv run biblio map "tema"` prints the outline in the terminal.

The model can still state a wrong relation between two real concepts; the map's quality is to be
measured on an evaluation set before it is considered finished (PLAN Phase 8).

**Practicar** makes quizzes from your own notes (ADR-020):

- Choose where the questions come from: one guide, one folder of your syllabus, or a topic.
- Choose how many (5, 10 or 20) and of which kinds: multiple choice, true or false, open answers you
  grade yourself against a reference answer, and open answers graded by the model (experimental).
- Each question is written from one section of your notes, appears as soon as it is ready (~3–4 s
  each), and after you answer shows the explanation, the reference answer and that section, with a
  link to it in Obsidian. **La pregunta está mal** discards a faulty question for good.
- The model's grades of open answers can be accepted or changed; how often it agrees with you is
  shown, with a warning if it agrees too little to be trusted.
- **Tu progreso** shows your score per guide and the sections you missed; **Repasar mis fallos**
  makes a quiz from those. Sections you missed are asked first, then new ones.

The history stays on this machine in `data/practice.db`, apart from the index, so rebuilding the
index never deletes it. The answer to a question is never sent to the page before you answer it.

The button next to the notes count (top right) **updates the library**: an incremental reindex,
like `biblio index`, with its progress in the counter; only new, changed or deleted notes are
processed (about a second when nothing changed). It embeds on the GPU when free and on the CPU while
the LLM server runs, and refuses politely if another index run is in progress.

The buttons at the top right switch between a light theme, a dark one, or following your system
(the default); the choice is remembered in the browser.

The server listens only on `127.0.0.1`, rejects requests addressed to any other host name, and loads
nothing from the internet (ADR-016). The query embedder stays loaded, so a search takes about half a
second.

## GPU and memory

The model budget is under 8 GB. The GPU serves whichever workload is active, and the LLM has priority
(ADR-014):

| | LLM server stopped | LLM server running |
|---|---|---|
| `index` | GPU (~2.7 GiB of VRAM) | refuses the GPU: stop the server or pass `--device cpu` |
| `watch` | GPU | CPU, automatically |
| `search` | hybrid + reranker on the GPU (p95 1.9 s) | hybrid without reranker (p95 0.5 s) |
| `ask`, `draft` | need the server | GPU for the LLM |

Query embeddings always run on the CPU. Start Ollama with `scripts/serve-llm.ps1`: started any other
way it uses a larger cache and may not fit (ADR-015). `biblio doctor` checks that the served model is
the pinned one.

## Evaluation

Retrieval changes are only accepted with before/after numbers (CLAUDE.md, rule 3). The question sets
are written by you, not generated.

```bash
uv run biblio eval --tag my-change           # retrieval: recall@k, MRR, nDCG, latency → eval/results/runs.jsonl
uv run biblio compare v2-hybrid-rerank my-change
uv run biblio candidates my-change           # what the search returned for missed questions, for you to judge
uv run biblio ask-eval run --tag my-run      # tutor answers → a review sheet in eval/review/ for you to grade
uv run biblio ask-eval score my-run          # faithfulness, coverage, abstention → eval/results/answers.jsonl
```

Current numbers (section level, 64 questions; `PLAN.md` has the history):

| | recall@8 | MRR | p95 |
|---|---|---|---|
| hybrid + reranker (default) | 0.770 | 0.597 | 1.85 s |
| hybrid (LLM server running) | 0.647 | 0.467 | 0.49 s |

Agreed targets: recall@8 ≥ 0.75, MRR ≥ 0.55, p95 ≤ 3 s. The tutor's measured faithfulness (claims
supported by their citations) is 0.87 on the 14-question answer set.

## Privacy and offline guarantees

- Only `biblio models pull` opens connections outside `localhost`; everything else runs with
  Hugging Face offline mode forced, and the LLM must be on a loopback address.
- The test suite blocks the network (`pytest-socket`).
- The web UI is loopback-only and a test fails if any page asset references an external URL.
- Drafts, the index and the practice history stay in this project's folders.

## Development

```bash
uv run pytest                     # network blocked; tests needing real models are marked `models`
uv run pytest -m models           # also run those
uv run ruff check . && uv run ruff format .
uv run mypy src                   # strict
```

Before changing the architecture, read `DECISIONS.md`; new decisions are recorded there as ADRs.
