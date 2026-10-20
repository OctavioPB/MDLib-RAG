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
- **A conversation keeps its topic** (ADR-038). The first message sets it («Tema: …» under the
  title, editable). Each message is read with the topic and the last three turns before searching:
  «¿Qué métodos se usan?» becomes «¿Qué algoritmos de ciencia de datos se usan en química?», and
  «hablamos de química» brings the conversation back to its topic. Under each answer, «Entendí: …»
  shows how the message was understood and what was searched.
- **Questions that join areas reach each of them**: the message is searched as written, and the
  model's searches per area add two more fragments the question alone missed (the tutor reads
  eight instead of six). The answer explains each side from your notes and then how they connect;
  a connection your notes do not write is marked **Relación propuesta** and cites both sides. If
  you ask for an example and your notes have none, an invented one is marked **Ejemplo ilustrativo
  (no está en tus notas)** and carries no citation.
- **Another topic**: if a message has nothing to do with the conversation's topic, the tutor asks
  first — **Nueva conversación con esta pregunta** or **Responder aquí** (answered on its own,
  without changing the topic). Planning adds ~3 s to each answer.

**Nivel** (ADR-032): next to «Explicar» in Conversar, choose how the tutor explains:
**Principiante** (the simplest words, ending with one everyday comparison, marked «Es como…», the
only uncited sentence it may add), **Intermedio** (the default: your guides' own terms, going
deeper) or **Avanzado** (technical terms and formulas, the topic related to others only through
what your notes say, as a master's-degree teacher). It applies from the next message, each answer
shows its level, and reopening a conversation restores the level of its last message. All three
use the same notes, citations and «No encuentro esto en tu biblioteca» rule; formulas in answers
are drawn with KaTeX. `uv run biblio ask "…" --level advanced` does the same in the terminal.

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

**Guías** opens on two tiles, **Plan de estudio** and **Escribir una guía**, with **Sigue con: …**
on top once you have a plan.

- **Plan de estudio** (ADR-028): write a folder of your syllabus (e.g. `Data Science/Syllabus/LLMs`)
  and the model proposes the order to study its guides, with a reason for each place (~15 s; the
  guide that links to all the others goes first). Folders with more than 30 guides are ordered by
  subfolder, then inside each. Each guide shows its state: **Por leer**, **Leída** (you mark it),
  **Practicada** (any answer on its sections) or **Dominada** (60 % of its sections practised and
  remembered today at 85 % or more). The first guide not yet practised is **Sigue aquí**. Each one
  can be opened in Obsidian, marked as read, practised (a quiz on that guide) or explained
  (Explícamelo tú), and moved up or down; **Proponer otro orden** asks the model again. Plans are
  kept in `data/practice.db`, with your order; guides added to the folder later appear at the end
  as new.

**Temas pendientes** (ADR-029, ADR-030), also in the Guías hub: whenever the tutor answers «No
encuentro esto en tu biblioteca», or a map comes out thin (fewer than 6 concepts, or its topic is
not written in any of the notes it read), the subject is noted as a pending topic. The model turns
the question into a short topic («Preparación de roux para bechamel»); a new question on a subject
already noted is added to it («3 veces»): embeddings pick the similar topics and the model decides
whether it is the same subject. The answer or map says «Apuntado como tema pendiente» with
**Deshacer**. In Guías each topic offers **Escribir guía** (the topic filled in; the topic then
shows «En borrador»), **Resuelto** and **Descartar**.

**Biblioteca** (ADR-033), in the top menu, lists every note of your library by folder, with the
number of notes in each, and a box that finds a note by title or folder as you type (accents and
capitals do not matter). The notes you read last are listed first; notes marked as read show
«✓ leída». Any of them opens in the reader.

**Moving the library** (ADR-039): on the PC, Ajustes › Biblioteca shows «Ubicación: …» with
**Cambiar**. Paste the folder or find it with **Examinar**, then **Comprobar**: if most notes match
the index (the same vault moved or synced elsewhere), the index is reused and the change takes
seconds; if it is another library, it is indexed in full (the page gives the estimate) while the
current one keeps working, and it replaces it only if it completes. The new path is saved in
`config/local.toml` and used at once, including automatic reindexing. In the terminal:
`uv run biblio library` shows it and `uv run biblio library set "<ruta>"` changes it. Paired phones
and tablets do not see or change it.

**Reset** (ADR-040): in Ajustes › Memoria de la app, on the PC, **Reset**
starts the app over without its memory: conversations, maps and diagrams, quizzes and reviews,
«Explícamelo tú», study plans and read marks, pending topics, dictionaries and the `drafts/` folder.
A «¿Estás seguro?» dialog lists what will go, with counts. Your library, its index, the models and
`config/local.toml` are never touched, and everything erased is moved to `data/backups/reset-<date>/`
in case you change your mind. `uv run biblio reset` does the same with the app closed.

**Leer** (ADR-031) opens a note inside the app, at the cited line, from the **Leer** button next to
every source (Conversar, Mapas, Practicar, Explícamelo tú and the study plan). Formulas are drawn
with KaTeX, which ships with the app, so nothing is downloaded; `[[links]]` open the linked note
in the reader. Select a paragraph and choose:

- **Explícame esto**: an explanation in a side panel, citing the passage [1], the section it is
  in [2] and whatever else your notes say about it. It is kept in the Conversar history, where
  **Seguir en Conversar** continues it.
- **Pregúntame sobre esto**: three questions written from that passage only (multiple choice, true
  or false, open), answered in the panel; they count as practice of that section for your reviews.

The reader also offers **Abrir en Obsidian** and **Marcar como leída** (for the study plan).

**Mapas** draws a topic from your notes as one of five diagrams (ADR-022, ADR-036, ADR-037):

- Write a topic, choose the type and press **Crear**. **Sugerir tipo** asks the model which type fits
  the topic (~1 s) and selects it; you confirm with **Crear**.
  - **Conceptos**: the concept map below, around the topic.
  - **Mapa conceptual**: the same concepts as a tree read left to right, every arrow with its
    linking phrase («es un tipo de», «se calcula con»).
  - **Flujo**: the steps and decisions of a process. Each note that describes the process is a
    version of its own (never mixed); the bar above the drawing switches between them. A dashed
    arrow is an order the note does not write; select a step to move it up or down.
  - **Secuencia**: the messages between participants, in order; a reply is dashed, and a frame marks
    the case in which messages happen. One version per note, as for flows.
  - **Causa-efecto**: the causes of the topic (the effect) as a fishbone. The model groups them into
    categories; **Usar las 6M** or **Mis categorías…** regroup them without reading the notes
    again, and a selected cause can be moved to another category.
- Every step, message and cause carries a quote that must be written in its fragment; the panel shows
  it with its citation, **Leer en la nota** and **Explícamelo tú**. **Copiar Mermaid** copies the
  diagram as a Mermaid block that Obsidian draws, and **Guardar** writes it to `drafts/maps/` with
  that block, its outline and its sources, plus a Canvas file.
- For concept maps, the ten most relevant sections are read one by one; the map grows as each is
  read (~3 s each, ~30 s in all). Flows, sequences and fishbones read eight (~15-35 s).
- Each concept comes from a fragment where it is written, and each relation was stated in a fragment;
  anything the model names that is not in the text is dropped. The topic is in the centre, the most
  connected concepts around it, and their sub-concepts further out.
- Click a concept to see its relations and the fragments where it appears. **Expandir** makes a map of
  that concept (the trail above lets you go back), **Explicar** asks about it in Conversar, and
  **Practicar** makes a quiz on it (**Practicar esto** once the map is kept: a quiz on the very
  sections the concept comes from).
- **Drag a concept** to move it; its lines and labels follow and the relations stay as they are
  (with a concept focused, the arrow keys move it too, Shift for bigger steps). The layout is kept
  with the map and used by **Guardar**; **Reorganizar** goes back to the automatic layout.
- **Dominio** (ADR-026) colours each concept by how well you remember its sections today, from your
  practice: green mastered, amber half-remembered, red to review, dashed grey never practised. The
  panel gives the percentage.
- **Concepto** shows or hides the panel of the selected concept, giving its width to the drawing.
- **Esquema** shows the same map as an indented list with each relation written as a sentence.
- **Every map is kept** (ADR-024) in `data/maps.db`: the list on the left shows them newest first,
  can be searched by topic or by the concepts in them, and reopens a map instantly, without the LLM.
  **Borrar** deletes the open map; **Historial** in the top bar shows or hides the list.
- **Guardar** writes it to `drafts/maps/` as Markdown and as an Obsidian Canvas file (`.canvas`):
  copy that file into your vault to open it in Obsidian, with each concept linked to its notes.
  **Descargar SVG** saves the drawing. `uv run biblio map "tema"` prints the outline in the terminal.

The model can still state a wrong relation between two real concepts, turn a description into a
«step» or a symptom into a «cause»; the quote shows where each piece comes from. The quality of every
type is to be measured on an evaluation set before it is considered finished (PLAN Phase 8 and 9.12).

**Practicar** makes quizzes from your own notes (ADR-020). The page opens on what is due today, with
one button to start it; below, three ways to practise something else (**Cuestionario**,
**Explícamelo tú**, **Repasar mis fallos**), each on its own screen with **← Practicar** to come back,
and a strip with your progress (**Ver todo** for the details).


- Choose where the questions come from: one guide, one folder of your syllabus, or a topic.
- Choose how many (5, 10 or 20) and of which kinds: multiple choice, true or false, open answers you
  grade yourself against a reference answer, and open answers graded by the model (experimental).
- Each question is written from one section of your notes, appears as soon as it is ready (~3–4 s
  each), and after you answer shows the explanation, the reference answer and that section, with a
  link to it in Obsidian. **La pregunta está mal** discards a faulty question for good.
- The model's grades of open answers can be accepted or changed; how often it agrees with you is
  shown, with a warning if it agrees too little to be trusted.
- **Hoy toca repasar** (ADR-025): every section you have practised is scheduled again just before
  you are likely to forget it (the FSRS algorithm): an answer right pushes it further away, a miss
  brings it back soon. The card shows how many are due today and the minutes it takes, up to 20 a
  day, the ones you are closest to forgetting first; the number also shows next to **Practicar** in
  the top bar. Each review writes a new question on the section; with the LLM off it asks the last
  stored one again.
- **Diccionario** (ADR-034): write a topic, or choose a guide, and get its main concepts with a
  short definition and the literal quote of your note that backs it, plus, where there is code or
  mathematics, its functions (name, signature as written, what it does) and formulas (drawn with
  KaTeX, what they compute). Every entry is checked against the text and links to its section in
  the reader. A topic takes ~40 s; a whole guide ~3.5 s per section (up to 60). A dictionary reads
  like a glossary: pick it at the top, browse its index by list and letter or filter it, and the
  selected entry shows on the right. **Repasar como fichas** shows each card (the ones you missed
  first), you turn it over and say whether you knew it, which counts for your reviews and marks the
  term in the index; **Exportar** downloads Markdown or Anki (in Anki: File › Import).
- **Tu progreso** shows your score per guide and the sections you missed; **Repasar mis fallos**
  makes a quiz from those. Sections you missed are asked first, then new ones.
- **Explícamelo tú** (ADR-027, experimental): write a concept and explain it in your own words, as
  if to someone who knows nothing about it. While you write, the app reads the five sections of your
  notes most about it and takes their key ideas (~10 s). Then **Comprobar** (~6 s) shows which key
  ideas you covered, half covered or left out, and which of your claims your notes back, contradict
  or do not mention (which does not mean they are false), each with the literal quote of the note
  it comes from. Accept or change the proposed grade and it counts as practice of those sections,
  for your reviews and the mastery map. **¿Es justo?** on every point tells the app how often it
  gets it right; **Tu progreso** shows the share. Concept panels in Mapas have an
  **Explícamelo tú** button too.

The history stays on this machine in `data/practice.db`, apart from the index, so rebuilding the
index never deletes it. The answer to a question is never sent to the page before you answer it.

**Ajustes** (the gear at the top right, on every page) holds everything that is not studying:
**Apariencia** (Claro, Sistema, Oscuro), **Biblioteca** (its location, with **Cambiar**, and
**Actualizar ahora**), **Dispositivos** (LAN mode) and **Memoria de la app** (**Reset**). Location,
devices and reset appear on the PC only. The top bar keeps just the pages, the status and the gear.

**Actualizar ahora** (in Ajustes) **updates the library**: an incremental reindex,
like `biblio index`, with its progress in the status at the top; only new, changed or deleted notes are
processed (about a second when nothing changed). It embeds on the GPU when free and on the CPU while
the LLM server runs, and refuses politely if another index run is in progress.

**Apariencia** in Ajustes switches between a light theme, a dark one, or following your system
(the default); the choice is remembered in the browser.

The server listens only on `127.0.0.1`, rejects requests addressed to any other host name, and loads
nothing from the internet (ADR-016). The query embedder stays loaded, so a search takes about half a
second.

## Phone and tablet on your home network

LAN mode (ADR-035) lets a phone or tablet on your home network use the app, with everything still
running on this PC. It is off by default; turn it on with `biblio start --lan` (or `serve --lan`), or
for good with this in `config/local.toml`:

```toml
[web]
lan = true
```

When it starts, the terminal shows the PC's address and a QR code. In the app on the PC, Ajustes ›
Dispositivos › **Conectar un dispositivo** shows a fresh QR code (one use, 10 minutes) and the
paired devices, each with **Quitar**. Scan the code with the phone on the same Wi-Fi: the app opens
and the device stays paired (180 days); next time the address is enough. Devices that were not
paired, and any address outside private networks, get nothing, not even the page. «Abrir en
Obsidian» is hidden on other devices. The first time, Windows asks to allow Python on the network:
allow **private networks** only. The connection is not encrypted, so use it on your own network,
not on a shared one.

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
uv run biblio eval-feynman --tag my-run      # «Explícamelo tú»: expected gaps reported, false alarms → eval/results/feynman_runs.jsonl
```

Current numbers (section level, 64 questions; `PLAN.md` has the history):

| | recall@8 | MRR | p95 |
|---|---|---|---|
| hybrid + reranker (default) | 0.770 | 0.597 | 1.85 s |
| hybrid (LLM server running) | 0.647 | 0.467 | 0.49 s |

`eval/feynman.jsonl` holds your own explanations of concepts in your library, each with the
ideas it leaves out and the claims your notes do not back; `eval/feynman.example.jsonl` shows the
format.

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
