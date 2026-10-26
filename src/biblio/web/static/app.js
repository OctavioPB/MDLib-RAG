// Biblioteca — local web UI (ADR-016, ADR-019). Library and model text is only ever inserted with
// textContent, never as HTML. Settings and the thread live in this browser's localStorage only.
"use strict";

const $ = (id) => document.getElementById(id);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

const store = {
  get(key, fallback) {
    try {
      const raw = localStorage.getItem(key);
      return raw === null ? fallback : JSON.parse(raw);
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      /* private mode or full storage: the page still works, it just forgets */
    }
  },
};

/** Show `message` in the error paragraph `id`, or hide it when empty. */
function showError(id, message) {
  $(id).textContent = message || "";
  $(id).hidden = !message;
}

async function postJSON(path, body, signal) {
  return fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
}

async function errorText(response) {
  try {
    return txServer((await response.json()).error || response.statusText); // ADR-046
  } catch {
    return response.statusText;
  }
}

function* sseEvents(buffer) {
  let index;
  while ((index = buffer.text.indexOf("\n\n")) !== -1) {
    const raw = buffer.text.slice(0, index);
    buffer.text = buffer.text.slice(index + 2);
    let event = "message";
    let data = "";
    for (const line of raw.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data += line.slice(5).trim();
    }
    yield [event, data ? JSON.parse(data) : {}];
  }
}

// --- minimal Markdown → DOM (model output: paragraphs, headings, lists, code, tables) -----------
// Builds nodes with textContent only. `cite(n)` returns the node for one citation number ("?" for
// an invalid one); math stays as written.

const INLINE_MD = /(\*\*[^\n]+?\*\*|`[^`\n]+`|\[(?:\d+(?:\s*[,;]\s*\d+)*|\?)\])/g;
// a table row may end with citations after its last pipe: `| a | b | [2]`
const TABLE_ROW = /^\s*\|.*\|\s*(?:\[[^\]\n]*\]\s*)*$/;
const TABLE_RULE = /^\s*:?-{2,}:?\s*$/;

function tableNode(rows, cite) {
  const parsed = rows.map((row) => {
    const cells = row.trim().replace(/^\|/, "").split("|");
    const tail = cells.pop().trim(); // "" or citations after the last pipe
    if (tail && cells.length) cells[cells.length - 1] += ` ${tail}`;
    return cells.map((c) => c.trim());
  });
  const table = el("table", "md-table");
  const hasHeader = parsed.length > 1 && parsed[1].every((c) => TABLE_RULE.test(c));
  const addRow = (parent, cells, tag) => {
    const tr = el("tr");
    for (const cell of cells) {
      const td = el(tag);
      td.append(...inlineNodes(cell.replace(/\*([^*\n]+)\*/g, "$1"), cite));
      tr.append(td);
    }
    parent.append(tr);
  };
  if (hasHeader) {
    const head = el("thead");
    addRow(head, parsed[0], "th");
    table.append(head);
  }
  const body = el("tbody");
  for (const cells of parsed.slice(hasHeader ? 2 : 0)) addRow(body, cells, "td");
  table.append(body);
  const wrap = el("div", "table-wrap"); // wide tables scroll inside their box
  wrap.append(table);
  return wrap;
}

function inlineNodes(text, cite) {
  const out = [];
  let last = 0;
  for (const match of text.matchAll(INLINE_MD)) {
    if (match.index > last) out.push(document.createTextNode(text.slice(last, match.index)));
    const token = match[0];
    if (token.startsWith("**")) out.push(el("strong", null, token.slice(2, -2).replace(/\*/g, "")));
    else if (token.startsWith("`")) out.push(el("code", null, token.slice(1, -1)));
    else for (const part of token.slice(1, -1).split(/[,;]/)) out.push(cite(part.trim()));
    last = match.index + token.length;
  }
  if (last < text.length) out.push(document.createTextNode(text.slice(last)));
  return out;
}

function markdownNode(text, cite, className = "md") {
  const root = el("div", className);
  const lines = text.replace(/\r/g, "").split("\n");
  let paragraph = [];
  let list = null;
  const flush = () => {
    if (paragraph.length) {
      const p = el("p");
      p.append(...inlineNodes(paragraph.join(" "), cite));
      root.append(p);
      paragraph = [];
    }
  };
  const block = (node) => {
    flush();
    list = null;
    root.append(node);
  };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const fence = line.match(/^\s*(```|~~~)/);
    if (fence) {
      const body = [];
      while (++i < lines.length && !lines[i].trim().startsWith(fence[1])) body.push(lines[i]);
      block(el("pre", "code-block", body.join("\n")));
      continue;
    }
    if (TABLE_ROW.test(line)) {
      const rows = [line];
      while (i + 1 < lines.length && TABLE_ROW.test(lines[i + 1])) rows.push(lines[++i]);
      block(tableNode(rows, cite));
      continue;
    }
    if (!line.trim()) {
      flush();
      list = null;
      continue;
    }
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      const node = el(heading[1].length <= 3 ? "h3" : "h4");
      node.append(...inlineNodes(heading[2], cite));
      block(node);
      continue;
    }
    const item = line.match(/^\s*([-*+]|\d+[.)])\s+(.*)$/);
    if (item) {
      flush();
      const ordered = /\d/.test(item[1]);
      if (!list || list.ordered !== ordered) {
        list = { ordered, node: el(ordered ? "ol" : "ul") };
        root.append(list.node);
      }
      list.last = el("li");
      list.last.append(...inlineNodes(item[2], cite));
      list.node.append(list.last);
      continue;
    }
    if (list && /^\s+\S/.test(line)) {
      list.last.append(document.createTextNode(" "), ...inlineNodes(line.trim(), cite));
      continue;
    }
    if (line.startsWith(">")) {
      const quote = el("blockquote");
      quote.append(...inlineNodes(line.replace(/^>\s?/, ""), cite));
      block(quote);
      continue;
    }
    list = null;
    paragraph.push(line.trim());
  }
  flush();
  return root;
}

/** POST and feed every server-sent event to `onEvent(name, data)`. */
async function streamEvents(path, body, onEvent, signal) {
  const response = await postJSON(path, body, signal); // aborting `signal` closes the stream
  if (!response.ok) throw new Error(await errorText(response));
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const buffer = { text: "" };
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer.text += decoder.decode(value, { stream: true });
    for (const [name, data] of sseEvents(buffer)) {
      if (name === "error") {
        const error = new Error(txServer(data.error));
        error.data = data; // e.g. the conversation the failed message was stored in
        throw error;
      }
      onEvent(name, data);
    }
  }
}

// --- shared state --------------------------------------------------------------------------

const app = {
  folders: [], // [{path, documents}]
  llmRunning: false,
  settings: Object.assign(
    { mode: "explain", precision: "precise", k: 8, scope: "", strategy: "direct", level: "intermediate" },
    store.get("biblio.settings", {}),
  ),
  thread: [], // the open conversation's messages, from the server (ADR-023)
  conversationId: store.get("biblio.conversation", null),
  title: "",
  frame: "", // the conversation's topic (ADR-038)
  busy: false,
  selection: null, // {turn, index}
};

// answer levels (ADR-032); chosen per message, the last one restored with a conversation
const LEVEL_LABELS = { beginner: tx("Principiante"), intermediate: tx("Intermedio"), advanced: tx("Avanzado") };

function setLevel(level) {
  if (!LEVEL_LABELS[level]) return;
  app.settings.level = level;
  for (const radio of document.querySelectorAll('input[name="level"]')) radio.checked = radio.value === level;
  saveSettings();
}

function syncLevelChoice() {
  $("level-choice").hidden = app.settings.mode !== "explain";
}

function saveSettings() {
  store.set("biblio.settings", app.settings);
}

/** The open conversation; only its id is kept in the browser, the history is on the server. */
function setConversation(id) {
  app.conversationId = id;
  store.set("biblio.conversation", id);
}

// --- router --------------------------------------------------------------------------------

const PAGES = ["conversar", "guias", "practicar", "mapas", "biblioteca", "leer"]; // leer: from Biblioteca
const pageHooks = {}; // page name → function run each time the page is shown (set by guias.js, …)

function currentPage() {
  const match = location.hash.match(/^#\/(\w+)/);
  return match && PAGES.includes(match[1]) ? match[1] : "conversar";
}

function route() {
  const page = currentPage();
  for (const name of PAGES) $(`page-${name}`).hidden = name !== page;
  const menuPage = page === "leer" ? "biblioteca" : page; // the reader belongs to Biblioteca (ADR-033)
  for (const link of document.querySelectorAll(".pages a")) {
    if (link.dataset.page === menuPage) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  $("new-thread").hidden = page !== "conversar";
  syncHistoryToggle();
  if (page === "conversar") {
    const prefill = location.hash.match(/[?&]q=([^&]*)/); // from Mapas' «Explicar»
    if (prefill) $("message").value = decodeURIComponent(prefill[1]);
    $("message").focus();
  }
  if (pageHooks[page]) pageHooks[page]();
}

// --- status and folders --------------------------------------------------------------------

async function loadStatus() {
  const pill = $("status");
  try {
    const status = await (await fetch("/api/status")).json();
    app.llmRunning = status.llm_running;
    const notes = status.documents.toLocaleString(i18nLocale());
    pill.textContent = status.llm_running ? tx("Lista · {0} notas", notes) : tx("Modelo apagado · {0} notas", notes);
    pill.classList.toggle("off", !status.llm_running);
    pill.title = status.llm_running
      ? tx("Modelo: {0} · búsqueda {1}", status.llm_model, status.mode)
      : tx("Sin modelo de lenguaje solo se puede ubicar. Inícialo con scripts/serve-llm.ps1.");
  } catch {
    pill.textContent = tx("Sin conexión con el servidor");
    pill.classList.add("off");
  }
}

/** «Actualizar»: incremental reindex of the library, with its progress in the status pill. */
async function reindexLibrary() {
  const pill = $("status");
  const button = $("reindex");
  if (button.disabled) return;
  button.disabled = true;
  button.classList.add("spinning");
  pill.classList.remove("off");
  pill.textContent = tx("Revisando tu biblioteca…");
  let device = "";
  try {
    await streamEvents("/api/reindex", {}, (name, data) => {
      if (name === "start") {
        device = data.device === "cpu" ? tx(" · en CPU") : "";
        pill.textContent = tx("Revisando {0} notas…", data.files.toLocaleString(i18nLocale()));
      } else if (name === "progress") {
        const chunks = data.chunks ? tx(" · {0} fragmentos nuevos", data.chunks) : "";
        pill.textContent = tx("Actualizando {0} de {1}{2}{3}", data.files.toLocaleString(i18nLocale()), data.total.toLocaleString(i18nLocale()), chunks, device);
      } else if (name === "done") {
        const changes = [
          data.added && `${data.added} ${data.added === 1 ? "nueva" : "nuevas"}`,
          data.updated && `${data.updated} ${data.updated === 1 ? "modificada" : "modificadas"}`,
          data.deleted && `${data.deleted} ${data.deleted === 1 ? "eliminada" : "eliminadas"}`,
        ].filter(Boolean);
        pill.textContent = changes.length ? `${changes.join(" · ")} · ${data.seconds} s` : tx("Tu biblioteca ya estaba al día");
        if (data.failed.length) pill.title = tx("No se pudieron leer: {0}", data.failed.join(", "));
      }
    });
    loadFolders();
    setTimeout(loadStatus, 6000); // leave the summary visible for a moment
  } catch (error) {
    pill.textContent = tx("No se pudo actualizar");
    pill.classList.add("off");
    pill.title = error.message;
    setTimeout(loadStatus, 8000);
  } finally {
    button.disabled = false;
    button.classList.remove("spinning");
  }
}

async function loadFolders() {
  try {
    app.folders = (await (await fetch("/api/folders")).json()).folders;
  } catch {
    app.folders = [];
  }
  renderScopeOptions();
}

function renderScopeOptions() {
  const select = $("set-scope");
  const top = app.folders.filter((f) => !f.path.includes("/"));
  const options = [el("option", null, tx("Toda la biblioteca"))];
  options[0].value = "";
  const scope = app.settings.scope;
  if (scope && !top.some((f) => f.path === scope)) top.unshift({ path: scope, documents: null });
  for (const folder of top) {
    const label = folder.documents === null ? folder.path : `${folder.path} (${folder.documents})`;
    const option = el("option", null, label);
    option.value = folder.path;
    options.push(option);
  }
  select.replaceChildren(...options);
  select.value = scope;
}

// --- slash commands ------------------------------------------------------------------------

const fold = (text) => text.normalize("NFD").replace(/\p{M}/gu, "").toLowerCase();

/** `/ubicar`, `/explicar` and `/en <carpeta>` at the start of a message. */
function parseCommands(raw) {
  let text = raw.trim();
  const result = { mode: null, scope: null, error: null };
  for (;;) {
    const lower = fold(text);
    if (lower.startsWith("/ubicar")) {
      result.mode = "find";
      text = text.slice(7).trim();
    } else if (lower.startsWith("/explicar")) {
      result.mode = "explain";
      text = text.slice(9).trim();
    } else if (lower.startsWith("/en ")) {
      const rest = text.slice(4).trim();
      const folded = fold(rest);
      const match = app.folders
        .filter((f) => {
          const name = fold(f.path);
          return folded === name || folded.startsWith(name + " ");
        })
        .sort((a, b) => b.path.length - a.path.length)[0];
      if (!match) {
        result.error = tx("No encuentro esa carpeta. Escribe su ruta desde la raíz, por ejemplo «/en Data Science».");
        return { ...result, text: "" };
      }
      result.scope = match.path;
      text = rest.slice(match.path.length).trim();
    } else {
      return { ...result, text };
    }
  }
}

// --- rendering the thread ------------------------------------------------------------------

function citeButton(turn, number, known) {
  const button = el("button", "cite", String(number));
  button.type = "button";
  if (!known.has(number)) {
    button.classList.add("invalid");
    button.disabled = true;
    button.title = tx("Cita a un fragmento que no existe");
    return button;
  }
  button.setAttribute("aria-label", tx("Ver fuente {0}", number));
  const selected = app.selection && app.selection.turn === turn.id;
  const index = turn.fragments.findIndex((f) => f.number === number);
  button.setAttribute("aria-pressed", String(Boolean(selected && app.selection.index === index)));
  button.addEventListener("click", () => selectSource(turn, index));
  return button;
}

function answerNode(turn) {
  const known = new Set((turn.fragments || []).map((f) => f.number));
  const node = markdownNode(turn.answer || "", (n) => citeButton(turn, Number(n), known), "md answer");
  // what the model adds of its own, marked by the prompt (ADR-038): a connection between areas the
  // notes do not write, or an example they do not have
  for (const p of node.querySelectorAll("p")) {
    if (/^\s*Relación propuesta/i.test(p.textContent)) p.classList.add("proposed-relation");
    if (/^\s*Ejemplo ilustrativo/i.test(p.textContent)) p.classList.add("proposed-relation", "illustrative");
  }
  // formulas in the answer, once it is complete (reader.js, KaTeX vendored: ADR-031)
  return turn.done && typeof renderMathIn === "function" ? renderMathIn(node) : node;
}

// «Apuntado como tema pendiente · Deshacer» under an answer or a map (ADR-029)
function gapNote(gap) {
  const box = el("p", "gap-note");
  if (gap.undone) {
    box.append(el("span", "muted", tx("No se apuntó como tema pendiente.")));
    return box;
  }
  const times = gap.count > 1 ? tx(" · preguntado {0} veces", gap.count) : "";
  box.append(el("span", null, tx("Apuntado como tema pendiente: «{0}»{1}. ", gap.topic, times)));
  const guides = el("a", null, tx("Ver en Guías"));
  guides.href = "#/guias";
  const undo = el("button", "link-button", tx("Deshacer"));
  undo.type = "button";
  undo.addEventListener("click", async () => {
    const response = await postJSON("/api/gaps/undo", { event_id: gap.event_id });
    if (!response.ok) return;
    gap.undone = true;
    box.replaceWith(gapNote(gap));
  });
  box.append(guides, el("span", "muted", " · "), undo);
  return box;
}

function flagsNode(done) {
  const flags = el("div", "flags");
  const add = (kind, text) => flags.append(el("span", `flag ${kind}`, text));
  if (done.abstained) add("warn", tx("Tus notas no contienen la respuesta."));
  else if (done.grounded) add("ok", tx("Cada cita apunta a un fragmento real."));
  else add("bad", tx("Respuesta sin citas válidas: no la des por buena sin revisar las fuentes."));
  if (done.invalid.length) add("bad", tx("Cita fragmentos que no existen: {0}", done.invalid.join(", ")));
  if (done.placeholders) add("bad", tx("{0} citas sin número.", done.placeholders));
  return flags;
}

function scopeLabel(scope) {
  return scope ? ` · en ${scope}` : "";
}

/** One search result: the section, its guide and lines, «Leer» and «Abrir». `compact` for a note's
 *  other sections under its best one (ADR-045). */
function resultRow(turn, hit, index, compact = false) {
  const li = el("li", compact ? "result result-sub" : "result");
  if (app.selection && app.selection.turn === turn.id && app.selection.index === index) {
    li.classList.add("selected");
  }
  const main = el("button", "result-main");
  main.type = "button";
  main.append(el("strong", null, hit.sections.at(-1) || hit.guide));
  main.append(el("span", null, compact ? tx("líneas {0}–{1}", hit.lines[0], hit.lines[1]) : tx("{0} · líneas {1}–{2}", hit.guide, hit.lines[0], hit.lines[1])));
  main.addEventListener("click", () => selectSource(turn, index));
  li.append(main);
  const read = readLink(hit.doc_path, hit.lines[0], null);
  read.setAttribute("aria-label", tx("Leer {0} aquí", hit.guide));
  li.append(read);
  if (hit.open_url && !compact) {
    const open = el("a", null, tx("Abrir"));
    open.href = hit.open_url;
    open.setAttribute("aria-label", tx("Abrir {0} en Obsidian", hit.guide));
    li.append(open);
  }
  return li;
}

/** [[hit, index into turn.hits], …] per note, in the server's order; null for flat results. */
function resultGroups(turn) {
  if (!turn.hits.length || turn.hits[0].group === undefined) return null;
  const groups = [];
  turn.hits.forEach((hit, index) => {
    if (!groups[hit.group]) groups[hit.group] = [];
    groups[hit.group].push([hit, index]);
  });
  return groups.filter(Boolean);
}

function resultsNode(turn) {
  const list = el("ul", "results");
  const groups = resultGroups(turn);
  if (!groups) {
    turn.hits.forEach((hit, index) => list.append(resultRow(turn, hit, index)));
    return list;
  }
  // one line per note, its next two sections beneath, the rest folded (ADR-045)
  for (const group of groups) {
    const [[lead, leadIndex], ...rest] = group;
    const note = resultRow(turn, lead, leadIndex);
    note.classList.add("result-note");
    const item = el("li", "result-group");
    const inner = el("ul", "result-subs");
    inner.append(note);
    for (const [hit, index] of rest.slice(0, 2)) inner.append(resultRow(turn, hit, index, true));
    item.append(inner);
    if (rest.length > 2) {
      const more = el("details", "result-more");
      more.append(el("summary", null, tx("{0} {1} en esta nota", rest.length - 2, rest.length - 2 === 1 ? tx("sección más") : tx("secciones más"))));
      const extra = el("ul", "result-subs");
      for (const [hit, index] of rest.slice(2)) extra.append(resultRow(turn, hit, index, true));
      more.append(extra);
      item.append(more);
    }
    list.append(item);
  }
  return list;
}

function followupsNode(turn) {
  const row = el("div", "followups");
  const where = el("button", "chip", tx("¿Dónde más aparece?"));
  where.type = "button";
  where.addEventListener("click", () => send(turn.question, { mode: "find", scope: turn.scope }));
  const practice = el("a", "chip", tx("Practicar este tema"));
  practice.href = `#/practicar?tema=${encodeURIComponent(turn.question)}`;
  const copy = el("button", "chip", tx("Copiar"));
  copy.type = "button";
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(turn.answer);
      copy.textContent = tx("Copiado");
    } catch {
      copy.textContent = tx("No se pudo copiar");
    }
  });
  row.append(where, practice, copy);
  return row;
}

/** «Entendí: …» and the searches, when the message was read with the conversation (ADR-038). */
function planNode(turn) {
  const plan = turn.plan;
  const box = el("details", "plan-note");
  const same = plan.question.trim().toLowerCase() === turn.question.trim().toLowerCase();
  box.append(el("summary", null, same ? tx("Cómo lo busqué") : tx("Entendí: «{0}»", plan.question)));
  const searches = [plan.question, ...plan.searches];
  box.append(el("p", "small muted", tx("Búsquedas: {0}{1}", searches.join(" · "), plan.relate ? tx(" · relacionando áreas") : "")));
  return box;
}

function offtopicNode(turn) {
  const box = el("div", "offtopic");
  const topic = turn.offtopic.frame ? tx(" distinto de «{0}»", turn.offtopic.frame) : "";
  box.append(el("p", null, tx("Esto parece otro tema{0}. Para no mezclarlos, conviene abrir una conversación nueva.", topic)));
  const row = el("div", "row");
  const fresh = el("button", "btn-primary", tx("Nueva conversación con esta pregunta"));
  fresh.type = "button";
  fresh.addEventListener("click", () => {
    app.thread = app.thread.filter((t) => t !== turn);
    startNewConversation();
    send(turn.question, { mode: turn.mode, scope: turn.scope });
  });
  const here = el("button", "btn", tx("Responder aquí"));
  here.type = "button";
  here.addEventListener("click", () => {
    app.thread = app.thread.filter((t) => t !== turn);
    renderThread();
    send(turn.question, { mode: turn.mode, scope: turn.scope, force: true });
  });
  row.append(fresh, here);
  box.append(row);
  return box;
}

function turnNode(turn) {
  const wrap = el("div");
  wrap.className = "turn";
  wrap.dataset.turn = turn.id;
  wrap.style.display = "contents";
  wrap.append(el("div", "bubble", turn.question));
  const reply = el("article", "reply");
  const meta = el("div", "reply-meta");
  const explain = turn.mode === "explain";
  const levelLabel = explain && turn.level ? ` · ${LEVEL_LABELS[turn.level] || turn.level}` : "";
  meta.append(el("span", `tag ${explain ? "explain" : ""}`, explain ? tx("Explicación{0}", levelLabel) : tx("Ubicación")));
  reply.append(meta);

  if (turn.status === "error") {
    reply.append(el("p", "error", turn.error));
  } else if (!explain) {
    if (turn.status === "pending") meta.append(document.createTextNode(tx("Buscando…")));
    else {
      const groups = resultGroups(turn);
      const found = groups ? tx("{0} {1} · {2} secciones", groups.length, groups.length === 1 ? tx("nota") : tx("notas"), turn.hits.length) : tx("{0} secciones", turn.hits.length);
      meta.append(document.createTextNode(`${found} · ${(turn.ms / 1000).toFixed(1)} s${scopeLabel(turn.scope)}`));
      reply.append(turn.hits.length ? resultsNode(turn) : el("p", "muted", tx("No encontré nada sobre esto.")));
    }
  } else if (turn.offtopic) {
    reply.append(offtopicNode(turn));
  } else {
    const n = (turn.fragments || []).length;
    if (!turn.fragments) meta.append(document.createTextNode(turn.planning ? tx("Entendiendo tu pregunta…") : tx("Buscando en tus notas…")));
    else if (!turn.done) meta.append(document.createTextNode(tx("Redactando con {0} fragmentos…", n)));
    else {
      const strategy = turn.strategy === "map-reduce" ? tx(" · por guía (experimental)") : "";
      meta.append(document.createTextNode(tx("Basada en {0} fragmentos · {1} s{2}{3}", n, turn.done.seconds, scopeLabel(turn.scope), strategy)));
    }
    if (turn.plan) reply.append(planNode(turn));
    if (turn.plan && turn.plan.relation === "nuevo") {
      reply.append(el("p", "small muted", tx("Respondida fuera del tema de la conversación; para seguir con este tema, abre una conversación nueva.")));
    }
    if (turn.answer) reply.append(answerNode(turn));
    if (turn.done) {
      reply.append(flagsNode(turn.done));
      if (turn.gap) reply.append(gapNote(turn.gap));
      reply.append(followupsNode(turn));
    }
  }
  wrap.append(reply);
  return wrap;
}

function renderThread() {
  const thread = $("thread");
  thread.querySelectorAll(".turn").forEach((node) => node.remove());
  $("welcome").hidden = app.thread.length > 0;
  for (const turn of app.thread) thread.append(turnNode(turn));
}

function rerender(turn) {
  const old = $("thread").querySelector(`.turn[data-turn="${turn.id}"]`);
  const fresh = turnNode(turn);
  if (old) old.replaceWith(fresh);
  else $("thread").append(fresh);
}

// --- the source panel ----------------------------------------------------------------------

function selectSource(turn, index) {
  const previous = app.selection && app.thread.find((t) => t.id === app.selection.turn);
  app.selection = { turn: turn.id, index };
  if (previous && previous !== turn) rerender(previous);
  rerender(turn);
  showSource();
}

function sourceItems(turn) {
  return turn.mode === "explain" ? turn.fragments || [] : turn.hits || [];
}

function showSource() {
  const turn = app.selection && app.thread.find((t) => t.id === app.selection.turn);
  const items = turn ? sourceItems(turn) : [];
  const item = items[app.selection ? app.selection.index : -1];
  $("source-empty").hidden = Boolean(item);
  $("source-view").hidden = !item;
  if (!item) return;
  const label = turn.mode === "explain" ? tx("Fuente") : tx("Resultado");
  const number = turn.mode === "explain" ? item.number : app.selection.index + 1;
  $("source-count").textContent = `${label} ${number} de ${items.length}`;
  $("source-section").textContent = item.sections.at(-1) || item.guide;
  const trail = [item.guide, ...item.sections.slice(0, -1)].join(" › ");
  $("source-guide").textContent = tx("{0} · líneas {1}–{2}", trail, item.lines[0], item.lines[1]);
  $("source-text").textContent = item.text;
  $("source-text").scrollTop = 0;
  $("source-open").hidden = !item.open_url;
  if (item.open_url) $("source-open").href = item.open_url;
  $("source-read").href = readerHref(item.doc_path, item.lines[0]); // ADR-031
  $("source-path").textContent = item.doc_path;
  $("source-prev").disabled = items.length < 2;
  $("source-next").disabled = items.length < 2;
}

function stepSource(delta) {
  const turn = app.selection && app.thread.find((t) => t.id === app.selection.turn);
  if (!turn) return;
  const count = sourceItems(turn).length;
  selectSource(turn, (app.selection.index + delta + count) % count);
}

// --- sending -------------------------------------------------------------------------------

async function runFind(turn) {
  const response = await postJSON("/api/search", {
    query: turn.question,
    k: Number(app.settings.k),
    precision: app.settings.precision,
    scope: turn.scope || null,
    group: true, // one line per note (ADR-045)
    conversation_id: app.conversationId,
  });
  if (!response.ok) throw new Error(await errorText(response));
  const data = await response.json();
  setConversation(data.conversation_id);
  turn.hits = data.hits;
  turn.ms = data.milliseconds;
  turn.retriever = data.retriever;
}

async function runExplain(turn) {
  if (!app.llmRunning) {
    await loadStatus();
    if (!app.llmRunning) {
      throw new Error(tx("El modelo de lenguaje no está encendido. Inícialo con scripts/serve-llm.ps1, o usa «Solo ubicar»."));
    }
  }
  turn.answer = "";
  const body = {
    question: turn.question,
    scope: turn.scope || null,
    strategy: turn.strategy,
    conversation_id: app.conversationId,
    level: turn.level,
    force: Boolean(turn.force),
  };
  turn.planning = true;
  try {
    await streamEvents("/api/ask", body, (name, data) => {
      if (name === "meta") {
        turn.fragments = data.fragments;
        turn.plan = data.plan || null;
      } else if (name === "token") turn.answer += data.text;
      else if (name === "done") {
        turn.done = data;
        setConversation(data.conversation_id);
        if (turn.plan && turn.plan.relation !== "nuevo") app.frame = turn.plan.frame;
        renderConversationBar();
      } else if (name === "gap") turn.gap = data;
      else if (name === "offtopic") turn.offtopic = data; // nothing answered yet: ask (ADR-038)
      rerender(turn);
    });
  } catch (error) {
    if (error.data && error.data.conversation_id) setConversation(error.data.conversation_id);
    throw error;
  }
}

async function send(raw, overrides = {}) {
  if (app.busy) return;
  const parsed = parseCommands(raw);
  if (parsed.error) {
    $("message").setCustomValidity(parsed.error);
    $("message").reportValidity();
    return;
  }
  if (!parsed.text) return;
  const turn = {
    id: `${Date.now()}`,
    question: parsed.text,
    mode: overrides.mode || parsed.mode || app.settings.mode,
    scope: overrides.scope !== undefined ? overrides.scope : parsed.scope ?? app.settings.scope,
    strategy: app.settings.strategy,
    level: app.settings.level,
    status: "pending",
    force: Boolean(overrides.force),
  };
  app.thread.push(turn);
  app.busy = true;
  $("send").disabled = true;
  $("welcome").hidden = true;
  rerender(turn);
  // the turn wrapper is `display: contents` and has no box of its own: scroll to its bubble
  $("thread").lastElementChild.firstElementChild.scrollIntoView({ behavior: "smooth", block: "start" });
  try {
    if (turn.mode === "find") await runFind(turn);
    else await runExplain(turn);
    turn.status = "done";
  } catch (error) {
    turn.status = "error";
    turn.error = error.message;
  } finally {
    app.busy = false;
    $("send").disabled = false;
    rerender(turn);
    loadHistory();
  }
}

// --- conversation history (ADR-023) -------------------------------------------------------

const MONTHS_SHORT = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"];

function formatWhen(iso) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const time = `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  if (I18N.lang === "en") return `${d.toLocaleDateString("en", { month: "short", day: "numeric" })}, ${time}`;
  return `${d.getDate()} ${MONTHS_SHORT[d.getMonth()]}, ${time}`;
}

/** A stored message as the page's turn object, so it renders like a live one. */
function turnFromRecord(record) {
  const turn = {
    id: `h${record.id}`,
    question: record.question,
    mode: record.mode,
    scope: record.scope,
    strategy: record.strategy,
    status: record.status,
    error: record.error,
  };
  if (record.mode === "explain") {
    turn.level = record.payload.level || null;
    turn.answer = record.answer;
    turn.fragments = record.payload.fragments || [];
    turn.done = record.payload.done || null;
    turn.plan = record.payload.plan || null;
  } else {
    turn.hits = record.payload.hits || [];
    turn.ms = record.payload.milliseconds || 0;
    turn.retriever = record.payload.retriever;
  }
  return turn;
}

function startNewConversation() {
  setConversation(null);
  app.title = "";
  app.frame = "";
  app.thread = [];
  app.selection = null;
  renderThread();
  showSource();
  renderConversationBar();
  markActiveConversation();
}

async function openConversation(id) {
  if (app.busy) return;
  let data = null;
  try {
    const response = await fetch(`/api/conversations/${id}`);
    if (response.ok) data = await response.json();
  } catch {
    /* the server is not answering: keep what is shown */
  }
  if (!data) {
    startNewConversation(); // deleted, or never stored
    return;
  }
  setConversation(data.id);
  app.title = data.title;
  app.frame = data.frame || "";
  app.thread = data.turns.map(turnFromRecord);
  const lastLevel = [...app.thread].reverse().find((t) => t.level);
  if (lastLevel) setLevel(lastLevel.level); // continue at the level the conversation was at
  app.selection = null;
  renderThread();
  showSource();
  renderConversationBar();
  markActiveConversation();
  const last = $("thread").lastElementChild;
  if (last && last.firstElementChild) last.firstElementChild.scrollIntoView({ block: "start" });
}

async function loadHistory() {
  const query = $("history-search").value.trim();
  let conversations = [];
  try {
    const url = `/api/conversations${query ? `?q=${encodeURIComponent(query)}` : ""}`;
    conversations = (await (await fetch(url)).json()).conversations;
  } catch {
    /* keep the list empty */
  }
  const items = conversations.map((c) => {
    const li = el("li");
    const button = el("button", "history-item");
    button.type = "button";
    button.dataset.id = String(c.id);
    button.append(el("strong", null, c.title));
    const count = `${c.turns} ${c.turns === 1 ? tx("mensaje") : tx("mensajes")}`;
    button.append(el("span", null, `${formatWhen(c.updated)} · ${count}`));
    if (c.snippet) button.append(el("span", "history-snippet", c.snippet));
    button.addEventListener("click", () => openConversation(c.id));
    li.append(button);
    if (c.id === app.conversationId) app.title = c.title;
    return li;
  });
  $("history-list").replaceChildren(...items);
  $("history-empty").hidden = items.length > 0;
  $("history-empty").textContent = query
    ? tx("Ninguna conversación contiene esas palabras.")
    : tx("Tus conversaciones aparecerán aquí.");
  markActiveConversation();
  renderConversationBar();
}

function markActiveConversation() {
  for (const item of document.querySelectorAll(".history-item")) {
    if (Number(item.dataset.id) === app.conversationId) item.setAttribute("aria-current", "true");
    else item.removeAttribute("aria-current");
  }
}

function renderConversationBar() {
  $("conversation-bar").hidden = !app.conversationId;
  $("conversation-title").textContent = app.title;
  $("conversation-frame").hidden = !app.frame || !$("conversation-frame-form").hidden;
  $("conversation-frame-text").textContent = app.frame;
}

/** «Tema: … · Editar» (ADR-038): the topic the planner reads every message against. */
function bindConversationFrame() {
  const form = $("conversation-frame-form");
  const close = () => {
    form.hidden = true;
    renderConversationBar();
  };
  $("conversation-frame-edit").addEventListener("click", () => {
    $("conversation-frame-input").value = app.frame;
    form.hidden = false;
    $("conversation-frame").hidden = true;
    $("conversation-frame-input").focus();
  });
  $("conversation-frame-cancel").addEventListener("click", close);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const frame = $("conversation-frame-input").value.trim();
    if (!frame || !app.conversationId) return;
    try {
      await postConversation("/api/conversations/frame", { id: app.conversationId, frame });
      app.frame = frame;
    } catch (error) {
      conversationStatus(error.message);
    }
    close();
  });
}

function conversationStatus(text) {
  $("conversation-status").textContent = text;
}

async function postConversation(path, body) {
  const response = await postJSON(path, body);
  if (!response.ok) throw new Error(await errorText(response));
  return response.json();
}

/** Pages with a history panel that the «Historial» button shows or hides (ADR-023); Mapas has
 *  its own drawer, «Mis mapas». Each page remembers its own choice. */
const HISTORY_PANELS = {
  conversar: { panel: "history", search: "history-search", key: "biblio.historyOpen" },
};

function setHistoryOpen(page, open) {
  const p = HISTORY_PANELS[page];
  $(p.panel).hidden = !open;
  store.set(p.key, open);
}

function syncHistoryToggle() {
  const p = HISTORY_PANELS[currentPage()];
  const button = $("history-toggle");
  button.hidden = !p;
  if (!p) return;
  const open = !$(p.panel).hidden;
  button.setAttribute("aria-controls", p.panel);
  button.setAttribute("aria-expanded", String(open));
  button.classList.toggle("active", open);
}

function bindHistory() {
  for (const [page, p] of Object.entries(HISTORY_PANELS)) setHistoryOpen(page, store.get(p.key, true));
  $("history-toggle").addEventListener("click", () => {
    const page = currentPage();
    const p = HISTORY_PANELS[page];
    if (!p) return;
    const open = $(p.panel).hidden;
    setHistoryOpen(page, open);
    syncHistoryToggle();
    if (open) $(p.search).focus();
  });
  let timer = null;
  $("history-search").addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(loadHistory, 250);
  });
  const showRename = (on) => {
    $("conversation-rename").hidden = !on;
    $("conversation-title").hidden = on;
    $("conversation-actions").hidden = on;
  };
  $("conversation-rename-start").addEventListener("click", () => {
    $("conversation-title-input").value = app.title;
    showRename(true);
    $("conversation-title-input").focus();
  });
  $("conversation-rename-cancel").addEventListener("click", () => showRename(false));
  $("conversation-rename").addEventListener("submit", async (event) => {
    event.preventDefault();
    const title = $("conversation-title-input").value.trim();
    if (!title) return;
    try {
      await postConversation("/api/conversations/rename", { id: app.conversationId, title });
      app.title = title;
      showRename(false);
      loadHistory();
    } catch (error) {
      conversationStatus(error.message);
    }
  });
  $("conversation-export").addEventListener("click", async () => {
    try {
      const data = await postConversation("/api/conversations/export", { id: app.conversationId });
      const link = el("a");
      link.href = URL.createObjectURL(new Blob([data.markdown], { type: "text/markdown" }));
      link.download = data.filename;
      link.click();
      setTimeout(() => URL.revokeObjectURL(link.href), 1000);
      conversationStatus(tx("Exportada a {0}", data.path));
    } catch (error) {
      conversationStatus(error.message);
    }
  });
  $("conversation-delete").addEventListener("click", async () => {
    if (!confirm(tx("¿Borrar «{0}»? No se puede deshacer.", app.title))) return;
    try {
      await postConversation("/api/conversations/delete", { id: app.conversationId });
      startNewConversation();
      conversationStatus("");
      loadHistory();
    } catch (error) {
      conversationStatus(error.message);
    }
  });
}

// --- wiring --------------------------------------------------------------------------------

/** Light, dark or "system" (follow the OS); theme.js applies the saved choice before drawing. */
function bindTheme() {
  const saved = store.get("biblio.theme", "system");
  for (const radio of document.querySelectorAll('input[name="theme"]')) {
    radio.checked = radio.value === saved;
    radio.addEventListener("change", () => {
      if (radio.value === "system") delete document.documentElement.dataset.theme;
      else document.documentElement.dataset.theme = radio.value;
      store.set("biblio.theme", radio.value);
    });
  }
  const contrast = $("code-contrast");
  contrast.checked = store.get("biblio.codeContrast", false) === true;
  contrast.addEventListener("change", () => {
    if (contrast.checked) document.documentElement.dataset.codeContrast = "high";
    else delete document.documentElement.dataset.codeContrast;
    store.set("biblio.codeContrast", contrast.checked);
  });
}

function bindSettings() {
  const bind = (id, key) => {
    const input = $(id);
    input.value = String(app.settings[key]);
    input.addEventListener("change", () => {
      app.settings[key] = input.value;
      saveSettings();
    });
  };
  bind("set-precision", "precision");
  bind("set-k", "k");
  bind("set-scope", "scope");
  bind("set-strategy", "strategy");
  for (const radio of document.querySelectorAll('input[name="mode"]')) {
    radio.checked = radio.value === app.settings.mode;
    radio.addEventListener("change", () => {
      app.settings.mode = radio.value;
      saveSettings();
      syncLevelChoice();
    });
  }
  for (const radio of document.querySelectorAll('input[name="level"]')) {
    radio.addEventListener("change", () => setLevel(radio.value));
  }
  setLevel(app.settings.level);
  syncLevelChoice();
  $("settings-toggle").addEventListener("click", () => {
    const open = $("settings").hidden;
    $("settings").hidden = !open;
    $("settings-toggle").setAttribute("aria-expanded", String(open));
  });
}

function bindComposer() {
  const message = $("message");
  message.addEventListener("input", () => message.setCustomValidity(""));
  message.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      $("composer").requestSubmit();
    }
  });
  $("composer").addEventListener("submit", (event) => {
    event.preventDefault();
    const text = message.value;
    if (!text.trim() || app.busy) return;
    message.value = "";
    send(text);
  });
  $("new-thread").addEventListener("click", () => {
    if (app.busy) return;
    startNewConversation();
    message.focus();
  });
  $("source-prev").addEventListener("click", () => stepSource(-1));
  $("source-next").addEventListener("click", () => stepSource(1));
}

// Sticky panels sit below the top bar, whose height changes when it wraps.
function watchTopbarHeight() {
  const bar = document.querySelector(".topbar");
  const set = () => document.documentElement.style.setProperty("--topbar-h", `${bar.offsetHeight}px`);
  new ResizeObserver(set).observe(bar);
  set();
}

document.addEventListener("DOMContentLoaded", () => {
  watchTopbarHeight();
  bindTheme();
  $("reindex").addEventListener("click", reindexLibrary);
  bindSettings();
  bindComposer();
  bindHistory();
  renderThread();
  showSource();
  bindConversationFrame();
  if (app.conversationId) openConversation(app.conversationId);
  loadHistory();
  window.addEventListener("hashchange", route);
  route();
  loadStatus();
  loadFolders();
});
