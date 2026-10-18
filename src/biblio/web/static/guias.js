// Biblioteca — page Guías (ADR-019): the two-step `biblio draft` flow in the browser.
// Uses the helpers of app.js ($, el, store, postJSON, errorText, streamEvents, loadStatus, app).
// Library and model text is only ever inserted with textContent, never as HTML.
"use strict";

const guide = {
  slug: null,
  topic: "",
  sections: [], // [{title, notes}] as edited in the page
  siblings: [],
  markdown: null,
  sources: null, // [{number, citation, text, open_url, lines, …}] after writing in this session
  step: "topic", // "topic" | "outline" | "draft"
  live: null, // while writing: [{title, status, text, body, flags}]
  busy: false,
};

const MONTHS = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"];

function shortDate(iso) {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "" : `${d.getDate()} ${MONTHS[d.getMonth()]}`;
}

async function requireLLM() {
  await loadStatus();
  if (!app.llmRunning) {
    throw new Error("El modelo de lenguaje no está encendido. Inícialo con scripts/serve-llm.ps1.");
  }
}

// --- the drafts list -----------------------------------------------------------------------

async function loadDraftList() {
  let drafts = [];
  try {
    drafts = (await (await fetch("/api/drafts")).json()).drafts;
  } catch {
    /* the list stays empty */
  }
  const items = drafts.map((d) => {
    const li = el("li");
    const button = el("button", "draft-item");
    button.type = "button";
    if (d.slug === guide.slug && guide.step !== "topic") button.setAttribute("aria-current", "true");
    button.append(el("strong", null, d.topic));
    const detail = d.has_draft ? `${d.sections} secciones · ${d.sources} fuentes` : "Índice sin escribir";
    button.append(el("span", null, `${shortDate(d.updated)} · ${detail}`));
    button.addEventListener("click", () => openGuide(d.slug));
    li.append(button);
    return li;
  });
  if (!items.length) items.push(el("li", "muted small", "Aún no tienes borradores."));
  $("draft-list").replaceChildren(...items);
}

async function openGuide(slug) {
  if (guide.busy) return;
  const response = await fetch(`/api/drafts/${encodeURIComponent(slug)}`);
  if (!response.ok) {
    loadDraftList();
    return;
  }
  const data = await response.json();
  guide.slug = data.slug;
  guide.topic = data.topic;
  guide.markdown = data.markdown;
  guide.sources = null;
  guide.live = null;
  guide.siblings = data.outline ? data.outline.siblings : [];
  guide.sections = data.outline
    ? data.outline.sections.map((s) => ({ ...s }))
    : sectionsFromMarkdown(data.markdown || "").map((title) => ({ title, notes: "" }));
  guide.step = data.markdown ? "draft" : "outline";
  store.set("biblio.guide", guide.slug);
  render();
}

function newGuide() {
  if (guide.busy) return;
  Object.assign(guide, { slug: null, topic: "", sections: [], siblings: [], markdown: null, sources: null, live: null, step: "topic" });
  store.set("biblio.guide", null);
  render();
  $("guide-topic").focus();
}

// --- step 1 and "propose another": the outline ----------------------------------------------

async function propose(topic, liveId, statusId, errorId) {
  if (guide.busy) return;
  showError(errorId, "");
  guide.busy = true;
  render();
  const live = $(liveId);
  live.textContent = "";
  live.hidden = false;
  $(statusId).textContent = "Leyendo tus notas y proponiendo un índice…";
  $(statusId).hidden = false;
  try {
    await requireLLM();
    await streamEvents("/api/drafts/outline", { topic }, (name, data) => {
      if (name === "token") live.textContent += data.text;
      else if (name === "outline") {
        guide.slug = data.slug;
        guide.topic = data.topic;
        guide.sections = data.sections;
        guide.siblings = data.siblings;
        guide.markdown = null;
        guide.sources = null;
        guide.live = null;
        guide.step = "outline";
        store.set("biblio.guide", guide.slug);
      }
    });
    if (guide.step === "outline" && !guide.sections.length) {
      throw new Error("El modelo no propuso ninguna sección. Prueba otra vez o cambia el tema.");
    }
  } catch (error) {
    showError(errorId, error.message);
  } finally {
    guide.busy = false;
    live.hidden = true;
    $(statusId).hidden = true;
    render();
    loadDraftList();
  }
}

// --- step 2: editing the outline -----------------------------------------------------------

function iconButton(label, icon, onClick, danger) {
  const button = el("button", `btn-icon-plain${danger ? " danger" : ""}`);
  button.type = "button";
  button.setAttribute("aria-label", label);
  // icons are cloned from <template id="icons"> in index.html
  button.append($("icons").content.querySelector(`[data-icon="${icon}"]`).cloneNode(true));
  button.addEventListener("click", onClick);
  return button;
}

function renderOutline() {
  const siblings = guide.siblings.map((p) => p.split("/").pop());
  $("outline-siblings").textContent = siblings.length
    ? `Este índice se propuso a partir de tus notas, imitando la estructura de ${siblings.join(" y ")}. Cambia los títulos, el orden o quita secciones antes de escribir.`
    : "Cambia los títulos, el orden o quita secciones antes de escribir.";
  const rows = guide.sections.map((section, i) => {
    const li = el("li", "outline-row");
    li.append(el("span", "outline-n", String(i + 1)));
    const fields = el("div", "outline-fields");
    const title = el("input");
    title.type = "text";
    title.value = section.title;
    title.maxLength = 200;
    title.setAttribute("aria-label", `Título de la sección ${i + 1}`);
    title.addEventListener("input", () => (section.title = title.value));
    fields.append(title);
    const notes = el("textarea");
    notes.rows = 2;
    notes.maxLength = 1000;
    notes.value = section.notes || "";
    notes.placeholder = "Notas para esta sección (opcional): qué debe cubrir, qué evitar…";
    notes.setAttribute("aria-label", `Notas de la sección ${i + 1}`);
    notes.hidden = !section.notes && !section.showNotes;
    notes.addEventListener("input", () => (section.notes = notes.value));
    fields.append(notes);
    li.append(fields);
    const actions = el("div", "outline-actions");
    const notesButton = el("button", "btn-ghost", notes.hidden ? "Notas" : "Ocultar notas");
    notesButton.type = "button";
    notesButton.addEventListener("click", () => {
      section.showNotes = notes.hidden;
      renderOutline();
    });
    actions.append(
      notesButton,
      iconButton(`Subir sección ${i + 1}`, "up", () => moveSection(i, -1)),
      iconButton(`Bajar sección ${i + 1}`, "down", () => moveSection(i, 1)),
      iconButton(`Quitar sección ${i + 1}`, "remove", () => {
        guide.sections.splice(i, 1);
        renderOutline();
      }, true),
    );
    li.append(actions);
    return li;
  });
  $("outline-list").replaceChildren(...rows);
  const n = guide.sections.filter((s) => s.title.trim()).length;
  $("write-draft").textContent = `Escribir borrador (${n} ${n === 1 ? "sección" : "secciones"})`;
  $("write-draft").disabled = guide.busy || n === 0;
}

function moveSection(i, delta) {
  const j = i + delta;
  if (j < 0 || j >= guide.sections.length) return;
  [guide.sections[i], guide.sections[j]] = [guide.sections[j], guide.sections[i]];
  renderOutline();
  $("outline-list").children[j].querySelector(`[aria-label^="${delta < 0 ? "Subir" : "Bajar"}"]`)?.focus();
}

// --- step 3: writing and showing the draft -------------------------------------------------

async function writeDraft() {
  if (guide.busy) return;
  const sections = guide.sections
    .map((s) => ({ title: s.title.trim(), notes: (s.notes || "").trim() }))
    .filter((s) => s.title);
  if (!sections.length) return;
  showError("outline-error", "");
  showError("draft-error", "");
  guide.busy = true;
  guide.step = "draft";
  guide.markdown = null;
  guide.sources = null;
  guide.live = sections.map((s) => ({ title: s.title, status: "pending", text: "", body: "", flags: [] }));
  render();
  try {
    await requireLLM();
    await streamEvents("/api/drafts/write", { topic: guide.topic, sections }, (name, data) => {
      if (name === "start") guide.slug = data.slug;
      else if (name === "section") guide.live[data.index].status = "writing";
      else if (name === "token") guide.live[data.index].text += data.text;
      else if (name === "section_done") {
        const s = guide.live[data.index];
        s.status = "done";
        s.body = data.body;
        if (data.missing_evidence) s.flags.push(["warn", "Sin evidencia en tu biblioteca"]);
        if (data.uncited) s.flags.push(["bad", "Sin citas: revísala"]);
        if (data.invalid_citations.length) s.flags.push(["bad", "Citas inexistentes, marcadas [?]"]);
      } else if (name === "done") {
        guide.markdown = data.markdown;
        guide.sources = data.sources;
        guide.live = null;
        $("draft-path").textContent = `Guardado en ${data.path}`;
      }
      renderDraft();
    });
  } catch (error) {
    showError("draft-error", `${error.message} Lo escrito hasta ahora no se guardó.`);
    if (!guide.markdown) guide.step = "outline";
  } finally {
    guide.busy = false;
    render();
    loadDraftList();
  }
}

function sourceLink(n) {
  if (n === "?") return el("span", "cite invalid", "?");
  const link = el("a", "cite", n);
  link.href = `#fuente-${n}`;
  link.setAttribute("aria-label", `Fuente ${n}`);
  link.addEventListener("click", (event) => {
    event.preventDefault(); // keep the page's hash route
    const target = document.getElementById(`fuente-${n}`);
    if (!target) return;
    const details = target.querySelector("details");
    if (details) details.open = true;
    target.scrollIntoView({ behavior: "smooth", block: "center" });
    target.classList.add("flash");
    setTimeout(() => target.classList.remove("flash"), 1500);
  });
  return link;
}

function textWithCitations(text) {
  return markdownNode(text, sourceLink, "md draft-text");
}

/** A written draft as {sections: [{title, body, flags}], sources: [{number, line}]}. */
function parseDraft(markdown) {
  const sections = [];
  const sources = [];
  let current = null;
  let inSources = false;
  for (const line of markdown.split("\n")) {
    if (line.startsWith("## ")) {
      inSources = line.trim() === "## Fuentes";
      current = inSources ? null : { title: line.slice(3).trim(), lines: [], flags: [] };
      if (current) sections.push(current);
    } else if (inSources) {
      const m = line.match(/^(\d+)\. (.*)$/);
      if (m) sources.push({ number: Number(m[1]), line: m[2] });
    } else if (current) {
      if (line.startsWith("> ⚠")) current.flags.push(["bad", line.slice(4).trim()]);
      else if (line.startsWith("> ")) current.flags.push(["warn", line.slice(2).trim()]);
      else current.lines.push(line);
    }
  }
  for (const s of sections) s.body = s.lines.join("\n").trim();
  return { sections, sources };
}

function sectionsFromMarkdown(markdown) {
  return parseDraft(markdown).sections.map((s) => s.title);
}

function sourceNode(source, detail) {
  const li = el("li", "source-item");
  li.id = `fuente-${source.number}`;
  li.append(el("span", "source-n", `${source.number}.`));
  const body = el("div");
  body.append(el("span", null, detail ? `${detail.citation} (líneas ${detail.lines[0]}-${detail.lines[1]})` : source.line));
  if (detail) {
    const more = el("details");
    more.append(el("summary", null, "Ver el fragmento"));
    more.append(el("div", "source-text", detail.text));
    if (detail.open_url) {
      const open = el("a", "btn-soft", "Abrir en Obsidian");
      open.href = detail.open_url;
      more.append(open);
    }
    body.append(more);
  }
  li.append(body);
  return li;
}

function renderDraft() {
  const progress = [];
  const doc = [];
  if (guide.live) {
    const done = guide.live.filter((s) => s.status === "done").length;
    const writing = guide.live.findIndex((s) => s.status === "writing");
    $("draft-status").textContent = writing >= 0
      ? `Escribiendo sección ${writing + 1} de ${guide.live.length}…`
      : `${done} de ${guide.live.length} secciones`;
    for (const s of guide.live) {
      const li = el("li", `progress-${s.status}`, s.title);
      if (s.status === "writing") li.setAttribute("aria-current", "step");
      progress.push(li);
      if (s.status === "pending") continue;
      doc.push(el("h2", null, s.title));
      // while streaming, numbers are the section's own: shown, not linked
      if (s.status === "writing") doc.push(markdownNode(s.text, (n) => el("span", "cite", n), "md draft-text streaming"));
      else {
        for (const [kind, text] of s.flags) doc.push(el("p", `flag ${kind}`, text));
        if (s.body) doc.push(textWithCitations(s.body));
      }
    }
  } else if (guide.markdown) {
    const parsed = parseDraft(guide.markdown);
    const details = new Map((guide.sources || []).map((s) => [s.number, s]));
    $("draft-status").textContent = `${parsed.sections.length} secciones · ${parsed.sources.length} fuentes`;
    for (const s of parsed.sections) {
      const li = el("li", "progress-done", s.title);
      if (s.flags.length) li.classList.add("has-flags");
      progress.push(li);
      doc.push(el("h2", null, s.title));
      for (const [kind, text] of s.flags) doc.push(el("p", `flag ${kind}`, text));
      if (s.body) doc.push(textWithCitations(s.body));
    }
    if (parsed.sources.length) {
      doc.push(el("h2", null, "Fuentes"));
      const list = el("ol", "source-list");
      for (const source of parsed.sources) list.append(sourceNode(source, details.get(source.number)));
      doc.push(list);
    }
    if (!$("draft-path").textContent) $("draft-path").textContent = `drafts/${guide.slug}.md`;
  }
  $("draft-progress").replaceChildren(...progress);
  $("draft-body").replaceChildren(...doc);
  const ready = Boolean(guide.markdown) && !guide.busy;
  $("copy-md").disabled = !ready;
  const download = $("download-md");
  download.classList.toggle("disabled", !ready);
  download.setAttribute("aria-disabled", String(!ready));
  if (ready) {
    const old = download.dataset.url;
    if (old) URL.revokeObjectURL(old);
    const url = URL.createObjectURL(new Blob([guide.markdown], { type: "text/markdown" }));
    download.href = url;
    download.dataset.url = url;
    download.download = `${guide.slug}.md`;
  } else {
    download.removeAttribute("href");
  }
}

// --- page state ----------------------------------------------------------------------------

function render() {
  const topicStep = guide.step === "topic";
  $("guide-topic-form").hidden = !topicStep;
  $("guide-view").hidden = topicStep;
  $("propose").disabled = guide.busy;
  if (topicStep) return;
  $("guide-title").textContent = guide.topic;
  $("outline-step").hidden = guide.step !== "outline";
  $("draft-step").hidden = guide.step !== "draft";
  for (const [id, step] of [["step-outline", "outline"], ["step-draft", "draft"]]) {
    const button = $(id);
    const active = guide.step === step;
    button.classList.toggle("active", active);
    if (active) button.setAttribute("aria-current", "step");
    else button.removeAttribute("aria-current");
  }
  $("step-draft").disabled = guide.busy || (!guide.markdown && !guide.live);
  $("step-outline").disabled = guide.busy;
  for (const id of ["add-section", "repropose"]) $(id).disabled = guide.busy;
  if (guide.step === "outline") renderOutline();
  else renderDraft();
}

function bindGuias() {
  $("new-guide").addEventListener("click", newGuide);
  $("guide-topic-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const topic = $("guide-topic").value.trim();
    if (topic) propose(topic, "outline-live", "propose-status", "topic-error");
  });
  $("repropose").addEventListener("click", () => {
    if (confirm("Se reemplazará el índice actual, con tus cambios. ¿Continuar?")) {
      propose(guide.topic, "repropose-live", "outline-status", "outline-error");
    }
  });
  $("add-section").addEventListener("click", () => {
    guide.sections.push({ title: "", notes: "" });
    renderOutline();
    $("outline-list").lastElementChild.querySelector("input").focus();
  });
  $("write-draft").addEventListener("click", writeDraft);
  $("step-outline").addEventListener("click", () => {
    guide.step = "outline";
    render();
  });
  $("step-draft").addEventListener("click", () => {
    guide.step = "draft";
    render();
  });
  $("copy-md").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(guide.markdown);
      $("copy-md").textContent = "Copiado";
    } catch {
      $("copy-md").textContent = "No se pudo copiar";
    }
    setTimeout(() => ($("copy-md").textContent = "Copiar Markdown"), 2000);
  });
}

let guiasStarted = false;

pageHooks.guias = () => {
  const linked = location.hash.match(/^#\/guias\/([a-z0-9-]+)$/); // `#/guias/<slug>` opens one
  if (!guiasStarted) {
    guiasStarted = true;
    bindGuias();
    const last = linked ? linked[1] : store.get("biblio.guide", null);
    if (last) openGuide(last);
    else render();
  } else if (linked && linked[1] !== guide.slug) {
    openGuide(linked[1]);
  }
  loadDraftList();
};
