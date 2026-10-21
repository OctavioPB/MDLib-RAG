// Biblioteca — «Gap Spotter» in Guías (ADR-047): a topic's subtopics the notes leave unexplained,
// and 5–10 guides to write. Uses the helpers of app.js ($, el, store, streamEvents, postJSON,
// errorText, showError, formatWhen), reader.js (readLink) and plan.js (showGuiasScreen).
// Library and model text is only ever inserted with textContent.
"use strict";

const spotter = { run: null, busy: false, controller: null };

const SPOTTER_SOURCES = {
  unexplained: tx("mencionado sin explicar"),
  syllabus: tx("propuesto por el modelo"),
};

const SPOTTER_VERDICTS = { missing: tx("falta"), partial: tx("a medias"), covered: tx("cubierto") };

function spotterStatus(step) {
  if (step.stage === "unexplained") return tx("Buscando lo que tus notas mencionan sin explicar… ({0} de {1})", step.done, step.total);
  if (step.stage === "syllabus") return tx("Pensando el temario completo del tema…");
  if (step.stage === "judge") return tx("Comprobando en tus notas: {0} de {1} subtemas…", step.done, step.total);
  return tx("Agrupando los huecos en guías…");
}

async function spotterLoadHistory(selected) {
  let runs = [];
  try {
    runs = (await (await fetch("/api/spotter/history")).json()).runs;
  } catch {
    // the screen works without it
  }
  const select = $("spotter-history");
  select.replaceChildren(el("option", null, runs.length ? tx("Elige una…") : tx("Aún no hay búsquedas")));
  select.firstChild.value = "";
  for (const r of runs) {
    const option = el("option", null, tx("{0} · {1} huecos · {2}", r.topic, r.gaps, formatWhen(r.created)));
    option.value = String(r.id);
    select.append(option);
  }
  select.value = selected ? String(selected) : "";
  $("spotter-delete").hidden = !selected;
}

async function spotterOpen(id) {
  showError("spotter-error", "");
  const response = await fetch(`/api/spotter/${id}`);
  if (!response.ok) {
    spotterLoadHistory(null);
    return;
  }
  spotter.run = await response.json();
  $("spotter-topic").value = spotter.run.topic;
  $("spotter-status").textContent = "";
  $("spotter-delete").hidden = false;
  spotterRender();
}

async function spotterStart(event) {
  event.preventDefault();
  const topic = $("spotter-topic").value.trim();
  if (!topic || spotter.busy) return;
  showError("spotter-error", "");
  await loadStatus();
  if (!app.llmRunning) {
    showError("spotter-error", tx("El modelo de lenguaje no está encendido. Inícialo con scripts/serve-llm.ps1."));
    return;
  }
  spotter.busy = true;
  spotter.run = { topic, coverage: [], guides: [], fragments: [] };
  spotter.controller = new AbortController();
  $("spotter-go").disabled = true;
  $("spotter-status").textContent = tx("Leyendo lo que tus notas dicen del tema…");
  spotterRender();
  try {
    await streamEvents(
      "/api/spotter",
      { topic },
      (name, data) => {
        if (name === "start") {
          spotter.run.fragments = data.fragments;
          $("spotter-status").textContent = tx("Leí {0} secciones de tus notas sobre el tema.", data.fragments.length);
        } else if (name === "step") {
          spotter.run.coverage = data.coverage;
          $("spotter-status").textContent = spotterStatus(data);
          spotterRender();
        } else if (name === "done") {
          spotter.run = { ...spotter.run, ...data, topic };
          $("spotter-status").textContent = "";
          spotterRender();
          spotterLoadHistory(data.id);
        }
      },
      spotter.controller.signal,
    );
  } catch (error) {
    if (error.name !== "AbortError") showError("spotter-error", error.message);
    $("spotter-status").textContent = "";
  } finally {
    spotter.busy = false;
    $("spotter-go").disabled = false;
  }
}

function spotterSource(source) {
  return el("span", `spotter-source ${source}`, SPOTTER_SOURCES[source] || source);
}

function spotterItem(c) {
  const li = el("li", "spotter-item");
  li.append(el("strong", null, c.name), spotterSource(c.source));
  if (c.about && c.source === "syllabus") li.append(el("span", "spotter-why", c.about));
  if (c.mention) {
    li.append(el("span", "small muted", tx("Se menciona en {0}:", c.mention.citation)));
    li.append(el("blockquote", null, `«${c.mention.quote}»`));
    li.append(readLink(c.mention.doc_path, c.mention.lines[0], "link-button"));
  }
  if (c.evidence) {
    li.append(el("span", "small muted", tx("En {0}:", c.evidence.citation)));
    li.append(el("blockquote", null, `«${c.evidence.quote}»`));
    li.append(readLink(c.evidence.doc_path, c.evidence.lines[0], "link-button"));
  }
  return li;
}

function spotterRender() {
  const run = spotter.run;
  $("spotter-result").hidden = !run || (!run.coverage.length && !run.guides.length);
  if (!run) return;
  const by = (verdict) => run.coverage.filter((c) => c.verdict === verdict);
  const missing = by("missing");
  const partial = by("partial");
  const covered = by("covered");
  $("spotter-summary").textContent = tx("Sobre «{0}»: {1} subtemas faltan, {2} están a medias y {3} están cubiertos en tus notas.", run.topic, missing.length, partial.length, covered.length);
  const box = $("spotter-coverage");
  const parts = [];
  for (const [title, items] of [[tx("Faltan"), missing], [tx("A medias"), partial]]) {
    if (!items.length) continue;
    const list = el("ul", "spotter-list");
    list.append(...items.map(spotterItem));
    parts.push(el("h3", null, `${title} (${items.length})`), list);
  }
  if (covered.length) {
    const details = el("details");
    details.append(el("summary", null, tx("Cubiertos en tus notas ({0})", covered.length)));
    const list = el("ul", "spotter-list");
    list.append(...covered.map(spotterItem));
    details.append(list);
    parts.push(details);
  }
  box.replaceChildren(...parts);
  $("spotter-guides").replaceChildren(...run.guides.map((g, i) => spotterGuide(g, i)));
  $("spotter-guides-title").hidden = !run.guides.length;
}

function spotterGuide(guide, index) {
  const li = el("li", "spotter-guide");
  li.append(el("h3", null, guide.title));
  if (guide.why) li.append(el("p", "small", guide.why));
  li.append(el("strong", "small", tx("Subtemas que cubre")));
  const subs = el("ul");
  for (const g of guide.gaps) {
    const item = el("li", null, `${g.name} `);
    item.append(el("span", "small muted", `(${SPOTTER_VERDICTS[g.verdict]} · ${SPOTTER_SOURCES[g.source]})`));
    subs.append(item);
  }
  li.append(subs);
  if (guide.questions.length) {
    li.append(el("strong", "small", tx("Preguntas que debe responder")));
    const qs = el("ul");
    qs.append(...guide.questions.map((q) => el("li", null, q)));
    li.append(qs);
  }
  if (guide.notes.length) {
    li.append(el("strong", "small", tx("Se conecta con tus notas")));
    const notes = el("ul");
    for (const n of guide.notes) {
      const item = el("li");
      item.append(readLink(n.doc_path, n.lines[0], "link-button", n.citation));
      notes.append(item);
    }
    li.append(notes);
  }
  const row = el("div", "row");
  const copy = el("button", "btn", tx("Copiar ficha"));
  copy.type = "button";
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(spotterMarkdown(guide));
      copy.textContent = tx("Copiada");
    } catch {
      copy.textContent = tx("No se pudo copiar");
    }
  });
  row.append(copy);
  if (spotter.run.id) {
    const note = el("button", "btn", tx("Apuntar en Temas pendientes"));
    note.type = "button";
    note.addEventListener("click", async () => {
      note.disabled = true;
      try {
        const response = await postJSON("/api/spotter/pending", { id: spotter.run.id, guide: index });
        if (!response.ok) throw new Error(await errorText(response));
        note.textContent = tx("Apuntada en Temas pendientes");
      } catch (error) {
        note.disabled = false;
        showError("spotter-error", error.message);
      }
    });
    row.append(note);
  }
  li.append(row);
  return li;
}

/** The card as Markdown, to write the guide elsewhere (the notes have no evidence for it). */
function spotterMarkdown(guide) {
  const lines = [
    `# ${guide.title}`,
    "",
    `> ${tx("Sugerida por Gap Spotter sobre «{0}». Los subtemas «propuesto por el modelo» no salen de tus notas.", spotter.run.topic)}`,
    "",
  ];
  if (guide.why) lines.push(`**${tx("Por qué")}:** ${guide.why}`, "");
  lines.push(`## ${tx("Subtemas que cubre")}`, "");
  for (const g of guide.gaps) {
    lines.push(`- ${g.name}${g.about && g.source === "syllabus" ? ` — ${g.about}` : ""} _(${SPOTTER_VERDICTS[g.verdict]}; ${SPOTTER_SOURCES[g.source]})_`);
  }
  if (guide.questions.length) {
    lines.push("", `## ${tx("Preguntas que debe responder")}`, "", ...guide.questions.map((q) => `- ${q}`));
  }
  if (guide.notes.length) {
    lines.push("", `## ${tx("Se conecta con tus notas")}`, "");
    for (const n of guide.notes) {
      const section = n.sections.length ? `#${n.sections.at(-1)}` : "";
      lines.push(`- [[${n.doc_path.replace(/\.md$/, "")}${section}|${n.citation}]]`);
    }
  }
  return `${lines.join("\n")}\n`;
}

function bindSpotter() {
  $("tile-spotter").addEventListener("click", () => {
    showGuiasScreen("spotter");
    spotterLoadHistory(spotter.run && spotter.run.id);
    $("spotter-topic").focus();
  });
  $("spotter-form").addEventListener("submit", spotterStart);
  $("spotter-history").addEventListener("change", (event) => {
    if (event.target.value) spotterOpen(Number(event.target.value));
  });
  $("spotter-delete").addEventListener("click", async () => {
    const run = spotter.run;
    if (!run || !run.id || !confirm(tx("¿Borrar la búsqueda de «{0}»?", run.topic))) return;
    const response = await postJSON("/api/spotter/delete", { id: run.id });
    if (!response.ok) return;
    spotter.run = null;
    spotterRender();
    spotterLoadHistory(null);
  });
}

let spotterBound = false;
const spotterGuiasHook = pageHooks.guias;

pageHooks.guias = () => {
  if (!spotterBound) {
    spotterBound = true;
    bindSpotter();
  }
  spotterGuiasHook();
};
