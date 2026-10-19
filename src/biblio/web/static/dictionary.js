// Biblioteca — «Diccionario» in Practicar (ADR-034), laid out as a glossary (the user's proposal A):
// an index of terms by list and letter on the left, the selected entry on the right, flashcards one
// click away, and each term marked with its last flashcard result.
// Uses app.js ($, el, store, postJSON, errorText, streamEvents, showError, loadStatus, app),
// practicar.js (showScreen, practice, fetchToday, showTodayBadge), reader.js (readerHref,
// renderMath, renderMathIn) and library.js (libraryFold). Library and model text is only inserted
// as text.
"use strict";

const DICT_KINDS = [["concept", "Conceptos", "Concepto"], ["function", "Funciones", "Función"], ["formula", "Fórmulas", "Fórmula"]];
const DICT_MARKS = { correct: "La sabías", partial: "A medias", wrong: "No la sabías" };
const dict = { list: [], view: null, kind: "concept", filter: "", selected: null, building: false, cards: null };

function dictSetStatus(text) {
  $("dict-status").hidden = !text;
  $("dict-status").textContent = text || "";
}

function dictShowNew(open) {
  $("dict-new").hidden = !open;
  $("dict-new-toggle").setAttribute("aria-expanded", String(open));
  if (open) $("dict-value").focus();
}

// --- the saved dictionaries -------------------------------------------------------------------

async function dictLoadList() {
  try {
    dict.list = (await (await fetch("/api/dictionary/history")).json()).dictionaries;
  } catch {
    dict.list = [];
  }
  const select = $("dict-select");
  const first = el("option", null, dict.list.length ? "Elige un diccionario…" : "Aún no tienes diccionarios");
  first.value = "";
  select.replaceChildren(
    first,
    ...dict.list.map((d) => {
      const total = d.counts.concept + d.counts.function + d.counts.formula;
      const option = el("option", null, `${d.title} · ${total}`);
      option.value = String(d.id);
      return option;
    }),
  );
  select.value = dict.view && dict.view.id ? String(dict.view.id) : "";
  if (!dict.list.length && !dict.view) dictShowNew(true);
}

async function dictOpen(id) {
  showError("dict-error", "");
  dictSetStatus("");
  const response = await fetch(`/api/dictionary/${id}`);
  if (!response.ok) {
    dictLoadList();
    return;
  }
  dict.view = await response.json();
  dict.cards = null;
  dict.filter = "";
  $("dict-filter").value = "";
  const kinds = DICT_KINDS.map(([k]) => k).filter((k) => dict.view.entries.some((e) => e.kind === k));
  dict.kind = kinds[0] || "concept";
  dict.selected = null;
  dictShowNew(false);
  dictRender();
  $("dict-select").value = String(id);
}

async function dictBuild(kind, value) {
  if (dict.building) return;
  showError("dict-error", "");
  await loadStatus();
  if (!app.llmRunning) {
    showError("dict-error", "El modelo de lenguaje no está encendido. Inícialo con scripts/serve-llm.ps1.");
    return;
  }
  dict.building = true;
  dict.cards = null;
  $("dict-create").disabled = true;
  dictSetStatus("Buscando en tus notas…");
  const view = { id: null, title: value, kind, entries: [], fragments: [], marks: {} };
  dict.view = view;
  dict.kind = "concept";
  dict.selected = null;
  dict.filter = "";
  $("dict-filter").value = "";
  dictShowNew(false);
  try {
    await streamEvents("/api/dictionary", { kind, value }, (name, data) => {
      if (name === "start") {
        view.title = data.title;
        view.fragments = data.fragments;
        view.partial = data.available > data.fragments.length ? data.available : 0;
        const part = view.partial ? ` (las primeras de ${data.available})` : "";
        dictSetStatus(`Leyendo ${data.fragments.length} secciones de tus notas${part}…`);
      } else if (name === "entries") {
        view.entries = data.entries;
        dictSetStatus(`Sección ${data.processed} de ${view.fragments.length} · ${data.entries.length} entradas`);
      } else if (name === "done") {
        view.id = data.id;
        const part = view.partial ? ` (la guía tiene ${view.partial}: sube dictionary.guide_max_fragments para leerla entera)` : "";
        const skipped = data.skipped ? ` · ${data.skipped} secciones no se pudieron leer` : "";
        dictSetStatus(`${data.count} entradas de ${view.fragments.length} secciones${part}${skipped}`);
      }
      if (dict.view === view) dictRender();
    });
    await dictLoadList();
  } catch (error) {
    dictSetStatus("");
    showError("dict-error", error.message);
    if (dict.view === view && !view.entries.length) dict.view = null;
  } finally {
    dict.building = false;
    $("dict-create").disabled = false;
    dictRender();
  }
}

// --- the glossary ---------------------------------------------------------------------------

function dictVisible() {
  // [index in view.entries, entry] of the current list, filtered and in alphabetical order
  const q = libraryFold(dict.filter.trim());
  return dict.view.entries
    .map((e, i) => [i, e])
    .filter(([, e]) => e.kind === dict.kind && (!q || libraryFold(`${e.term} ${e.text} ${e.detail}`).includes(q)))
    .sort((a, b) => a[1].term.localeCompare(b[1].term, "es", { sensitivity: "base" }));
}

function dictLetter(term) {
  const first = libraryFold(term).replace(/^[^a-z0-9]+/, "").charAt(0).toUpperCase();
  return /[A-Z]/.test(first) ? first : "#";
}

function dictMark(view, index) {
  return view.marks ? view.marks[String(index)] : undefined;
}

function dictRender() {
  const view = dict.view;
  const has = Boolean(view);
  $("dict-actions").hidden = !has || !view.id || !view.entries.length || dict.building;
  $("dict-empty").hidden = has || !dict.list.length;
  $("dict-body").hidden = !has || Boolean(dict.cards);
  $("dict-cards").hidden = !dict.cards;
  if (has && view.id) {
    $("dict-md").href = `/api/dictionary/${view.id}/export?format=md`;
    $("dict-anki").href = `/api/dictionary/${view.id}/export?format=anki`;
  }
  if (dict.cards) {
    $("dict-cards").replaceChildren(dictCardNode());
    return;
  }
  if (!has) return;
  $("dict-heading").textContent = view.title;
  for (const tab of document.querySelectorAll("#dict-tabs button")) {
    const kind = tab.dataset.kind;
    const count = view.entries.filter((e) => e.kind === kind).length;
    const label = DICT_KINDS.find(([k]) => k === kind)[1];
    tab.textContent = `${label} ${count}`;
    tab.setAttribute("aria-selected", String(kind === dict.kind));
    tab.disabled = !count && kind !== dict.kind;
  }
  const visible = dictVisible();
  if (dict.selected === null || !visible.some(([i]) => i === dict.selected)) {
    dict.selected = visible.length ? visible[0][0] : null;
  }
  dictRenderIndex(visible);
  dictRenderDetail(visible);
}

function dictRenderIndex(visible) {
  const list = $("dict-terms");
  const letters = [];
  const nodes = [];
  let current = null;
  for (const [index, entry] of visible) {
    const letter = dictLetter(entry.term);
    if (letter !== current) {
      current = letter;
      letters.push(letter);
      const head = el("li", "dict-letter", letter);
      head.id = `dict-letter-${letter}`;
      nodes.push(head);
    }
    const li = el("li");
    const button = el("button", "dict-term-button");
    button.type = "button";
    const mark = dictMark(dict.view, index);
    const dot = el("span", `dict-dot ${mark ? `is-${mark}` : ""}`);
    dot.setAttribute("aria-hidden", "true");
    const name = el(entry.kind === "function" ? "code" : "span", "dict-term-name", entry.kind === "function" ? `${entry.term}()` : entry.term);
    button.append(dot, name);
    button.setAttribute("aria-label", `${entry.term}${mark ? `, ${DICT_MARKS[mark].toLowerCase()}` : ""}`);
    if (index === dict.selected) button.setAttribute("aria-current", "true");
    button.addEventListener("click", () => {
      dict.selected = index;
      dictRender();
      if (window.innerWidth < 760) $("dict-detail").scrollIntoView({ block: "start" });
    });
    li.append(button);
    nodes.push(li);
  }
  if (!visible.length) nodes.push(el("li", "muted dict-none", dict.building ? "Leyendo tus notas…" : "Nada coincide."));
  list.replaceChildren(...nodes);
  const strip = $("dict-letters");
  strip.replaceChildren(
    ...letters.map((letter) => {
      const link = el("button", "dict-letter-link", letter);
      link.type = "button";
      link.setAttribute("aria-label", `Ir a la ${letter}`);
      link.addEventListener("click", () => {
        const target = document.getElementById(`dict-letter-${letter}`);
        if (target) $("dict-terms").scrollTop = target.offsetTop; // the list is its offset parent
      });
      return link;
    }),
  );
  strip.hidden = letters.length < 2;
  const active = list.querySelector('[aria-current="true"]');
  if (active) {
    const box = $("dict-terms");
    if (active.offsetTop < box.scrollTop || active.offsetTop > box.scrollTop + box.clientHeight - 40) {
      box.scrollTop = active.offsetTop - 40;
    }
  }
}

function dictSourceOf(view, number) {
  return view.fragments.find((x) => x.number === number) || null;
}

function dictRenderDetail(visible) {
  const box = $("dict-detail");
  const view = dict.view;
  const position = visible.findIndex(([i]) => i === dict.selected);
  if (position < 0) {
    box.replaceChildren(el("p", "muted", dict.building ? "Las entradas aparecen aquí mientras leo tus notas." : "Elige un término de la lista."));
    return;
  }
  const entry = view.entries[dict.selected];
  const label = DICT_KINDS.find(([k]) => k === entry.kind)[2];
  const nodes = [el("span", "dict-kind", label)];
  if (entry.kind === "function") {
    nodes.push(el("h2", "dict-detail-term", `${entry.term}()`), el("pre", "dict-code", entry.detail));
  } else if (entry.kind === "formula") {
    nodes.push(el("h2", "dict-detail-term", entry.term), renderMath(el("div", "math-block dict-formula"), entry.detail, true));
  } else {
    nodes.push(el("h2", "dict-detail-term", entry.term));
  }
  nodes.push(renderMathIn(el("p", "dict-definition", entry.text)));
  const first = dictSourceOf(view, entry.sources[0]);
  if (entry.kind === "concept") {
    const figure = el("figure", "dict-quote");
    figure.append(renderMathIn(el("blockquote", null, `«${entry.detail}»`)));
    if (first) figure.append(el("figcaption", null, `${first.citation} · líneas ${first.lines[0]}–${first.lines[1]}`));
    nodes.push(figure);
  } else if (first) {
    nodes.push(el("p", "small muted", `${first.citation} · líneas ${first.lines[0]}–${first.lines[1]}`));
  }
  if (entry.sources.length > 1) {
    const more = el("p", "small muted dict-more-sources", "También en: ");
    for (const n of entry.sources.slice(1, 6)) {
      const f = dictSourceOf(view, n);
      if (!f) continue;
      const link = el("a", "cite", f.sections.at(-1) || f.guide);
      link.href = readerHref(f.doc_path, f.lines[0]);
      link.title = f.citation;
      more.append(link, document.createTextNode(" "));
    }
    nodes.push(more);
  }
  const mark = dictMark(view, dict.selected);
  nodes.push(el("p", `dict-mark-line ${mark ? `is-${mark}` : ""}`, mark ? `En fichas: ${DICT_MARKS[mark].toLowerCase()}` : "Aún no la repasaste como ficha"));
  const actions = el("div", "row");
  if (first) {
    const read = el("a", "btn-soft", "Leer en la nota");
    read.href = readerHref(first.doc_path, first.lines[0]);
    actions.append(read);
  }
  const explain = el("a", "btn", "Explícamelo tú");
  explain.href = `#/practicar?explicar=${encodeURIComponent(entry.term)}`;
  actions.append(explain);
  nodes.push(actions);
  const nav = el("div", "dict-prev-next");
  const step = (delta) => {
    const next = visible[position + delta];
    if (!next) return null;
    const button = el("button", "link-button", delta < 0 ? `← ${next[1].term}` : `${next[1].term} →`);
    button.type = "button";
    button.addEventListener("click", () => {
      dict.selected = next[0];
      dictRender();
    });
    return button;
  };
  nav.append(step(-1) || el("span"), step(1) || el("span"));
  nodes.push(nav);
  box.replaceChildren(...nodes);
}

// --- flashcards ------------------------------------------------------------------------------

function dictStartCards() {
  const view = dict.view;
  const rank = { wrong: 0, partial: 1, undefined: 2, correct: 3 }; // the ones you missed first
  const order = view.entries.map((_, i) => i);
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  order.sort((a, b) => rank[dictMark(view, a)] - rank[dictMark(view, b)]);
  dict.cards = { order, at: 0, flipped: false, quizId: null, results: { correct: 0, partial: 0, wrong: 0 } };
  dictRender();
  $("dict-cards").scrollIntoView({ block: "start" });
}

function dictStopCards() {
  dict.cards = null;
  dictRender();
}

function dictCardNode() {
  const c = dict.cards;
  const view = dict.view;
  const wrap = el("div", "dict-cards-inner");
  const top = el("div", "row-between");
  top.append(el("span", "small muted", c.at < c.order.length ? `Ficha ${c.at + 1} de ${c.order.length} · primero las que no sabías` : "Terminado"));
  const exit = el("button", "btn", "Volver al diccionario");
  exit.type = "button";
  exit.addEventListener("click", dictStopCards);
  top.append(exit);
  wrap.append(top);
  if (c.at >= c.order.length) {
    const r = c.results;
    const done = el("div", "practice-card dict-card");
    done.append(el("h2", null, "Fichas repasadas"), el("p", null, `La sabías: ${r.correct} · A medias: ${r.partial} · No la sabías: ${r.wrong}`));
    done.append(el("p", "small muted", "Cuentan para tus repasos, y cada término queda marcado en el índice."));
    wrap.append(done);
    return wrap;
  }
  const index = c.order[c.at];
  const entry = view.entries[index];
  const card = el("div", "practice-card dict-card");
  card.append(el("span", "eyebrow", DICT_KINDS.find(([k]) => k === entry.kind)[2]));
  if (entry.kind === "concept") card.append(el("h2", "dict-front", entry.term));
  else if (entry.kind === "function") card.append(el("pre", "dict-code dict-front", entry.detail));
  else card.append(el("h2", "dict-front", entry.term), renderMath(el("div", "math-block"), entry.detail, true));
  if (!c.flipped) {
    const flip = el("button", "btn-primary btn-big", "Girar la ficha");
    flip.type = "button";
    flip.addEventListener("click", () => {
      c.flipped = true;
      dictRender();
    });
    card.append(el("p", "small muted", "Intenta recordarlo antes de girarla."), flip);
  } else {
    card.append(renderMathIn(el("p", "dict-back", entry.text)));
    if (entry.kind === "concept") card.append(renderMathIn(el("p", "dict-quote-small", `«${entry.detail}»`)));
    const row = el("div", "row");
    for (const [result, label] of [["correct", "La sabía"], ["partial", "A medias"], ["wrong", "No la sabía"]]) {
      const b = el("button", result === "correct" ? "btn-primary" : "btn", label);
      b.type = "button";
      b.addEventListener("click", async () => {
        for (const x of row.querySelectorAll("button")) x.disabled = true;
        const response = await postJSON("/api/dictionary/review", { id: view.id, index, result, quiz_id: c.quizId });
        if (response.ok) {
          c.quizId = (await response.json()).quiz_id;
          view.marks = { ...(view.marks || {}), [String(index)]: result };
        }
        c.results[result] += 1;
        c.at += 1;
        c.flipped = false;
        dictRender();
        if (c.at >= c.order.length) fetchToday().then(showTodayBadge);
      });
      row.append(b);
    }
    card.append(row);
  }
  wrap.append(card);
  return wrap;
}

// --- wiring ----------------------------------------------------------------------------------

function dictBind() {
  $("tile-dictionary").addEventListener("click", () => {
    showError("hub-error", "");
    showScreen("dictionary");
    dictLoadList();
    dictRender();
  });
  $("dict-select").addEventListener("change", () => {
    const id = Number($("dict-select").value);
    if (id) dictOpen(id);
  });
  $("dict-new-toggle").addEventListener("click", () => dictShowNew($("dict-new").hidden));
  for (const tab of document.querySelectorAll("#dict-tabs button")) {
    tab.addEventListener("click", () => {
      dict.kind = tab.dataset.kind;
      dict.selected = null;
      dictRender();
    });
  }
  let timer = null;
  $("dict-filter").addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      dict.filter = $("dict-filter").value;
      dictRender();
    }, 120);
  });
  $("dict-cards-start").addEventListener("click", dictStartCards);
  $("dict-delete").addEventListener("click", async () => {
    const view = dict.view;
    if (!view || !confirm(`¿Borrar el diccionario «${view.title}»? Lo que practicaste con él se mantiene.`)) return;
    await postJSON("/api/dictionary/delete", { id: view.id });
    dict.view = null;
    dictSetStatus("");
    dictRender();
    dictLoadList();
  });
  for (const radio of document.querySelectorAll('input[name="dict-kind"]')) {
    radio.addEventListener("change", () => {
      const guide = radio.value === "guide" && radio.checked;
      $("dict-value").placeholder = guide ? "Escribe parte del título de la guía…" : "p. ej. regresión logística";
      if (guide) $("dict-value").setAttribute("list", "guide-options");
      else $("dict-value").removeAttribute("list");
    });
  }
  $("dict-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const kind = document.querySelector('input[name="dict-kind"]:checked').value;
    const raw = $("dict-value").value.trim();
    if (!raw) {
      showError("dict-error", kind === "guide" ? "Elige una guía." : "Escribe un tema.");
      return;
    }
    const value = kind === "guide" ? practice.guides.get(raw) || raw : raw;
    dictBuild(kind, value);
  });
}

let dictBound = false;
const dictPracticarHook = pageHooks.practicar;

pageHooks.practicar = () => {
  dictPracticarHook();
  if (!dictBound) {
    dictBound = true;
    dictBind();
  }
};
