// Biblioteca — choosing where to practice (ADR-063): the whole library, a guide, a book, a chapter or a
// folder. Shared by the quiz form, «Explícamelo tú» and the dictionary (practicar.js, feynman.js,
// dictionary.js). Lists with a filter instead of a search by text: with several books on the same
// subject, a title in a list with its area, chapters and read marks tells them apart.
// Uses the helpers of app.js ($, el). Library text is only ever inserted with textContent.
"use strict";

const SCOPE_LABELS = {
  library: tx("Toda la biblioteca"),
  guide: tx("Una guía"),
  book: tx("Un libro"),
  chapter: tx("Un capítulo"),
  folder: tx("Una carpeta"),
};
const SCOPE_SHOWN = 40; // rows drawn at once; typing narrows the rest

const scopeData = { guides: null, books: null, folders: null, chapters: new Map() };

function scopeFold(text) {
  return text.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
}

/** Every word of the query is somewhere in the text, give or take accents and case. */
function scopeMatches(query, text) {
  const words = scopeFold(query).split(/\s+/).filter(Boolean);
  const folded = scopeFold(text);
  return words.every((w) => folded.includes(w));
}

async function scopeFetch(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(String(response.status));
  return response.json();
}

async function scopeItems(kind, book) {
  if (kind === "guide") {
    scopeData.guides ??= (await scopeFetch("/api/practice/guides")).guides;
    return scopeData.guides.map((g) => ({
      title: g.title,
      meta: [g.folder || g.path],
      search: `${g.title} ${g.folder}`,
      choose: { kind: "guide", value: g.path, label: g.title },
    }));
  }
  if (kind === "folder") {
    scopeData.folders ??= (await scopeFetch("/api/folders")).folders;
    return scopeData.folders.map((f) => ({
      title: f.path,
      meta: [tx("{0} notas", f.documents)],
      search: f.path,
      choose: { kind: "folder", value: f.path, label: f.path },
    }));
  }
  if (kind === "chapter" && book) {
    if (!scopeData.chapters.has(book.id)) {
      scopeData.chapters.set(book.id, (await scopeFetch(`/api/books/one?id=${encodeURIComponent(book.id)}`)).chapters);
    }
    return scopeData.chapters.get(book.id).map((c, i) => ({
      title: c.title,
      number: i + 1,
      meta: c.read ? [tx("leído")] : [],
      search: c.title,
      choose: { kind: "chapter", value: c.path, label: `${book.title} › ${c.title}` },
    }));
  }
  scopeData.books ??= await scopeFetch("/api/books");
  return scopeData.books.books.map((b) => ({
    title: b.title,
    area: b.area,
    meta: [
      b.area,
      tx("{0} capítulos", b.chapters),
      ...(b.read ? [tx("{0} leídos", b.read)] : []),
      ...(b.english ? ["EN"] : []),
    ],
    search: `${b.title} ${b.area}`,
    book: { id: b.id, title: b.title },
    choose: { kind: "book", value: b.id, label: b.title },
  }));
}

const SCOPE_PROMPTS = {
  guide: tx("Elige una guía de la lista."),
  book: tx("Elige un libro de la lista."),
  chapter: tx("Elige un libro y, después, uno de sus capítulos."),
  folder: tx("Elige una carpeta de la lista."),
};

/** Draws the picker in `host`; `kinds` are the sources it offers, `onChange` hears every choice.
 *  Returns {get(), set(source), reset()}: get() gives {kind, value, label} or throws a Spanish
 *  message when nothing is chosen. */
function scopePickerCreate(host, kinds, onChange = () => {}) {
  const state = { kind: kinds[0], chosen: null, book: null, query: "", area: "" };
  const tabs = el("div", "scope-tabs");
  tabs.setAttribute("role", "radiogroup");
  const chosenBox = el("p", "scope-chosen");
  const body = el("div", "scope-body");
  host.replaceChildren(tabs, chosenBox, body);

  function current() {
    return state.kind === "library"
      ? { kind: "library", value: "", label: SCOPE_LABELS.library }
      : state.chosen && state.chosen.kind === state.kind
        ? state.chosen
        : null;
  }

  function drawTabs() {
    tabs.replaceChildren(
      ...kinds.map((kind) => {
        const tab = el("button", "scope-tab", SCOPE_LABELS[kind]);
        tab.type = "button";
        tab.setAttribute("role", "radio");
        tab.setAttribute("aria-checked", String(kind === state.kind));
        tab.addEventListener("click", () => {
          if (kind === state.kind) return;
          state.kind = kind;
          state.query = "";
          state.area = "";
          state.book = null;
          drawAll();
          onChange(current());
        });
        return tab;
      }),
    );
  }

  function drawChosen() {
    const now = current();
    chosenBox.replaceChildren();
    chosenBox.hidden = !now;
    if (!now) return;
    chosenBox.append(el("span", "muted", tx("Elegido:")), el("strong", null, now.label));
    if (now.kind !== "library") {
      const clear = el("button", "btn-ghost scope-clear", tx("Quitar"));
      clear.type = "button";
      clear.addEventListener("click", () => {
        state.chosen = null;
        state.book = null;
        drawAll();
        onChange(current());
      });
      chosenBox.append(clear);
    }
  }

  async function drawList(list, hint) {
    list.replaceChildren(el("p", "muted", tx("Cargando…")));
    let items;
    try {
      items = await scopeItems(state.kind, state.book);
    } catch {
      list.replaceChildren(el("p", "error", tx("No se pudo leer la lista.")));
      return;
    }
    const found = items.filter(
      (it) => (!state.area || it.area === state.area) && (!state.query || scopeMatches(state.query, it.search)),
    );
    if (!found.length) {
      list.replaceChildren(el("p", "muted", tx("Nada coincide con eso.")));
      hint.textContent = "";
      return;
    }
    hint.textContent = found.length > SCOPE_SHOWN ? tx("Mostrando {0} de {1}: escribe para acotar.", SCOPE_SHOWN, found.length) : tx("{0} en total.", found.length);
    list.replaceChildren(
      ...found.slice(0, SCOPE_SHOWN).map((it) => {
        const row = el("button", "scope-item");
        row.type = "button";
        const picked = state.chosen && it.choose && state.chosen.value === it.choose.value && state.chosen.kind === it.choose.kind;
        row.setAttribute("aria-pressed", String(Boolean(picked)));
        if (it.number) row.append(el("span", "scope-item-n", String(it.number)));
        const text = el("span", "scope-item-text");
        text.append(el("span", "scope-item-title", it.title));
        if (it.meta.length) text.append(el("span", "scope-item-meta", it.meta.join(" · ")));
        row.append(text);
        row.addEventListener("click", () => {
          if (state.kind === "chapter" && !state.book) {
            state.book = it.book; // the book first; its chapters come next
            state.query = "";
            drawAll();
            return;
          }
          state.chosen = it.choose;
          drawAll();
          onChange(current());
        });
        return row;
      }),
    );
  }

  function drawBody() {
    body.replaceChildren();
    if (state.kind === "library") {
      body.append(el("p", "muted", tx("Se usan todas tus notas y libros.")));
      return;
    }
    if (state.kind === "chapter") {
      if (state.book) {
        const crumb = el("p", "scope-crumb");
        crumb.append(el("span", "muted", tx("Libro:")), el("strong", null, state.book.title));
        const change = el("button", "btn-ghost scope-clear", tx("Cambiar libro"));
        change.type = "button";
        change.addEventListener("click", () => {
          state.book = null;
          state.chosen = null;
          state.query = "";
          drawAll();
          onChange(current());
        });
        crumb.append(change);
        body.append(crumb);
      } else {
        body.append(el("p", "muted", tx("Primero el libro; después, el capítulo.")));
      }
    }
    const filter = el("input", "scope-filter");
    filter.type = "search";
    filter.autocomplete = "off";
    filter.maxLength = 120;
    filter.value = state.query;
    filter.placeholder = state.kind === "folder" ? tx("Filtrar carpetas…") : tx("Filtrar por título…");
    filter.setAttribute("aria-label", tx("Filtrar la lista"));
    const row = el("div", "scope-filter-row");
    row.append(filter);
    body.append(row);
    const hint = el("p", "small muted scope-hint");
    const list = el("div", "scope-list");
    const choosingBook = state.kind === "book" || (state.kind === "chapter" && !state.book);
    if (choosingBook) {
      // the books of one area: with the same subject in several, the area narrows them
      scopeItems("book").then(() => {
        const area = el("select", "scope-area");
        area.setAttribute("aria-label", tx("Área"));
        const option = (name, label) => {
          const o = el("option", null, label);
          o.value = name;
          return o;
        };
        area.append(option("", tx("Todas las áreas")), ...scopeData.books.areas.map((a) => option(a.name, `${a.name} · ${a.count}`)));
        area.value = state.area;
        area.addEventListener("change", () => {
          state.area = area.value;
          drawList(list, hint);
        });
        row.append(area);
      }).catch(() => {});
    }
    body.append(hint, list);
    filter.addEventListener("input", () => {
      state.query = filter.value;
      drawList(list, hint);
    });
    drawList(list, hint);
  }

  function drawAll() {
    drawTabs();
    drawChosen();
    drawBody();
  }

  drawAll();
  return {
    get() {
      const now = current();
      if (!now) throw new Error(SCOPE_PROMPTS[state.kind]);
      return { ...now };
    },
    set(source) {
      if (!kinds.includes(source.kind)) return;
      state.kind = source.kind;
      state.chosen = source;
      state.book = null;
      state.query = "";
      drawAll();
      onChange(current());
    },
    reset() {
      state.kind = kinds[0];
      state.chosen = null;
      state.book = null;
      state.query = "";
      state.area = "";
      drawAll();
      onChange(current());
    },
  };
}
