// Biblioteca — the «Biblioteca» page: the user's books as a shelf (ADR-053), their own notes by
// folder (ADR-033), and «Seguir leyendo» with where each note was left; everything opens in the
// reader (reader.js, ADR-031).
// Uses the helpers of app.js ($, el, store) and readerHref/recentNotes of reader.js.
"use strict";

const library = {
  notes: [],
  tree: null,
  loaded: false,
  signature: "",
  tab: "books",
  shelf: null, // {books, areas} from /api/books
  area: null, // the area chosen on the shelf, null for all
};
const LIBRARY_CONTINUE_MAX = 3;
const LIBRARY_BOOK_COLORS = ["#1A4D80", "#5B3A8A", "#226B5B", "#7A4B1E", "#1F5F8B", "#8A5A12", "#2E4A7A", "#6B2F3A", "#3D5A1E"];

function libraryFold(text) {
  return text.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
}

function libraryTree(notes) {
  const root = { name: "", folders: new Map(), notes: [], count: 0 };
  for (const note of notes) {
    const parts = note.path.split("/");
    let node = root;
    node.count += 1;
    for (const part of parts.slice(0, -1)) {
      if (!node.folders.has(part)) node.folders.set(part, { name: part, folders: new Map(), notes: [], count: 0 });
      node = node.folders.get(part);
      node.count += 1;
    }
    node.notes.push(note);
  }
  return root;
}

function libraryNoteItem(note, showPath) {
  const li = el("li", "library-note");
  const link = el("a", null, note.title);
  link.href = readerHref(note.path);
  link.title = note.title;
  li.append(link);
  if (note.read) {
    const mark = el("span", "library-read", tx("✓ leída"));
    mark.title = tx("Marcada como leída");
    li.append(mark);
  }
  if (showPath) {
    const path = note.path.split("/").slice(0, -1).join(" › ") || tx("Raíz");
    const crumb = el("span", "library-path", path);
    crumb.title = path;
    li.append(crumb);
  }
  return li;
}

function libraryFolder(folder) {
  const details = el("details", "library-folder");
  const summary = el("summary");
  summary.append(el("span", "library-folder-name", folder.name), el("span", "library-count", String(folder.count)));
  details.append(summary);
  // the contents are built when the folder is first opened: ~2,700 notes would be slow at once
  details.addEventListener("toggle", () => {
    if (!details.open || details.dataset.built) return;
    details.dataset.built = "1";
    details.append(libraryLevel(folder));
  });
  return details;
}

function libraryLevel(folder) {
  const box = el("div", "library-level");
  const folders = [...folder.folders.values()].sort((a, b) => a.name.localeCompare(b.name, "es"));
  for (const sub of folders) box.append(libraryFolder(sub));
  if (folder.notes.length) {
    const list = el("ul", "library-list");
    const notes = [...folder.notes].sort((a, b) => a.title.localeCompare(b.title, "es"));
    list.append(...notes.map((n) => libraryNoteItem(n, false)));
    box.append(list);
  }
  return box;
}

// --- «Mis notas»: the user's own notes by folder (ADR-033) -----------------------------------

function libraryOwnNotes() {
  return library.notes.filter((n) => !n.book);
}

function renderLibraryNotes(query) {
  const tree = $("library-tree");
  if (!query) {
    $("library-empty").hidden = true;
    tree.replaceChildren(libraryLevel(library.tree));
    return;
  }
  const words = query.split(/\s+/);
  const found = libraryOwnNotes().filter((n) => {
    const hay = libraryFold(`${n.title} ${n.path}`);
    return words.every((w) => hay.includes(w));
  });
  found.sort((a, b) => {
    // titles that contain the words first, then by title
    const ta = words.every((w) => libraryFold(a.title).includes(w)) ? 0 : 1;
    const tb = words.every((w) => libraryFold(b.title).includes(w)) ? 0 : 1;
    return ta - tb || a.title.localeCompare(b.title, "es");
  });
  $("library-empty").hidden = found.length > 0;
  const list = el("ul", "library-list library-results");
  list.append(...found.slice(0, 200).map((n) => libraryNoteItem(n, true)));
  const nodes = [list];
  if (found.length > 200) nodes.push(el("p", "small muted", tx("Y {0} más: escribe algo más concreto.", found.length - 200)));
  tree.replaceChildren(...nodes);
}

// --- the shelf (ADR-053) -----------------------------------------------------------------

function libraryBookColor(area) {
  let hash = 0;
  for (const ch of area) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
  return LIBRARY_BOOK_COLORS[hash % LIBRARY_BOOK_COLORS.length];
}

function libraryProgressText(book) {
  if (!book.read) return tx("Sin empezar");
  if (book.read >= book.chapters) return tx("Terminado");
  return tx("{0} de {1} leídos", book.read, book.chapters);
}

function libraryMeter(fraction) {
  const meter = el("span", "library-meter");
  const fill = el("span");
  fill.style.width = `${Math.round(Math.min(Math.max(fraction, 0), 1) * 100)}%`;
  meter.append(fill);
  meter.setAttribute("aria-hidden", "true");
  return meter;
}

function libraryBookCard(book) {
  const card = el("a", "library-book-card");
  card.href = `#/biblioteca?libro=${encodeURIComponent(book.id)}`;
  const cover = el("span", "library-cover");
  cover.style.backgroundColor = libraryBookColor(book.area);
  cover.append(el("span", "library-cover-area", book.area), el("span", "library-cover-title", book.title));
  const lang = book.english ? tx("inglés") : tx("español");
  card.append(
    cover,
    el("span", "library-book-meta", tx("{0} capítulos · {1}", book.chapters, lang)),
    libraryMeter(book.chapters ? book.read / book.chapters : 0),
    el("span", "library-book-progress", libraryProgressText(book)),
  );
  card.title = book.title;
  return card;
}

function renderLibraryAreas() {
  const box = $("library-areas");
  const chip = (name, label) => {
    const b = el("button", "chip", label);
    b.type = "button";
    b.setAttribute("aria-pressed", String(library.area === name));
    b.addEventListener("click", () => {
      library.area = name;
      renderLibrary();
    });
    return b;
  };
  box.replaceChildren(
    chip(null, tx("Todas las áreas")),
    ...library.shelf.areas.map((a) => chip(a.name, `${a.name} · ${a.count}`)),
  );
}

function renderLibraryShelf(query) {
  const words = query ? query.split(/\s+/) : [];
  // a book matches by its title or area, or by the title of one of its chapters
  const chapterText = new Map();
  if (words.length) {
    for (const n of library.notes) {
      if (n.book) chapterText.set(n.book, `${chapterText.get(n.book) || ""} ${n.title}`);
    }
  }
  const books = library.shelf.books.filter((b) => {
    if (library.area && b.area !== library.area) return false;
    if (!words.length) return true;
    const hay = libraryFold(`${b.title} ${b.area} ${chapterText.get(b.id) || ""}`);
    return words.every((w) => hay.includes(w));
  });
  // books in progress first, then the rest by title
  books.sort((a, b) => {
    const pa = a.read && a.read < a.chapters ? 0 : 1;
    const pb = b.read && b.read < b.chapters ? 0 : 1;
    return pa - pb || a.title.localeCompare(b.title, "es");
  });
  $("library-books-empty").hidden = books.length > 0;
  $("library-books").replaceChildren(...books.map(libraryBookCard));
}

// --- one book: its chapters in order --------------------------------------------------------

async function openLibraryBook(id) {
  $("library-books-view").hidden = true;
  $("library-notes-view").hidden = true;
  const view = $("library-book");
  view.hidden = false;
  $("library-book-title").textContent = tx("Abriendo…");
  $("library-book-meta").textContent = "";
  $("library-book-chapters").replaceChildren();
  $("library-book-go").hidden = true;
  let book;
  try {
    const response = await fetch(`/api/books/one?id=${encodeURIComponent(id)}`);
    if (!response.ok) throw new Error(tx("No encontré ese libro."));
    book = await response.json();
  } catch (error) {
    $("library-book-title").textContent = error.message;
    return;
  }
  const read = book.chapters.filter((c) => c.read).length;
  $("library-book-title").textContent = book.title;
  const lang = book.english ? tx("inglés") : tx("español");
  $("library-book-meta").textContent = tx("{0} · {1} capítulos · {2} · {3}", book.area, book.chapters.length, lang, libraryProgressText({ read, chapters: book.chapters.length }));
  const next = book.chapters.find((c) => !c.read);
  const go = $("library-book-go");
  if (book.chapters.length) {
    go.hidden = false;
    const target = next || book.chapters[0];
    const saved = recentNotes().find((r) => r.path === target.path);
    go.href = readerHref(target.path, saved && saved.line);
    go.textContent = !read ? tx("Empezar: {0}", target.title) : next ? tx("Seguir: {0}", target.title) : tx("Leer de nuevo");
  }
  const list = $("library-book-chapters");
  list.replaceChildren(...book.chapters.map((c) => {
    const li = el("li", c.read ? "library-chapter read" : "library-chapter");
    const link = el("a", null, c.title);
    link.href = readerHref(c.path);
    li.append(link);
    if (c.read) li.append(el("span", "library-read", tx("✓ leído")));
    return li;
  }));
}

// --- «Seguir leyendo»: notes left part-way (where the reader saved them) ---------------------

function renderLibraryContinue() {
  const known = new Map(library.notes.map((n) => [n.path, n]));
  const items = recentNotes()
    .filter((r) => known.has(r.path) && r.percent > 0 && r.percent < 95)
    .slice(0, LIBRARY_CONTINUE_MAX);
  $("library-continue").hidden = !items.length;
  $("library-continue-list").replaceChildren(...items.map((r) => {
    const card = el("a", "library-continue-card");
    card.href = readerHref(r.path, r.line);
    const where = r.book || r.path.split("/").slice(0, -1).join(" › ");
    const left = r.minutes ? tx(" · quedan ~{0} min", r.minutes) : "";
    const line = r.section ? tx("Te quedaste en «{0}»{1}", r.section, left) : tx("Vas en el {0} %{1}", r.percent, left);
    card.append(el("span", "library-continue-where", where), el("strong", null, r.title), el("span", "library-continue-line", line), libraryMeter(r.percent / 100));
    return card;
  }));
}

// --- the page ----------------------------------------------------------------------------

function librarySetTab(tab) {
  library.tab = tab;
  store.set("biblio.libraryTab", tab);
  if (location.hash.includes("libro=")) location.hash = "#/biblioteca";
  renderLibrary();
}

function renderLibrary() {
  const query = libraryFold($("library-filter").value.trim());
  const books = library.tab === "books";
  $("library-tab-books").setAttribute("aria-selected", String(books));
  $("library-tab-notes").setAttribute("aria-selected", String(!books));
  $("library-book").hidden = true;
  $("library-books-view").hidden = !books;
  $("library-notes-view").hidden = books;
  if (library.shelf) {
    const chapters = library.shelf.books.reduce((n, b) => n + b.chapters, 0);
    $("library-count").textContent = tx("{0} libros · {1} capítulos · {2} notas tuyas", library.shelf.books.length, chapters, libraryOwnNotes().length);
    $("library-tab-books").textContent = tx("Libros · {0}", library.shelf.books.length);
    $("library-tab-notes").textContent = tx("Mis notas · {0}", libraryOwnNotes().length);
  }
  renderLibraryContinue();
  if (books && library.shelf) {
    renderLibraryAreas();
    renderLibraryShelf(query);
  } else if (!books) {
    renderLibraryNotes(query);
  }
}

async function loadLibrary() {
  try {
    const [notes, shelf] = await Promise.all([
      fetch("/api/notes").then((r) => r.json()),
      fetch("/api/books").then((r) => r.json()),
    ]);
    library.notes = notes.notes;
    library.shelf = shelf;
  } catch {
    $("library-count").textContent = tx("No se pudo leer la lista de notas.");
    return false;
  }
  const signature = library.notes.map((n) => `${n.path}${n.read ? "+" : ""}`).join("|");
  if (signature !== library.signature) {
    library.signature = signature;
    library.tree = libraryTree(libraryOwnNotes());
  }
  library.loaded = true;
  return true;
}

let libraryBound = false;

pageHooks.biblioteca = async () => {
  if (!libraryBound) {
    libraryBound = true;
    const saved = store.get("biblio.libraryTab", "books");
    library.tab = saved === "notes" ? "notes" : "books";
    let timer = null;
    $("library-filter").addEventListener("input", () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        if (location.hash.includes("libro=")) location.hash = "#/biblioteca";
        else renderLibrary();
      }, 120);
    });
    $("library-tab-books").addEventListener("click", () => librarySetTab("books"));
    $("library-tab-notes").addEventListener("click", () => librarySetTab("notes"));
    $("library-book-back").addEventListener("click", () => {
      location.hash = "#/biblioteca";
    });
  }
  if (!(await loadLibrary())) return; // each visit: read marks and saved places may have changed
  const book = location.hash.match(/[?&]libro=([^&]*)/);
  if (book) {
    renderLibraryContinue();
    openLibraryBook(decodeURIComponent(book[1]));
  } else {
    renderLibrary();
    $("library-filter").focus();
  }
};
