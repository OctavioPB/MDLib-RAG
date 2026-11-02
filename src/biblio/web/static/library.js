// Biblioteca — the «Biblioteca» page: the user's books as a shelf or a list (ADR-053, ADR-054),
// their own notes by area and topic (ADR-033, ADR-055), and «Seguir leyendo» with where each note
// was left; everything opens in the reader (reader.js, ADR-031).
// Uses the helpers of app.js ($, el, store) and readerHref/recentNotes of reader.js.
"use strict";

const library = {
  notes: [],
  areaNames: [], // every area of books and notes, alphabetical: their colours
  loaded: false,
  tab: "books",
  shelf: null, // {books, areas} from /api/books
  area: null, // the area chosen on the shelf, null for all
  status: "all", // all | reading | done | spanish (ADR-054)
  sort: "recent", // recent | title | chapters | area
  view: "grid", // grid | list
  notesArea: null, // «Mis notas» (ADR-055): the area chosen, null for all
  notesStatus: "all", // all | practiced | due
  notesSort: "size", // size | edited | title
  notesView: "grid", // grid (topics) | list (the folder tree)
  reveal: null, // a folder to open and scroll to in the folder tree (`#/biblioteca?carpeta=`)
  summaries: [], // kept summaries of chapters, notes and books (ADR-056 stage E)
};
const LIBRARY_TABS = ["books", "notes", "summaries"];
const LIBRARY_EXCERPT = 240;
const LIBRARY_CONTINUE_MAX = 3;
// one colour per area, given in the areas' alphabetical order so no two of the first sixteen
// share one; all keep white text above 4.5:1 (ADR-054, ADR-055)
const LIBRARY_BOOK_COLORS = ["#2F6B3A", "#6A2C70", "#4B3B8F", "#9A4410", "#1F5F7A", "#2A4F8F", "#7A2E3A", "#5E544A", "#7A5A0E", "#3A4256", "#1A6B57", "#8A2F55", "#4A5D23", "#6B3E26", "#2E3F8F", "#7B3F61"];
const LIBRARY_SORTS = ["recent", "title", "chapters", "area"];
const LIBRARY_STATUSES = ["all", "reading", "done", "spanish"];
const LIBRARY_NOTE_SORTS = ["size", "edited", "title"];
const LIBRARY_NOTE_STATUSES = ["all", "practiced", "due"];
const LIBRARY_EDITED_MAX = 4;
const LIBRARY_TOPIC_SUBS = 4;
const LIBRARY_SEARCH_MAX = 300;
const LIBRARY_OPEN_MAX = 200; // folders remembered as open

function libraryFold(text) {
  return text.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
}

function libraryNatural(a, b) {
  return a.localeCompare(b, "es", { numeric: true });
}

// one colour per area for books and notes alike: areas in alphabetical order (ADR-054, ADR-055)
function libraryAreaColor(area) {
  const i = Math.max(library.areaNames.indexOf(area), 0);
  return LIBRARY_BOOK_COLORS[i % LIBRARY_BOOK_COLORS.length];
}

function libraryNoteItem(note) {
  const li = el("li", "library-note");
  const link = el("a", null, note.title);
  link.href = readerHref(note.path);
  link.title = note.title;
  li.append(link);
  if (note.due) li.append(el("span", "library-mark due", tx("Para repasar")));
  else if (note.practiced) li.append(el("span", "library-mark", tx("Practicada")));
  if (note.read) {
    const mark = el("span", "library-read", tx("✓ leída"));
    mark.title = tx("Marcada como leída");
    li.append(mark);
  }
  return li;
}

// --- «Mis notas»: the user's own notes by area and topic (ADR-033, ADR-055) ----------------

function libraryOwnNotes() {
  return library.notes.filter((n) => !n.book);
}

// a topic is a folder inside an area (`Data Science/Syllabus`); an area's loose notes are a
// topic of their own (id = the area) and so are the notes at the library's root (id = "")
function libraryTopicId(path) {
  const parts = path.split("/");
  if (parts.length >= 3) return `${parts[0]}/${parts[1]}`;
  return parts.length === 2 ? parts[0] : "";
}

function libraryTopicName(id) {
  if (id.includes("/")) return id.split("/")[1];
  return id ? tx("Notas sueltas") : tx("Raíz");
}

function libraryTopicArea(id) {
  return id ? id.split("/")[0] : tx("Raíz");
}

function libraryNoteMatches(note) {
  if (library.notesStatus === "practiced" && !note.practiced) return false;
  if (library.notesStatus === "due" && !note.due) return false;
  if (library.notesArea && note.path.split("/")[0] !== library.notesArea) return false;
  return true;
}

function libraryTopics(notes) {
  const topics = new Map();
  for (const note of notes) {
    const id = libraryTopicId(note.path);
    if (!topics.has(id)) topics.set(id, { id, area: libraryTopicArea(id), name: libraryTopicName(id), notes: [], subs: new Map(), changed: "", due: 0 });
    const topic = topics.get(id);
    topic.notes.push(note);
    if (note.changed > topic.changed) topic.changed = note.changed;
    if (note.due) topic.due += 1;
    const parts = note.path.split("/");
    if (parts.length >= 4) topic.subs.set(parts[2], (topic.subs.get(parts[2]) || 0) + 1);
  }
  return [...topics.values()];
}

function libraryTopicCard(topic) {
  const card = el("a", "library-topic-card");
  card.href = `#/biblioteca?tema=${encodeURIComponent(topic.id)}`;
  const head = el("span", "library-topic-cover");
  head.style.backgroundColor = libraryAreaColor(topic.area);
  head.append(el("span", "library-cover-area", topic.area), el("span", "library-topic-name", topic.name));
  const body = el("span", "library-topic-body");
  const meta = topic.due ? tx("{0} notas · {1} para repasar", topic.notes.length, topic.due) : tx("{0} notas", topic.notes.length);
  body.append(el("span", "library-book-meta", meta));
  if (topic.subs.size) {
    const subs = [...topic.subs.entries()].sort((a, b) => b[1] - a[1] || libraryNatural(a[0], b[0]));
    const chips = el("span", "library-subs");
    chips.append(...subs.slice(0, LIBRARY_TOPIC_SUBS).map(([name, n]) => el("span", "library-sub", `${name} · ${n}`)));
    if (subs.length > LIBRARY_TOPIC_SUBS) chips.append(el("span", "library-sub", tx("+{0} más", subs.length - LIBRARY_TOPIC_SUBS)));
    body.append(chips);
  }
  card.append(head, body);
  card.title = topic.id || topic.name;
  return card;
}

function libraryOpenFolders() {
  const saved = store.get("biblio.libraryOpenFolders", []);
  return new Set(Array.isArray(saved) ? saved : []);
}

function librarySaveFolder(folder, open) {
  const folders = libraryOpenFolders();
  if (open) folders.add(folder);
  else folders.delete(folder);
  store.set("biblio.libraryOpenFolders", [...folders].slice(-LIBRARY_OPEN_MAX));
}

const libraryBuilders = new WeakMap(); // a folder panel → the function that fills it

function libraryBuildPanel(panel) {
  const build = libraryBuilders.get(panel);
  if (build) build();
}

// «Expandir todo» / «Retraer todo» over every folder inside `box`, at any depth; expanding is
// not remembered (it would fill the list of open folders), retracting forgets them
function libraryFoldBar(box, label) {
  const bar = el("div", "library-fold-bar");
  const button = (text, open) => {
    const b = el("button", "btn-ghost", text);
    b.type = "button";
    b.addEventListener("click", () => {
      if (open) {
        // each panel is filled when opened, which adds the next level's panels
        let closed = [...box.querySelectorAll("details.library-folder:not([open])")];
        while (closed.length) {
          for (const panel of closed) {
            libraryBuildPanel(panel);
            panel.open = true;
          }
          closed = [...box.querySelectorAll("details.library-folder:not([open])")];
        }
      } else {
        for (const panel of box.querySelectorAll("details.library-folder[open]")) {
          panel.open = false;
          librarySaveFolder(panel.dataset.folder, false);
        }
      }
    });
    return b;
  };
  bar.append(el("span", "small muted", label), button(tx("Expandir todo"), true), button(tx("Retraer todo"), false));
  return bar;
}

// the notes as a folder tree under `base` ("" = the library): each folder with its subfolders,
// its own notes and how many notes it holds in all
function libraryFolderTree(notes, base) {
  const prefix = base ? `${base}/` : "";
  const root = { name: "", path: base, folders: new Map(), notes: [], count: 0 };
  for (const note of notes) {
    const parts = (note.path.startsWith(prefix) ? note.path.slice(prefix.length) : note.path).split("/");
    let node = root;
    node.count += 1;
    for (const part of parts.slice(0, -1)) {
      if (!node.folders.has(part)) {
        node.folders.set(part, { name: part, path: node.path ? `${node.path}/${part}` : part, folders: new Map(), notes: [], count: 0 });
      }
      node = node.folders.get(part);
      node.count += 1;
    }
    node.notes.push(note);
  }
  return root;
}

function libraryFolderCount(node) {
  let n = node.folders.size;
  for (const sub of node.folders.values()) n += libraryFolderCount(sub);
  return n;
}

// a folder's contents as in a file explorer (ADR-055): its subfolders, then its notes, both in
// the natural order of their names
function libraryFolderContents(node, opened, openAll) {
  const nodes = [...node.folders.values()]
    .sort((a, b) => libraryNatural(a.name, b.name))
    .map((sub) => libraryFolderPanel(sub, opened, openAll));
  if (node.notes.length) {
    const list = el("ul", "library-list");
    list.append(...[...node.notes].sort((a, b) => libraryNatural(a.path, b.path)).map(libraryNoteItem));
    nodes.push(list);
  }
  return nodes;
}

// a folder that opens and closes; it is filled when first opened, and a folder the person
// opened stays open next time
function libraryFolderPanel(node, opened, openAll) {
  const panel = el("details", "library-folder");
  panel.dataset.folder = node.path;
  const head = el("summary", "library-folder-head");
  head.append(el("span", "library-group-name", node.name), el("span", "library-count", String(node.count)));
  const body = el("div", "library-folder-body");
  panel.append(head, body);
  libraryBuilders.set(panel, () => {
    if (panel.dataset.built) return;
    panel.dataset.built = "1";
    body.append(...libraryFolderContents(node, opened, openAll));
  });
  panel.addEventListener("toggle", () => {
    if (panel.open) libraryBuildPanel(panel);
  });
  head.addEventListener("click", () => setTimeout(() => librarySaveFolder(node.path, panel.open), 0));
  if (openAll || opened.has(node.path)) {
    libraryBuildPanel(panel);
    panel.open = true;
  }
  return panel;
}

// the tree under `base` in `box`, with its fold bar; `reveal` is a folder to open and scroll to
function renderLibraryFolders(box, notes, base, openAll, reveal) {
  const tree = libraryFolderTree(notes, base);
  const opened = libraryOpenFolders();
  if (reveal) {
    const parts = reveal.split("/");
    for (let i = 1; i <= parts.length; i++) opened.add(parts.slice(0, i).join("/"));
  }
  const nodes = libraryFolderContents(tree, opened, openAll);
  const folders = libraryFolderCount(tree);
  box.replaceChildren(...(folders > 1 ? [libraryFoldBar(box, tx("{0} notas en {1} carpetas", tree.count, folders))] : []), ...nodes);
  if (reveal) {
    const target = [...box.querySelectorAll("details.library-folder")].find((d) => d.dataset.folder === reveal);
    if (target) setTimeout(() => target.scrollIntoView({ block: "start" }), 0);
  }
}

function libraryWhen(iso) {
  const days = Math.floor((Date.now() - Date.parse(iso)) / 86400000);
  if (days < 1) return tx("hoy");
  if (days < 2) return tx("ayer");
  return tx("hace {0} días", days);
}

// notes whose text changed after the index was built: a full rebuild dates every note alike,
// so the oldest date is the build's and says nothing
function renderLibraryEdited(own, query) {
  const oldest = own.reduce((m, n) => (n.changed && n.changed < m ? n.changed : m), "9999");
  const edited = own.filter((n) => n.changed && n.changed > oldest)
    .sort((a, b) => b.changed.localeCompare(a.changed))
    .slice(0, LIBRARY_EDITED_MAX);
  $("library-edited").hidden = Boolean(query) || !edited.length;
  $("library-edited-list").replaceChildren(...edited.map((n) => {
    const card = el("a", "library-edited-card");
    card.href = readerHref(n.path);
    const where = n.path.split("/").slice(0, -1).join(" › ") || tx("Raíz");
    card.append(el("span", "library-continue-where", where), el("strong", null, n.title), el("span", "library-edited-when", libraryWhen(n.changed)));
    card.title = n.path;
    return card;
  }));
}

function renderLibraryNotesToolbar(own) {
  const areas = new Map();
  for (const n of own) {
    const area = n.path.includes("/") ? n.path.split("/")[0] : "";
    if (area) areas.set(area, (areas.get(area) || 0) + 1);
  }
  const select = $("library-notes-area");
  const option = (value, label) => {
    const o = el("option", null, label);
    o.value = value;
    return o;
  };
  const sorted = [...areas.entries()].sort((a, b) => b[1] - a[1]);
  select.replaceChildren(option("", tx("Todas las áreas ({0})", sorted.length)), ...sorted.map(([a, n]) => option(a, `${a} · ${n}`)));
  if (library.notesArea && !areas.has(library.notesArea)) library.notesArea = null;
  select.value = library.notesArea || "";
  $("library-notes-sort").value = library.notesSort;
  const counts = { practiced: own.filter((n) => n.practiced).length, due: own.filter((n) => n.due).length };
  if (library.notesStatus !== "all" && !counts[library.notesStatus]) library.notesStatus = "all";
  const labels = { all: tx("Todas"), practiced: tx("Con práctica · {0}", counts.practiced), due: tx("Para repasar · {0}", counts.due) };
  const chips = LIBRARY_NOTE_STATUSES.filter((s) => s === "all" || counts[s] > 0).map((s) => {
    const b = el("button", "chip", labels[s]);
    b.type = "button";
    b.setAttribute("aria-pressed", String(library.notesStatus === s));
    b.addEventListener("click", () => {
      library.notesStatus = s;
      store.set("biblio.libraryNotesStatus", s);
      renderLibrary();
    });
    return b;
  });
  $("library-notes-status").replaceChildren(...(chips.length > 1 ? chips : []));
  $("library-notes-view-grid").setAttribute("aria-pressed", String(library.notesView === "grid"));
  $("library-notes-view-list").setAttribute("aria-pressed", String(library.notesView === "list"));
}

function renderLibraryNotes(query) {
  const own = libraryOwnNotes();
  renderLibraryEdited(own, query);
  renderLibraryNotesToolbar(own);
  let notes = own.filter(libraryNoteMatches);
  if (query) {
    const words = query.split(/\s+/);
    notes = notes.filter((n) => {
      const hay = libraryFold(`${n.title} ${n.path}`);
      return words.every((w) => hay.includes(w));
    });
  }
  $("library-notes-shown").textContent = notes.length === own.length ? "" : tx("{0} de {1} notas", notes.length, own.length);
  $("library-empty").hidden = notes.length > 0;
  // a search always shows the notes found where they are; without one, the view chosen
  const list = Boolean(query) || library.notesView === "list";
  // the folder tree goes by name, as a file explorer: the order only sorts the topics
  $("library-notes-sort").closest("label").hidden = list;
  $("library-topics").hidden = list;
  $("library-tree").hidden = !list;
  if (!list) {
    const topics = libraryTopics(notes);
    if (library.notesSort === "size") topics.sort((a, b) => b.notes.length - a.notes.length || libraryNatural(a.id, b.id));
    else if (library.notesSort === "edited") topics.sort((a, b) => b.changed.localeCompare(a.changed));
    else topics.sort((a, b) => libraryNatural(a.name, b.name) || libraryNatural(a.area, b.area));
    $("library-topics").replaceChildren(...topics.map(libraryTopicCard));
    $("library-tree").replaceChildren();
    return;
  }
  const shown = query ? notes.slice(0, LIBRARY_SEARCH_MAX) : notes;
  // an area chosen in the toolbar starts open, so its folders are the first level seen
  const reveal = library.reveal || (!query && library.notesArea) || null;
  renderLibraryFolders($("library-tree"), shown, "", Boolean(query), reveal);
  if (notes.length > shown.length) $("library-tree").append(el("p", "small muted", tx("Y {0} más: escribe algo más concreto.", notes.length - shown.length)));
  $("library-topics").replaceChildren();
}

// --- one topic: its notes grouped by subfolder ---------------------------------------------

function openLibraryTopic(id) {
  $("library-books-view").hidden = true;
  $("library-notes-view").hidden = true;
  $("library-book").hidden = true;
  $("library-topic").hidden = false;
  $("library-tab-books").setAttribute("aria-selected", "false");
  $("library-tab-notes").setAttribute("aria-selected", "true");
  const notes = libraryOwnNotes().filter((n) => libraryTopicId(n.path) === id);
  const name = libraryTopicName(id);
  $("library-topic-title").textContent = name;
  const due = notes.filter((n) => n.due).length;
  const meta = [tx("{0} notas", notes.length)];
  if (due) meta.push(tx("{0} para repasar", due));
  $("library-topic-meta").textContent = meta.join(" · ");
  const crumbs = $("library-topic-crumbs");
  const home = el("a", null, tx("Mis notas"));
  home.href = "#/biblioteca";
  crumbs.replaceChildren(home);
  // each folder of the path opens the folder view at that place
  const parts = id ? id.split("/") : [];
  parts.forEach((part, i) => {
    const link = el("a", null, i === 1 ? name : part);
    link.href = `#/biblioteca?carpeta=${encodeURIComponent(parts.slice(0, i + 1).join("/"))}`;
    crumbs.append(el("span", null, "›"), link);
  });
  if (!id) crumbs.append(el("span", null, "›"), el("span", null, name));
  // the study plan and the quiz work on a folder: a topic's, or an area's for its loose notes
  $("library-topic-plan").hidden = !id;
  $("library-topic-practice").hidden = !id;
  $("library-topic-plan").href = `#/guias?plan=${encodeURIComponent(id)}`;
  $("library-topic-practice").href = `#/practicar?carpeta=${encodeURIComponent(id)}`;
  if (notes.length) renderLibraryFolders($("library-topic-groups"), notes, id, false, null);
  else $("library-topic-groups").replaceChildren(el("p", "muted", tx("Este tema ya no tiene notas.")));
}

// --- the shelf (ADR-053) -----------------------------------------------------------------

function libraryReading(book) {
  return book.read > 0 && book.read < book.chapters;
}

function libraryDone(book) {
  return book.chapters > 0 && book.read >= book.chapters;
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

function libraryBookHref(book) {
  return `#/biblioteca?libro=${encodeURIComponent(book.id)}`;
}

// a card says only what tells books apart (ADR-054): the language only for the few in Spanish,
// the progress only once a book is started
function libraryBookCard(book) {
  const card = el("a", "library-book-card");
  card.href = libraryBookHref(book);
  const cover = el("span", "library-cover");
  cover.style.backgroundColor = libraryAreaColor(book.area);
  cover.append(el("span", "library-cover-area", book.area), el("span", "library-cover-title", book.title));
  card.append(cover);
  if (book.read) {
    const line = el("span", "library-book-progress");
    line.append(libraryMeter(book.chapters ? book.read / book.chapters : 0), el("span", null, libraryProgressText(book)));
    card.append(line);
  } else {
    const meta = book.english ? tx("{0} capítulos", book.chapters) : tx("{0} capítulos · {1}", book.chapters, tx("español"));
    card.append(el("span", "library-book-meta", meta));
  }
  card.title = book.title;
  return card;
}

function libraryTableRow(book, hit) {
  const tr = el("tr");
  const title = el("td", "library-table-title");
  const link = el("a", null);
  link.href = libraryBookHref(book);
  const dot = el("span", "library-dot");
  dot.style.backgroundColor = libraryAreaColor(book.area);
  dot.setAttribute("aria-hidden", "true");
  link.append(dot, el("span", null, book.title));
  title.append(link);
  const lang = book.english ? "" : ` · ${tx("español")}`;
  tr.append(
    title,
    el("td", "library-table-area", book.area),
    el("td", "num", String(book.chapters)),
    el("td", "library-table-read", libraryProgressText(book) + lang),
  );
  if (hit !== undefined) tr.append(el("td", "library-table-hit", hit));
  return tr;
}

function renderLibraryToolbar() {
  const select = $("library-area");
  const option = (value, label) => {
    const o = el("option", null, label);
    o.value = value;
    return o;
  };
  select.replaceChildren(
    option("", tx("Todas las áreas ({0})", library.shelf.areas.length)),
    ...library.shelf.areas.map((a) => option(a.name, `${a.name} · ${a.count}`)),
  );
  select.value = library.area || "";
  $("library-sort").value = library.sort;
  // a status filter shows only when some book has it, so an unread library is not a row of zeros
  const books = library.shelf.books;
  const counts = {
    reading: books.filter(libraryReading).length,
    done: books.filter(libraryDone).length,
    spanish: books.filter((b) => !b.english).length,
  };
  const labels = { all: tx("Todos"), reading: tx("En curso · {0}", counts.reading), done: tx("Terminados · {0}", counts.done), spanish: tx("En español · {0}", counts.spanish) };
  if (library.status !== "all" && !counts[library.status]) library.status = "all";
  const chips = LIBRARY_STATUSES.filter((s) => s === "all" || counts[s] > 0).map((s) => {
    const b = el("button", "chip", labels[s]);
    b.type = "button";
    b.setAttribute("aria-pressed", String(library.status === s));
    b.addEventListener("click", () => {
      library.status = s;
      store.set("biblio.libraryStatus", s);
      renderLibrary();
    });
    return b;
  });
  $("library-status").replaceChildren(...(chips.length > 1 ? chips : []));
  $("library-view-grid").setAttribute("aria-pressed", String(library.view === "grid"));
  $("library-view-list").setAttribute("aria-pressed", String(library.view === "list"));
}

// books whose chapters were opened lately, most recent first: from this browser's recent notes
function libraryRecentRank() {
  const bookOf = new Map(library.notes.filter((n) => n.book).map((n) => [n.path, n.book]));
  const rank = new Map();
  recentNotes().forEach((r, i) => {
    const id = bookOf.get(r.path);
    if (id && !rank.has(id)) rank.set(id, i);
  });
  return rank;
}

function librarySorted(books) {
  const byTitle = (a, b) => a.title.localeCompare(b.title, "es", { numeric: true });
  if (library.sort === "title") return books.sort(byTitle);
  if (library.sort === "chapters") return books.sort((a, b) => b.chapters - a.chapters || byTitle(a, b));
  if (library.sort === "area") return books.sort((a, b) => a.area.localeCompare(b.area, "es") || byTitle(a, b));
  // «Abiertos hace poco»: opened lately, then the others in progress, then by title
  const rank = libraryRecentRank();
  const key = (b) => (rank.has(b.id) ? rank.get(b.id) : libraryReading(b) ? 10000 : 20000);
  return books.sort((a, b) => key(a) - key(b) || byTitle(a, b));
}

function renderLibraryShelf(query) {
  const words = query ? query.split(/\s+/) : [];
  // a book matches by its title or area, or by the title of one of its chapters
  const chapters = new Map();
  if (words.length) {
    for (const n of library.notes) {
      if (!n.book) continue;
      if (!chapters.has(n.book)) chapters.set(n.book, []);
      chapters.get(n.book).push(n.title);
    }
  }
  const hits = new Map();
  const books = library.shelf.books.filter((b) => {
    if (library.area && b.area !== library.area) return false;
    if (library.status === "reading" && !libraryReading(b)) return false;
    if (library.status === "done" && !libraryDone(b)) return false;
    if (library.status === "spanish" && b.english) return false;
    if (!words.length) return true;
    const own = libraryFold(`${b.title} ${b.area}`);
    if (words.every((w) => own.includes(w))) {
      hits.set(b.id, tx("Título"));
      return true;
    }
    const titles = chapters.get(b.id) || [];
    const all = libraryFold(`${own} ${titles.join(" ")}`);
    if (!words.every((w) => all.includes(w))) return false;
    // the chapter to show is the one with most of the words
    const best = titles.reduce((top, t) => {
      const n = words.filter((w) => libraryFold(t).includes(w)).length;
      return n > top.n ? { t, n } : top;
    }, { t: "", n: 0 });
    hits.set(b.id, best.t || tx("Varios capítulos"));
    return true;
  });
  librarySorted(books);
  const total = library.shelf.books.length;
  $("library-shown").textContent = books.length === total ? "" : tx("{0} de {1} libros", books.length, total);
  $("library-books-empty").hidden = books.length > 0;
  const list = library.view === "list";
  $("library-books").hidden = list;
  $("library-table-wrap").hidden = !list;
  if (list) {
    $("library-col-hit").hidden = !words.length;
    $("library-table-body").replaceChildren(...books.map((b) => libraryTableRow(b, words.length ? hits.get(b.id) : undefined)));
    $("library-books").replaceChildren();
  } else {
    $("library-books").replaceChildren(...books.map(libraryBookCard));
    $("library-table-body").replaceChildren();
  }
}

// ↑ and ↓ move between the list's rows; Enter opens the book (the row's link)
function libraryTableKeys(event) {
  if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
  const links = [...$("library-table-body").querySelectorAll("a")];
  const i = links.indexOf(document.activeElement);
  if (i < 0) return;
  const next = links[i + (event.key === "ArrowDown" ? 1 : -1)];
  if (next) {
    event.preventDefault();
    next.focus();
  }
}

// --- one book: its chapters in order --------------------------------------------------------

async function openLibraryBook(id) {
  $("library-books-view").hidden = true;
  $("library-notes-view").hidden = true;
  $("library-topic").hidden = true;
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
  const summarized = library.summaries.some((s) => s.kind === "book" && s.key === book.id);
  $("library-book-summary").hidden = !summarized;
  if (summarized) $("library-book-summary").href = librarySummaryHref({ kind: "book", key: book.id });
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
  const visible = items.length > 0 && store.get("biblio.libraryContinueVisible", true) !== false;
  const panel = $("library-continue");
  panel.hidden = !visible;
  const toggle = $("library-continue-toggle");
  toggle.hidden = items.length === 0;
  toggle.textContent = visible ? tx("Ocultar «Seguir leyendo»") : tx("Mostrar «Seguir leyendo»");
  toggle.setAttribute("aria-expanded", String(visible));
  toggle.setAttribute("aria-label", tx(visible ? "Ocultar panel Seguir leyendo" : "Mostrar panel Seguir leyendo"));
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
  if (/[?&](libro|tema|carpeta|resumen)=/.test(location.hash)) location.hash = "#/biblioteca";
  renderLibrary();
}

// --- kept summaries (ADR-056 stage E) ------------------------------------------------------

function librarySummaryHref(s) {
  return `#/biblioteca?resumen=${encodeURIComponent(`${s.kind}:${s.key}`)}`;
}

function librarySummaryMeta(s) {
  const where = s.kind === "book" ? tx("Libro · {0} capítulos", s.chapters)
    : s.book ? tx("Capítulo de «{0}»", s.book.title) : tx("Nota · {0}", s.area || tx("Raíz"));
  const when = s.created ? formatWhen(s.created) : "";
  return `${where} · ${when} · ${s.model}`;
}

function librarySummaryCard(s) {
  const card = el("a", "library-summary-card");
  card.href = librarySummaryHref(s);
  card.append(el("strong", null, s.title), el("span", "small muted", librarySummaryMeta(s)));
  if (s.stale) card.append(el("span", "library-mark due", tx("Desactualizado")));
  const text = s.overview.replace(/\s*\[\d+\]/g, ""); // the citations, with the space before them
  card.append(el("p", "small", text.length > LIBRARY_EXCERPT ? `${text.slice(0, LIBRARY_EXCERPT - 1)}…` : text));
  return card;
}

function renderLibrarySummaries(query) {
  const words = query.split(/\s+/).filter(Boolean);
  const shown = library.summaries.filter((s) => {
    const hay = libraryFold(`${s.title} ${s.overview} ${s.book ? s.book.title : ""} ${s.area}`);
    return words.every((w) => hay.includes(w));
  });
  const groups = new Map();
  for (const s of shown) {
    const name = s.kind === "book" ? tx("Libros") : s.book ? s.book.title : s.area || tx("Raíz");
    if (!groups.has(name)) groups.set(name, []);
    groups.get(name).push(s);
  }
  const order = [...groups.keys()].sort((a, b) => (a === tx("Libros") ? -1 : b === tx("Libros") ? 1 : a.localeCompare(b, "es")));
  $("library-summaries").replaceChildren(...order.map((name) => {
    const box = el("section", "library-summary-group");
    box.append(el("h2", "eyebrow", tx("{0} · {1}", name, groups.get(name).length)));
    const grid = el("div", "library-summary-grid");
    grid.append(...groups.get(name).map(librarySummaryCard));
    box.append(grid);
    return box;
  }));
  $("library-summaries-empty").hidden = shown.length > 0;
  $("library-summaries-empty").textContent = library.summaries.length
    ? tx("Ningún resumen coincide con tu búsqueda.")
    : tx("Todavía no hay resúmenes. Pídelos desde el lector con el botón «Resumen», en cualquier nota o capítulo.");
}

async function openLibrarySummary(kind, key) {
  for (const id of ["library-books-view", "library-notes-view", "library-summaries-view", "library-book", "library-topic"]) $(id).hidden = true;
  $("library-summary").hidden = false;
  $("library-summary-title").textContent = tx("Abriendo…");
  $("library-summary-meta").textContent = "";
  $("library-summary-actions").replaceChildren();
  $("library-summary-body").replaceChildren();
  let data;
  try {
    const response = await fetch(`/api/summaries/one?kind=${encodeURIComponent(kind)}&key=${encodeURIComponent(key)}`);
    if (!response.ok) throw new Error(tx("No encontré ese resumen."));
    data = await response.json();
  } catch (error) {
    $("library-summary-title").textContent = error.message;
    return;
  }
  const listed = library.summaries.find((s) => s.kind === kind && s.key === key);
  $("library-summary-title").textContent = data.title;
  $("library-summary-meta").textContent = listed ? librarySummaryMeta(listed) : data.model;
  const open = el("a", "btn-primary", kind === "book" ? tx("Ver el libro") : tx("Abrir en el lector"));
  open.href = kind === "book" ? `#/biblioteca?libro=${encodeURIComponent(key)}` : readerHref(key);
  $("library-summary-actions").append(open);
  const body = [];
  if (data.stale) body.push(el("p", "small error", tx("El texto cambió desde este resumen. Ábrelo en el lector y pide el resumen otra vez para rehacerlo.")));
  body.push(...(kind === "book" ? readerSummaryBook(data) : readerSummaryNote(data)));
  $("library-summary-body").replaceChildren(...body);
}

function libraryRenderCounts() {
  if (!library.shelf) return;
  const chapters = library.shelf.books.reduce((n, b) => n + b.chapters, 0);
  $("library-count").textContent = tx("{0} libros · {1} capítulos · {2} notas tuyas", library.shelf.books.length, chapters, libraryOwnNotes().length);
  $("library-tab-books").textContent = tx("Libros · {0}", library.shelf.books.length);
  $("library-tab-notes").textContent = tx("Mis notas · {0}", libraryOwnNotes().length);
  $("library-tab-summaries").textContent = tx("Resúmenes · {0}", library.summaries.length);
}

function renderLibrary() {
  const query = libraryFold($("library-filter").value.trim());
  const books = library.tab === "books";
  const summaries = library.tab === "summaries";
  $("library-tab-books").setAttribute("aria-selected", String(books));
  $("library-tab-notes").setAttribute("aria-selected", String(library.tab === "notes"));
  $("library-tab-summaries").setAttribute("aria-selected", String(summaries));
  $("library-book").hidden = true;
  $("library-topic").hidden = true;
  $("library-summary").hidden = true;
  $("library-books-view").hidden = !books;
  $("library-notes-view").hidden = library.tab !== "notes";
  $("library-summaries-view").hidden = !summaries;
  libraryRenderCounts();
  renderLibraryContinue();
  if (books && library.shelf) {
    renderLibraryToolbar();
    renderLibraryShelf(query);
  } else if (summaries) {
    renderLibrarySummaries(query);
  } else if (!books) {
    renderLibraryNotes(query);
  }
}

async function loadLibrary() {
  try {
    const [notes, shelf, summaries] = await Promise.all([
      fetch("/api/notes").then((r) => r.json()),
      fetch("/api/books").then((r) => r.json()),
      fetch("/api/summaries").then((r) => (r.ok ? r.json() : { summaries: [] })),
    ]);
    library.notes = notes.notes;
    library.shelf = shelf;
    library.summaries = summaries.summaries;
  } catch {
    $("library-count").textContent = tx("No se pudo leer la lista de notas.");
    return false;
  }
  const areas = new Set(library.shelf.areas.map((a) => a.name));
  for (const n of libraryOwnNotes()) if (n.path.includes("/")) areas.add(n.path.split("/")[0]);
  library.areaNames = [...areas].sort((a, b) => a.localeCompare(b, "es"));
  library.loaded = true;
  return true;
}

let libraryBound = false;

pageHooks.biblioteca = async () => {
  if (!libraryBound) {
    libraryBound = true;
    const saved = store.get("biblio.libraryTab", "books");
    library.tab = LIBRARY_TABS.includes(saved) ? saved : "books";
    const sort = store.get("biblio.librarySort", "recent");
    library.sort = LIBRARY_SORTS.includes(sort) ? sort : "recent";
    library.view = store.get("biblio.libraryView", "grid") === "list" ? "list" : "grid";
    const status = store.get("biblio.libraryStatus", "all");
    library.status = LIBRARY_STATUSES.includes(status) ? status : "all";
    $("library-area").addEventListener("change", (event) => {
      library.area = event.target.value || null;
      renderLibrary();
    });
    $("library-sort").addEventListener("change", (event) => {
      library.sort = event.target.value;
      store.set("biblio.librarySort", library.sort);
      renderLibrary();
    });
    for (const view of ["grid", "list"]) {
      $(`library-view-${view}`).addEventListener("click", () => {
        library.view = view;
        store.set("biblio.libraryView", view);
        renderLibrary();
      });
    }
    $("library-table-body").addEventListener("keydown", libraryTableKeys);
    const notesSort = store.get("biblio.libraryNotesSort", "size");
    library.notesSort = LIBRARY_NOTE_SORTS.includes(notesSort) ? notesSort : "size";
    library.notesView = store.get("biblio.libraryNotesView", "grid") === "list" ? "list" : "grid";
    const notesStatus = store.get("biblio.libraryNotesStatus", "all");
    library.notesStatus = LIBRARY_NOTE_STATUSES.includes(notesStatus) ? notesStatus : "all";
    $("library-notes-area").addEventListener("change", (event) => {
      library.notesArea = event.target.value || null;
      renderLibrary();
    });
    $("library-notes-sort").addEventListener("change", (event) => {
      library.notesSort = event.target.value;
      store.set("biblio.libraryNotesSort", library.notesSort);
      renderLibrary();
    });
    for (const view of ["grid", "list"]) {
      $(`library-notes-view-${view}`).addEventListener("click", () => {
        library.notesView = view;
        store.set("biblio.libraryNotesView", view);
        renderLibrary();
      });
    }
    $("library-topic-back").addEventListener("click", () => {
      location.hash = "#/biblioteca";
    });
    let timer = null;
    $("library-filter").addEventListener("input", () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        if (/[?&](libro|tema|carpeta|resumen)=/.test(location.hash)) location.hash = "#/biblioteca";
        else renderLibrary();
      }, 120);
    });
    $("library-tab-books").addEventListener("click", () => librarySetTab("books"));
    $("library-tab-notes").addEventListener("click", () => librarySetTab("notes"));
    $("library-tab-summaries").addEventListener("click", () => librarySetTab("summaries"));
    $("library-summary-back").addEventListener("click", () => {
      library.tab = "summaries";
      location.hash = "#/biblioteca";
    });
    $("library-continue-toggle").addEventListener("click", () => {
      store.set("biblio.libraryContinueVisible", $("library-continue").hidden);
      renderLibraryContinue();
    });
    $("library-book-back").addEventListener("click", () => {
      location.hash = "#/biblioteca";
    });
  }
  if (!(await loadLibrary())) return; // each visit: read marks and saved places may have changed
  libraryRenderCounts(); // also on a book's or a topic's page, opened by its link
  const book = location.hash.match(/[?&]libro=([^&]*)/);
  const topic = location.hash.match(/[?&]tema=([^&]*)/);
  const folder = location.hash.match(/[?&]carpeta=([^&]*)/);
  const summary = location.hash.match(/[?&]resumen=([^&]*)/);
  if (summary) {
    const [kind, ...key] = decodeURIComponent(summary[1]).split(":");
    renderLibraryContinue();
    openLibrarySummary(kind, key.join(":"));
  } else if (folder) {
    // a folder's place: «Mis notas» as a folder tree, opened down to it
    library.tab = "notes";
    library.notesView = "list";
    library.notesArea = null;
    library.reveal = decodeURIComponent(folder[1]);
    renderLibrary();
    library.reveal = null;
  } else if (book) {
    renderLibraryContinue();
    openLibraryBook(decodeURIComponent(book[1]));
  } else if (topic) {
    library.tab = "notes";
    renderLibraryContinue();
    openLibraryTopic(decodeURIComponent(topic[1]));
  } else {
    renderLibrary();
    $("library-filter").focus();
  }
};
