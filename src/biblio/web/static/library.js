// Biblioteca — the «Biblioteca» page (ADR-033): every indexed note by folder, a filter, and the
// notes read most recently; each opens in the reader (reader.js, ADR-031).
// Uses the helpers of app.js ($, el, store) and readerHref/recentNotes of reader.js.
"use strict";

const library = { notes: [], tree: null, loaded: false, signature: "" };

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
  li.append(link);
  if (note.read) {
    const mark = el("span", "library-read", "✓ leída");
    mark.title = "Marcada como leída";
    li.append(mark);
  }
  if (showPath) li.append(el("span", "library-path", note.path.split("/").slice(0, -1).join(" › ") || "Raíz"));
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

function renderLibrary() {
  const query = libraryFold($("library-filter").value.trim());
  const tree = $("library-tree");
  $("library-count").textContent = `${library.notes.length} notas en tu biblioteca`;
  renderLibraryRecent(!query);
  if (!query) {
    $("library-empty").hidden = true;
    tree.replaceChildren(libraryLevel(library.tree));
    return;
  }
  const words = query.split(/\s+/);
  const found = library.notes.filter((n) => {
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
  if (found.length > 200) nodes.push(el("p", "small muted", `Y ${found.length - 200} más: escribe algo más concreto.`));
  tree.replaceChildren(...nodes);
}

function renderLibraryRecent(show) {
  const known = new Map(library.notes.map((n) => [n.path, n]));
  const recent = recentNotes().map((r) => known.get(r.path)).filter(Boolean);
  $("library-recent").hidden = !show || !recent.length;
  $("library-recent-list").replaceChildren(...recent.map((n) => libraryNoteItem(n, true)));
}

async function loadLibrary() {
  try {
    const data = await (await fetch("/api/notes")).json();
    library.notes = data.notes;
  } catch {
    $("library-count").textContent = "No se pudo leer la lista de notas.";
    return;
  }
  const signature = library.notes.map((n) => `${n.path}${n.read ? "+" : ""}`).join("|");
  if (signature === library.signature && !$("library-filter").value.trim()) {
    renderLibraryRecent(true); // same notes: keep the folders as they were opened
    return;
  }
  library.signature = signature;
  library.tree = libraryTree(library.notes);
  library.loaded = true;
  renderLibrary();
}

let libraryBound = false;

pageHooks.biblioteca = () => {
  if (!libraryBound) {
    libraryBound = true;
    let timer = null;
    $("library-filter").addEventListener("input", () => {
      clearTimeout(timer);
      timer = setTimeout(renderLibrary, 120);
    });
  }
  loadLibrary(); // each visit: read marks and recent notes may have changed
  $("library-filter").focus();
};
