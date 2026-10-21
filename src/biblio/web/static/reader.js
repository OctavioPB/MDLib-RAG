// Biblioteca — the built-in reader (ADR-031): a note rendered from the server's block tree, and
// «Explícame esto» / «Pregúntame sobre esto» on a selected passage, in a side panel.
// Uses the helpers of app.js ($, el, postJSON, errorText, streamEvents, markdownNode, gapNote,
// loadStatus, openConversation, app). Library and model text is only ever inserted as text; KaTeX
// (vendored, ADR-031) builds formulas through the DOM API.
"use strict";

const reader = {
  path: null,
  view: null,
  conversationId: null, // «Explícame esto» on this note continue one conversation
  passage: null, // the current selection: {text, lines}
};
const READER_KINDS = ["mc", "tf", "open_self"]; // answered in the panel (ADR-031)
const PASSAGE_MIN = 20; // as reader.passage_min_chars; the server says so if it changes
const PASSAGE_MAX = 3000;

function readerHref(path, line) {
  return `#/leer?path=${encodeURIComponent(path)}${line ? `&line=${line}` : ""}`;
}

// the notes opened most recently, for the «Biblioteca» page (ADR-033); kept in this browser only
const RECENT_MAX = 8;

function recentNotes() {
  const list = store.get("biblio.recentNotes", []);
  return Array.isArray(list) ? list : [];
}

function rememberRecent(path, title) {
  const list = recentNotes().filter((r) => r.path !== path);
  list.unshift({ path, title });
  store.set("biblio.recentNotes", list.slice(0, RECENT_MAX));
}

// «Leer» next to «Abrir en Obsidian» wherever a note appears
function readLink(path, line, cls = "btn-soft", label = tx("Leer")) {
  const link = el("a", cls, label);
  link.href = readerHref(path, line);
  link.title = tx("Leer la nota dentro de la app");
  return link;
}

// --- rendering -------------------------------------------------------------------------------

function renderMath(target, tex, display) {
  if (window.katex) {
    try {
      window.katex.render(tex, target, { displayMode: display, throwOnError: false, output: "htmlAndMathml" });
      return target;
    } catch {
      // shown as its source below
    }
  }
  target.classList.add("math-raw");
  target.textContent = display ? `$$${tex}$$` : `$${tex}$`;
  return target;
}

// Formulas written inside model text or a quoted passage: $…$, $$…$$, \(…\), \[…\]
const TEX_RE = /\$\$([^$]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|\$(?!\s)([^$\n]+?)(?<!\s)\$/g;

function renderMathIn(root) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const texts = [];
  while (walker.nextNode()) texts.push(walker.currentNode);
  for (const node of texts) {
    if (node.parentElement.closest(".katex, code, pre")) continue;
    const text = node.textContent;
    TEX_RE.lastIndex = 0;
    if (!TEX_RE.test(text)) continue;
    TEX_RE.lastIndex = 0;
    const parts = [];
    let last = 0;
    for (const m of text.matchAll(TEX_RE)) {
      if (m.index > last) parts.push(document.createTextNode(text.slice(last, m.index)));
      const display = Boolean(m[1] || m[2]);
      const tex = m[1] || m[2] || m[3] || m[4];
      parts.push(renderMath(el("span", display ? "math-display-inline" : "math-inline"), tex, display));
      last = m.index + m[0].length;
    }
    if (last < text.length) parts.push(document.createTextNode(text.slice(last)));
    node.replaceWith(...parts);
  }
  return root;
}

function readerInlines(nodes) {
  return nodes.map(readerInline);
}

function readerInline(n) {
  switch (n.t) {
    case "text":
      return document.createTextNode(n.text === "\n" ? " " : n.text);
    case "br":
      return document.createElement("br");
    case "strong":
    case "em":
    case "s": {
      const node = el(n.t);
      node.append(...readerInlines(n.children || []));
      return node;
    }
    case "code":
      return el("code", null, n.text);
    case "math":
      return renderMath(el(n.display ? "div" : "span", n.display ? "math-block" : "math-inline"), n.tex, Boolean(n.display));
    case "link": {
      let node;
      if (n.href) {
        node = el("a");
        node.href = n.href;
        node.target = "_blank";
        node.rel = "noopener noreferrer";
      } else if (n.note) {
        node = el("a");
        node.href = readerHref(n.note);
      } else {
        node = el("span", "link-missing");
      }
      node.append(...readerInlines(n.children || []));
      return node;
    }
    case "wikilink": {
      if (!n.note) {
        const missing = el("span", "wikilink missing", n.text);
        missing.title = tx("Esta nota no está en tu biblioteca indexada");
        return missing;
      }
      const link = el("a", "wikilink", n.text);
      link.href = readerHref(n.note);
      return link;
    }
    case "image":
      return el("span", "img-alt", tx("[Imagen: {0}]", n.alt || tx("sin descripción")));
    default:
      return document.createTextNode(n.text || "");
  }
}

function readerBlock(b) {
  let node;
  if (b.type === "heading") {
    node = el(`h${Math.min(Math.max(b.level, 1), 6)}`);
    node.append(...readerInlines(b.inlines || []));
  } else if (b.type === "p" || b.type === "th" || b.type === "td") {
    node = el(b.type);
    node.append(...readerInlines(b.inlines || []));
    if (b.align) node.style.textAlign = b.align;
  } else if (b.type === "code") {
    node = el("pre", "reader-code");
    const code = el("code", null, b.text);
    if (b.lang) node.dataset.lang = b.lang;
    node.append(code);
  } else if (b.type === "math") {
    node = renderMath(el("div", "math-block"), b.tex, true);
  } else if (b.type === "hr") {
    node = el("hr");
  } else if (b.type === "table") {
    node = el("div", "table-wrap");
    const table = el("table");
    table.append(...(b.children || []).map(readerBlock));
    node.append(table);
  } else {
    node = el(b.type); // ul, ol, li, blockquote, thead, tbody, tr
    if (b.type === "ol" && b.start) node.start = Number(b.start);
    node.append(...(b.children || []).map(readerBlock));
  }
  if (b.lines && b.type !== "tr" && b.type !== "thead" && b.type !== "tbody") {
    node.dataset.start = String(b.lines[0]);
    node.dataset.end = String(b.lines[1]);
  }
  return node;
}

function scrollToLine(line) {
  if (!line) {
    window.scrollTo(0, 0);
    return;
  }
  let best = null;
  for (const node of $("reader-body").querySelectorAll("[data-start]")) {
    if (Number(node.dataset.start) <= line && line <= Number(node.dataset.end)) best = node; // deepest last
    else if (!best && Number(node.dataset.start) > line) best = node;
  }
  if (!best) return;
  best.scrollIntoView({ block: "center" });
  best.classList.add("flash");
  setTimeout(() => best.classList.remove("flash"), 2200);
}

async function openNote(path, line) {
  if (reader.path !== path || !reader.view) {
    reader.path = path;
    reader.view = null;
    reader.conversationId = null;
    $("reader-panel-list").replaceChildren();
    $("reader-panel").hidden = true;
    readerFindClear();
    $("reader-find-input").value = "";
    $("reader-title").textContent = tx("Abriendo…");
    $("reader-body").replaceChildren();
    const response = await fetch(`/api/notes/view?path=${encodeURIComponent(path)}`);
    if (!response.ok) {
      $("reader-title").textContent = tx("No encontré esa nota");
      $("reader-path").textContent = path;
      $("reader-body").replaceChildren(el("p", "muted", tx("Solo se pueden leer las notas indexadas de tu biblioteca.")));
      $("reader-actions").hidden = true;
      return;
    }
    reader.view = await response.json();
    rememberRecent(reader.view.path, reader.view.title);
    renderNote();
  }
  scrollToLine(line);
}

function renderNote() {
  const view = reader.view;
  $("reader-title").textContent = view.title;
  $("reader-path").textContent = view.path;
  $("reader-actions").hidden = false;
  $("reader-open").hidden = !view.open_url;
  if (view.open_url) $("reader-open").href = view.open_url;
  renderReadMark();
  const blocks = view.blocks.slice();
  // the note's own title is the page header already
  if (blocks[0] && blocks[0].type === "heading" && blocks[0].level === 1) blocks.shift();
  $("reader-body").replaceChildren(...blocks.map(readerBlock));
  $("reader-find-wrap").hidden = false;
}

// --- search inside the note (ADR-045): its sections ranked, a click jumps to one ---------------

function readerFindClear() {
  $("reader-find-results").replaceChildren();
  $("reader-find-results").hidden = true;
}

async function readerFind(event) {
  event.preventDefault();
  const query = $("reader-find-input").value.trim();
  if (!query || !reader.path) return;
  const box = $("reader-find-results");
  box.hidden = false;
  box.replaceChildren(el("p", "small muted", tx("Buscando en la nota…")));
  try {
    const response = await postJSON("/api/notes/search", { path: reader.path, query });
    if (!response.ok) throw new Error(await errorText(response));
    const data = await response.json();
    const head = el("div", "reader-find-head");
    head.append(el("span", "small muted", data.hits.length
      ? tx("{0} {1} de esta nota, de la más a la menos relevante", data.hits.length, data.hits.length === 1 ? tx("sección") : tx("secciones"))
      : tx("Nada en esta nota trata de eso.")));
    const close = el("button", "btn-icon-plain", "✕");
    close.type = "button";
    close.setAttribute("aria-label", tx("Cerrar los resultados"));
    close.addEventListener("click", readerFindClear);
    head.append(close);
    const list = el("ol", "reader-find-list");
    for (const hit of data.hits) {
      const item = el("li");
      const go = el("button", "reader-find-item");
      go.type = "button";
      go.append(el("strong", null, hit.sections.join(" › ") || hit.guide), el("span", null, tx("líneas {0}–{1} · {2}", hit.lines[0], hit.lines[1], hit.excerpt.replace(/[*_`#>]+/g, ""))));
      go.addEventListener("click", () => {
        for (const other of list.querySelectorAll(".reader-find-item")) other.classList.toggle("active", other === go);
        scrollToLine(hit.lines[0]);
        // the section lands below the search bar, not under its results
        const target = $("reader-body").querySelector(".flash");
        const below = $("reader-find-wrap").getBoundingClientRect().bottom + 12;
        if (target && target.getBoundingClientRect().top < below) {
          window.scrollBy(0, target.getBoundingClientRect().top - below);
        }
      });
      item.append(go);
      list.append(item);
    }
    box.replaceChildren(head, list);
  } catch (error) {
    box.replaceChildren(el("p", "error", error.message));
  }
}

function renderReadMark() {
  const button = $("reader-read");
  button.textContent = reader.view.read ? tx("Leída ✓") : tx("Marcar como leída");
  button.setAttribute("aria-pressed", String(Boolean(reader.view.read)));
}

// --- selecting a passage -----------------------------------------------------------------------

const BLOCK_TAGS = new Set(["P", "LI", "H1", "H2", "H3", "H4", "H5", "H6", "PRE", "DIV", "TR", "BLOCKQUOTE", "TABLE", "UL", "OL"]);

// The selection as the note wrote it: each formula as its LaTeX source (KaTeX keeps it in an
// annotation), not the drawn glyphs plus their MathML copy; a formula cut by the selection whole.
function passageText(range) {
  const whole = range.cloneRange();
  const formula = (n) => (n.nodeType === 1 ? n : n.parentElement).closest(".katex-display, .katex");
  const first = formula(whole.startContainer);
  const last = formula(whole.endContainer);
  if (first) whole.setStartBefore(first);
  if (last) whole.setEndAfter(last);
  const box = document.createElement("div");
  box.append(whole.cloneContents());
  for (const k of box.querySelectorAll(".katex")) {
    const tex = k.querySelector('annotation[encoding="application/x-tex"]');
    const display = Boolean(k.closest(".katex-display"));
    const source = tex ? tex.textContent.trim() : k.textContent;
    k.replaceWith(document.createTextNode(display ? `$$${source}$$` : `$${source}$`));
  }
  const plain = (node) => {
    if (node.nodeType === 3) return node.textContent;
    if (node.nodeType !== 1) return "";
    const inner = [...node.childNodes].map(plain).join("");
    if (node.tagName === "BR") return "\n";
    if (node.tagName === "TD" || node.tagName === "TH") return `${inner} | `;
    return BLOCK_TAGS.has(node.tagName) ? `\n${inner}\n` : inner;
  };
  return plain(box).replace(/[ \t]+\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
}

function selectedPassage() {
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed || !sel.rangeCount) return null;
  const body = $("reader-body");
  if (!body.contains(sel.anchorNode) || !body.contains(sel.focusNode)) return null;
  const text = passageText(sel.getRangeAt(0));
  if (text.length < PASSAGE_MIN) return null;
  const blockOf = (n) => (n.nodeType === 1 ? n : n.parentElement).closest("[data-start]");
  const blocks = [blockOf(sel.anchorNode), blockOf(sel.focusNode)].filter(Boolean);
  if (!blocks.length) return null;
  return {
    text: text.slice(0, PASSAGE_MAX),
    lines: [Math.min(...blocks.map((b) => Number(b.dataset.start))), Math.max(...blocks.map((b) => Number(b.dataset.end)))],
    cut: text.length > PASSAGE_MAX,
  };
}

function updateToolbar() {
  const bar = $("reader-toolbar");
  const passage = currentPage() === "leer" ? selectedPassage() : null;
  reader.passage = passage;
  if (!passage) {
    bar.hidden = true;
    return;
  }
  const rect = window.getSelection().getRangeAt(0).getBoundingClientRect();
  bar.hidden = false;
  const width = bar.offsetWidth;
  const top = rect.top > 70 ? rect.top - bar.offsetHeight - 10 : rect.bottom + 10;
  const left = Math.min(Math.max(rect.left + rect.width / 2 - width / 2, 8), window.innerWidth - width - 8);
  bar.style.top = `${Math.round(top)}px`;
  bar.style.left = `${Math.round(left)}px`;
}

// --- the side panel --------------------------------------------------------------------------

function panelCard(kind, passage) {
  $("reader-panel").hidden = false;
  const card = el("section", "reader-card");
  const level = kind === tx("Explícame esto") ? ` · ${LEVEL_LABELS[app.settings.level]}` : "";
  card.append(el("span", "eyebrow", `${kind}${level}`));
  const quote = el("blockquote", "reader-quote", passage.text.length > 260 ? `${passage.text.slice(0, 259)}…` : passage.text);
  card.append(renderMathIn(quote));
  if (passage.cut) card.append(el("p", "small muted", tx("Usé los primeros {0} caracteres de la selección.", PASSAGE_MAX)));
  $("reader-panel-list").append(card);
  card.scrollIntoView({ block: "nearest" });
  return card;
}

async function needLLM(card) {
  await loadStatus();
  if (app.llmRunning) return true;
  card.append(el("p", "error", tx("El modelo de lenguaje no está encendido. Inícialo con scripts/serve-llm.ps1.")));
  return false;
}

function citeNode(fragments) {
  return (n) => {
    const f = fragments.find((x) => x.number === Number(n));
    const pill = el("a", "cite", `[${n}]`);
    if (!f) {
      pill.classList.add("invalid");
      return pill;
    }
    pill.title = f.citation;
    pill.href = readerHref(f.doc_path, f.lines[0]);
    if (f.doc_path === reader.path) {
      pill.addEventListener("click", (event) => {
        event.preventDefault();
        scrollToLine(f.lines[0]);
      });
    }
    return pill;
  };
}

async function explainPassage(passage) {
  const card = panelCard(tx("Explícame esto"), passage);
  if (!(await needLLM(card))) return;
  const status = el("p", "small muted", tx("Leyendo el pasaje y lo que dicen tus otras notas…"));
  const answer = el("div", "reader-answer");
  card.append(status, answer);
  let text = "";
  let fragments = [];
  try {
    await streamEvents(
      "/api/notes/explain",
      { path: reader.path, lines: passage.lines, text: passage.text, conversation_id: reader.conversationId, level: app.settings.level },
      (name, data) => {
        if (name === "meta") {
          fragments = data.fragments;
          status.textContent = tx("Con el pasaje y {0} {1} más de tus notas", fragments.length - 1, fragments.length === 2 ? tx("fragmento") : tx("fragmentos"));
        } else if (name === "token") {
          text += data.text;
          answer.replaceChildren(markdownNode(text, citeNode(fragments), "md answer")); // formulas once done
        } else if (name === "done") {
          reader.conversationId = data.conversation_id;
          answer.replaceChildren(renderMathIn(markdownNode(text, citeNode(fragments), "md answer")));
          card.append(flagsNode(data));
          const more = el("button", "link-button", tx("Seguir en Conversar →"));
          more.type = "button";
          more.addEventListener("click", async () => {
            location.hash = "#/conversar";
            await openConversation(data.conversation_id);
          });
          card.append(more);
        } else if (name === "gap") {
          card.append(gapNote(data));
        }
      },
    );
  } catch (error) {
    card.append(el("p", "error", error.message));
  }
}

function checkedKinds() {
  const kinds = [...document.querySelectorAll('input[name="practice-kind"]:checked')].map((i) => i.value).filter((k) => READER_KINDS.includes(k));
  return kinds.length ? kinds : READER_KINDS;
}

async function quizPassage(passage) {
  const card = panelCard(tx("Pregúntame sobre esto"), passage);
  if (!(await needLLM(card))) return;
  const status = el("p", "small muted", tx("Escribiendo preguntas sobre el pasaje…"));
  const list = el("div", "reader-questions");
  card.append(status, list);
  try {
    await streamEvents(
      "/api/notes/quiz",
      { path: reader.path, lines: passage.lines, text: passage.text, kinds: checkedKinds() },
      (name, data) => {
        if (name === "question") list.append(questionBox(data));
        else if (name === "done") {
          status.textContent = data.generated
            ? tx("Cuentan como práctica de esta sección para tus repasos.")
            : tx("No pude escribir preguntas sobre este pasaje.");
        }
      },
    );
  } catch (error) {
    status.textContent = "";
    card.append(el("p", "error", error.message));
  }
}

async function practicePost(path, body) {
  const response = await postJSON(path, body);
  if (!response.ok) throw new Error(await errorText(response));
  return response.json();
}

function questionBox(q) {
  const box = el("div", "reader-question");
  box.append(renderMathIn(el("p", "reader-question-text", q.text)));
  const feedback = el("div", "reader-feedback");
  const show = (good, data) => {
    feedback.replaceChildren();
    feedback.className = `reader-feedback ${good === true ? "good" : good === false ? "bad" : "neutral"}`;
    if (good !== null) feedback.append(el("strong", null, good ? tx("Correcto") : tx("No es correcto")));
    if (data.reference) feedback.append(renderMathIn(el("p", null, tx("Referencia: {0}", data.reference))));
    if (data.explanation) feedback.append(renderMathIn(el("p", null, data.explanation)));
  };
  if (q.kind === "mc" || q.kind === "tf") {
    const choices = q.kind === "mc" ? q.options.map((o, i) => [i, `${"ABCD"[i]}. ${o}`]) : [[true, tx("Verdadero")], [false, tx("Falso")]];
    const row = el("div", "reader-choices");
    for (const [value, label] of choices) {
      const button = renderMathIn(el("button", "option", label));
      button.type = "button";
      button.addEventListener("click", async () => {
        for (const b of row.querySelectorAll("button")) b.disabled = true;
        try {
          const data = await practicePost("/api/practice/check", { question_id: q.id, choice: value });
          const right = q.kind === "mc" ? data.correct : data.truth;
          [...row.children].forEach((b, i) => {
            if (choices[i][0] === right) b.classList.add("is-correct");
            else if (choices[i][0] === value) b.classList.add("is-wrong");
          });
          show(data.result === "correct", data);
        } catch (error) {
          feedback.replaceChildren(el("p", "error", error.message));
        }
      });
      row.append(button);
    }
    box.append(row);
  } else {
    const area = el("textarea", "answer-box");
    area.rows = 3;
    area.placeholder = tx("Tu respuesta…");
    area.setAttribute("aria-label", tx("Tu respuesta"));
    const reveal = el("button", "btn", tx("Mostrar respuesta"));
    reveal.type = "button";
    reveal.addEventListener("click", async () => {
      reveal.disabled = true;
      try {
        const data = await practicePost("/api/practice/reveal", { question_id: q.id });
        show(null, data);
        const grade = el("div", "row");
        grade.append(el("span", "small", tx("¿Cómo te fue?")));
        for (const [result, label] of [["correct", tx("La sabía")], ["partial", tx("A medias")], ["wrong", tx("No la sabía")]]) {
          const b = el("button", "btn", label);
          b.type = "button";
          b.addEventListener("click", async () => {
            await practicePost("/api/practice/self", { question_id: q.id, answer: area.value, result });
            grade.replaceChildren(el("span", "small muted", tx("Guardado: {0}.", label.toLowerCase())));
          });
          grade.append(b);
        }
        feedback.append(grade);
      } catch (error) {
        reveal.disabled = false;
        feedback.replaceChildren(el("p", "error", error.message));
      }
    });
    box.append(area, reveal);
  }
  box.append(feedback);
  return box;
}

// --- wiring ----------------------------------------------------------------------------------

function bindReader() {
  document.addEventListener("selectionchange", () => {
    if (currentPage() !== "leer") return;
    clearTimeout(reader.selectionTimer);
    reader.selectionTimer = setTimeout(updateToolbar, 150);
  });
  window.addEventListener("scroll", () => ($("reader-toolbar").hidden = true), { passive: true });
  for (const [id, run] of [["reader-explain", explainPassage], ["reader-quiz", quizPassage]]) {
    const button = $(id);
    button.addEventListener("mousedown", (event) => event.preventDefault()); // keep the selection
    button.addEventListener("click", () => {
      const passage = reader.passage || selectedPassage();
      $("reader-toolbar").hidden = true;
      if (passage) run(passage);
    });
  }
  $("reader-panel-close").addEventListener("click", () => ($("reader-panel").hidden = true));
  $("reader-find").addEventListener("submit", readerFind);
  $("reader-find-input").addEventListener("keydown", (event) => {
    if (event.key === "Escape") readerFindClear();
  });
  $("reader-back").addEventListener("click", () => history.back());
  $("reader-read").addEventListener("click", async () => {
    const read = !reader.view.read;
    const response = await postJSON("/api/plans/read", { doc_path: reader.path, read });
    if (!response.ok) return;
    reader.view.read = read;
    renderReadMark();
  });
}

let readerBound = false;

pageHooks.leer = () => {
  if (!readerBound) {
    readerBound = true;
    bindReader();
  }
  const path = location.hash.match(/[?&]path=([^&]*)/);
  const line = location.hash.match(/[?&]line=(\d+)/);
  if (path) openNote(decodeURIComponent(path[1]), line ? Number(line[1]) : null);
};
