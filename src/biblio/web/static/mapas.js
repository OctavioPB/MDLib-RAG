// Biblioteca — page Mapas (ADR-022): concept maps drawn from the server's layout.
// Uses the helpers of app.js ($, el, store, streamEvents, postJSON, errorText, showError, app).
// Library and model text is only ever inserted with textContent, never as HTML.
"use strict";

const mapState = {
  trail: [], // [{id, topic, map, fragments}] — expanded maps, the last one shown (id: ADR-024)
  selected: null,
  view: "map",
  zoom: { x: 0, y: 0, k: 1 },
  busy: false,
  controller: null,
  moved: false, // the user panned or zoomed: stop re-fitting the growing map
};

const current = () => mapState.trail[mapState.trail.length - 1] || null;
const svgNs = () => $("map-svg-root").namespaceURI; // no namespace URL written in this file
const svgEl = (tag, attrs = {}) => {
  const node = document.createElementNS(svgNs(), tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
};

function nodeSize(node) {
  const font = node.level === 0 ? 17 : node.level === 1 ? 15 : 13;
  const width = Math.min(260, Math.max(90, node.label.length * font * 0.56 + 28));
  return { width, height: node.level === 0 ? 52 : node.level === 1 ? 44 : 36, font };
}

// --- building a map ------------------------------------------------------------------------

async function createMap(topic, expand = false) {
  if (mapState.busy) return;
  showError("map-error", "");
  await loadStatus();
  if (!app.llmRunning) {
    showError("map-error", "El modelo de lenguaje no está encendido. Inícialo con scripts/serve-llm.ps1.");
    return;
  }
  const entry = { id: null, topic, map: null, fragments: [] };
  if (!expand) mapState.trail = [];
  mapState.trail.push(entry);
  mapState.selected = null;
  mapState.moved = false;
  mapState.busy = true;
  mapState.controller = new AbortController();
  $("map-topic").value = topic;
  setStatus("Buscando en tus notas…");
  renderAll();
  try {
    await streamEvents(
      "/api/maps",
      { topic },
      (name, data) => {
        if (name === "start") {
          entry.fragments = data.fragments;
          setStatus(`Leyendo ${data.fragments.length} fragmentos de tus notas…`);
        } else if (name === "map") {
          entry.map = data.map;
          setStatus(`Fragmento ${data.processed} de ${entry.fragments.length} · ${data.map.nodes.length - 1} conceptos`);
          renderAll();
          if (!mapState.moved) fitView();
        } else if (name === "done") {
          entry.id = data.map_id;
          loadMapHistory();
          const left = data.omitted ? ` · ${data.omitted} conceptos sin relación clara se dejaron fuera` : "";
          const skipped = data.skipped ? ` · ${data.skipped} fragmentos no se pudieron leer` : "";
          setStatus(`${entry.map.nodes.length - 1} conceptos de ${entry.fragments.length} fragmentos${left}${skipped}`);
        }
      },
      mapState.controller.signal,
    );
  } catch (error) {
    if (error.name !== "AbortError") {
      showError("map-error", error.message);
      if (!entry.map) mapState.trail.pop();
    }
    setStatus("");
  } finally {
    mapState.busy = false;
    renderAll();
  }
}

function setStatus(text) {
  $("map-status").textContent = text;
}

// --- drawing -------------------------------------------------------------------------------

function edgePath(a, b) {
  // a gentle curve, bending away from the centre
  const mx = (a.x + b.x) / 2;
  const my = (a.y + b.y) / 2;
  const bend = 0.12;
  return `M${a.x},${a.y} Q${mx - (b.y - a.y) * bend},${my + (b.x - a.x) * bend} ${b.x},${b.y}`;
}

function drawMap() {
  const entry = current();
  const viewport = $("map-viewport");
  viewport.replaceChildren();
  if (!entry || !entry.map) return;
  const byId = new Map(entry.map.nodes.map((n) => [n.id, n]));
  const edges = svgEl("g", { class: "map-edges" });
  const labels = svgEl("g", { class: "map-edge-labels" });
  for (const e of entry.map.edges) {
    const a = byId.get(e.source);
    const b = byId.get(e.target);
    if (!a || !b) continue;
    const touches = mapState.selected && (e.source === mapState.selected || e.target === mapState.selected);
    edges.append(svgEl("path", { d: edgePath(a, b), class: `map-edge ${e.kind}${touches ? " active" : ""}` }));
    if (e.label && (e.kind === "tree" || touches)) {
      const text = svgEl("text", { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, class: "map-edge-label" });
      text.textContent = e.label;
      labels.append(text);
    }
  }
  const nodes = svgEl("g", { class: "map-nodes" });
  for (const n of entry.map.nodes) {
    const { width, height, font } = nodeSize(n);
    const g = svgEl("g", {
      class: `map-node level-${n.level}${n.id === mapState.selected ? " selected" : ""}`,
      transform: `translate(${n.x},${n.y})`,
      tabindex: 0,
      role: "button",
      "aria-label": `${n.label}, ${n.sources.length} ${n.sources.length === 1 ? "fuente" : "fuentes"}`,
    });
    g.append(svgEl("rect", { x: -width / 2, y: -height / 2, width, height, rx: height / 2 }));
    const text = svgEl("text", { y: 1, "font-size": font, "dominant-baseline": "middle", "text-anchor": "middle" });
    text.textContent = n.label.length > 34 ? `${n.label.slice(0, 33)}…` : n.label;
    g.append(text);
    const select = () => {
      mapState.selected = n.id;
      renderAll();
    };
    g.addEventListener("click", select);
    g.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        select();
      }
    });
    nodes.append(g);
  }
  viewport.append(edges, labels, nodes);
  applyZoom();
  // halo behind edge labels so they stay readable over lines (needs the drawing on screen)
  if ($("map-stage").hidden) return;
  for (const text of [...labels.children]) {
    const box = text.getBBox();
    labels.insertBefore(svgEl("rect", { x: box.x - 4, y: box.y - 1, width: box.width + 8, height: box.height + 2, rx: 4, class: "map-edge-halo" }), text);
  }
}

function applyZoom() {
  const { x, y, k } = mapState.zoom;
  $("map-viewport").setAttribute("transform", `translate(${x},${y}) scale(${k})`);
}

function fitView() {
  const entry = current();
  const stage = $("map-stage");
  if (!entry || !entry.map) return;
  const xs = entry.map.nodes.map((n) => n.x);
  const ys = entry.map.nodes.map((n) => n.y);
  const pad = 160;
  const w = Math.max(...xs) - Math.min(...xs) + pad * 2;
  const h = Math.max(...ys) - Math.min(...ys) + pad;
  const box = stage.getBoundingClientRect();
  if (!box.width || !box.height) return; // hidden: fitted when shown
  const k = Math.min(1.2, Math.max(0.3, Math.min(box.width / w, box.height / h)));
  const cx = (Math.max(...xs) + Math.min(...xs)) / 2;
  const cy = (Math.max(...ys) + Math.min(...ys)) / 2;
  mapState.zoom = { k, x: box.width / 2 - cx * k, y: box.height / 2 - cy * k };
  applyZoom();
}

function zoomBy(factor, cx, cy) {
  const box = $("map-stage").getBoundingClientRect();
  const px = cx ?? box.width / 2;
  const py = cy ?? box.height / 2;
  const z = mapState.zoom;
  const k = Math.min(3, Math.max(0.2, z.k * factor));
  mapState.moved = true;
  mapState.zoom = { k, x: px - ((px - z.x) * k) / z.k, y: py - ((py - z.y) * k) / z.k };
  applyZoom();
}

// --- outline and panel ---------------------------------------------------------------------

function sentence(map, e) {
  const names = new Map(map.nodes.map((n) => [n.id, n.label]));
  return `${names.get(e.source)} ${e.label} ${names.get(e.target)}`;
}

function cites(node) {
  return node.sources.map((s) => `[${s}]`).join("");
}

function renderOutline() {
  const entry = current();
  const box = $("map-outline");
  if (!entry || !entry.map) {
    box.replaceChildren();
    return;
  }
  const map = entry.map;
  const tree = new Map(map.edges.filter((e) => e.kind === "tree").map((e) => [[e.source, e.target].sort().join("|"), e]));
  const relation = (a, b) => {
    const e = tree.get([a, b].sort().join("|"));
    return e && e.label ? ` — ${sentence(map, e)}` : "";
  };
  const list = el("ul");
  for (const n1 of map.nodes.filter((n) => n.level === 1)) {
    const li = el("li");
    const name = el("button", "link-button", n1.label);
    name.type = "button";
    name.addEventListener("click", () => { mapState.selected = n1.id; renderAll(); });
    li.append(name, document.createTextNode(` ${cites(n1)}${relation("root", n1.id)}`));
    const kids = map.nodes.filter((n) => n.parent === n1.id);
    if (kids.length) {
      const sub = el("ul");
      for (const n2 of kids) {
        const item = el("li");
        const child = el("button", "link-button", n2.label);
        child.type = "button";
        child.addEventListener("click", () => { mapState.selected = n2.id; renderAll(); });
        item.append(child, document.createTextNode(` ${cites(n2)}${relation(n1.id, n2.id)}`));
        sub.append(item);
      }
      li.append(sub);
    }
    list.append(li);
  }
  const cross = map.edges.filter((e) => e.kind === "cross");
  const nodes = [el("h2", null, entry.topic), list];
  if (cross.length) {
    nodes.push(el("h3", null, "Otras relaciones"));
    const other = el("ul");
    for (const e of cross) other.append(el("li", null, sentence(map, e)));
    nodes.push(other);
  }
  box.replaceChildren(...nodes);
}

function renderPanel() {
  const entry = current();
  const node = entry && entry.map && entry.map.nodes.find((n) => n.id === mapState.selected);
  $("map-panel-empty").hidden = Boolean(node);
  const view = $("map-panel-view");
  view.hidden = !node;
  if (!node) return;
  const parts = [el("span", "eyebrow", node.level === 0 ? "Tema" : "Concepto"), el("h2", null, node.label)];
  const related = entry.map.edges.filter((e) => e.label && (e.source === node.id || e.target === node.id));
  if (related.length) {
    parts.push(el("h3", null, "Relaciones"));
    const list = el("ul", "map-relations");
    for (const e of related) list.append(el("li", null, sentence(entry.map, e)));
    parts.push(list);
  }
  const actions = el("div", "row");
  if (node.level > 0) {
    const expand = el("button", "btn-primary", "Expandir");
    expand.type = "button";
    expand.disabled = mapState.busy;
    expand.title = "Crear un mapa de este concepto";
    expand.addEventListener("click", () => createMap(node.label, true));
    actions.append(expand);
  }
  const explain = el("a", "btn", "Explicar");
  explain.href = `#/conversar?q=${encodeURIComponent(`¿Qué es ${node.label}?`)}`;
  const practice = el("a", "btn", "Practicar");
  practice.href = `#/practicar?tema=${encodeURIComponent(node.label)}`;
  actions.append(explain, practice);
  parts.push(actions);
  parts.push(el("h3", null, `Aparece en ${node.sources.length} ${node.sources.length === 1 ? "fragmento" : "fragmentos"}`));
  const fragments = new Map(entry.fragments.map((f) => [f.number, f]));
  for (const number of node.sources) {
    const f = fragments.get(number);
    if (!f) continue;
    const item = el("details", "map-source");
    item.append(el("summary", null, `[${number}] ${f.sections.at(-1) || f.guide} · ${f.guide}`));
    item.append(el("div", "source-text", f.text));
    if (f.open_url) {
      const open = el("a", "btn-soft", "Abrir en Obsidian");
      open.href = f.open_url;
      item.append(open);
    }
    parts.push(item);
  }
  view.replaceChildren(...parts);
}

function renderTrail() {
  const trail = $("map-trail");
  trail.hidden = mapState.trail.length < 2;
  const items = [];
  mapState.trail.forEach((entry, i) => {
    if (i) items.push(el("span", "muted", "›"));
    const last = i === mapState.trail.length - 1;
    const button = el("button", `link-button${last ? " current" : ""}`, entry.topic);
    button.type = "button";
    button.disabled = last || mapState.busy;
    button.addEventListener("click", () => {
      mapState.trail = mapState.trail.slice(0, i + 1);
      mapState.selected = null;
      $("map-topic").value = entry.topic;
      renderAll();
      fitView();
    });
    items.push(button);
  });
  trail.replaceChildren(...items);
}

function renderAll() {
  const entry = current();
  const hasMap = Boolean(entry && entry.map);
  $("map-empty").hidden = Boolean(entry);
  $("map-stage").hidden = !entry || mapState.view !== "map";
  $("map-outline").hidden = !entry || mapState.view !== "outline";
  $("map-create").disabled = mapState.busy;
  $("map-save").disabled = !hasMap || !entry.id || mapState.busy;
  $("map-delete").disabled = !hasMap || !entry.id || mapState.busy;
  markActiveMap();
  $("map-svg").disabled = !hasMap || mapState.busy;
  renderTrail();
  drawMap();
  renderOutline();
  renderPanel();
}

// --- the history of maps (ADR-024) --------------------------------------------------------

async function loadMapHistory() {
  const query = $("map-history-search").value.trim();
  let maps = [];
  try {
    const url = `/api/maps/history${query ? `?q=${encodeURIComponent(query)}` : ""}`;
    maps = (await (await fetch(url)).json()).maps;
  } catch {
    /* keep the list empty */
  }
  const items = maps.map((m) => {
    const li = el("li");
    const button = el("button", "history-item");
    button.type = "button";
    button.dataset.id = String(m.id);
    button.append(el("strong", null, m.topic));
    button.append(el("span", null, `${formatWhen(m.created)} · ${m.concepts} conceptos`));
    if (m.snippet) button.append(el("span", "history-snippet", m.snippet));
    button.addEventListener("click", () => openStoredMap(m.id));
    li.append(button);
    return li;
  });
  $("map-history-list").replaceChildren(...items);
  $("map-history-empty").hidden = items.length > 0;
  $("map-history-empty").textContent = query
    ? "Ningún mapa trata de eso."
    : "Los mapas que crees aparecerán aquí.";
  markActiveMap();
}

function markActiveMap() {
  const entry = current();
  for (const item of document.querySelectorAll("#map-history-list .history-item")) {
    if (entry && Number(item.dataset.id) === entry.id) item.setAttribute("aria-current", "true");
    else item.removeAttribute("aria-current");
  }
}

async function openStoredMap(id) {
  if (mapState.busy) return;
  showError("map-error", "");
  const response = await fetch(`/api/maps/history/${id}`);
  if (!response.ok) {
    loadMapHistory();
    return;
  }
  const data = await response.json();
  mapState.trail = [{ id: data.id, topic: data.topic, map: data.map, fragments: data.fragments }];
  mapState.selected = null;
  mapState.moved = false;
  $("map-topic").value = data.topic;
  setStatus(`${data.concepts} conceptos · creado el ${formatWhen(data.created)}`);
  renderAll();
  fitView();
}

async function deleteMap() {
  const entry = current();
  if (!entry || !entry.id || !confirm(`¿Borrar el mapa «${entry.topic}»? No se puede deshacer.`)) return;
  try {
    const response = await postJSON("/api/maps/delete", { id: entry.id });
    if (!response.ok) throw new Error(await errorText(response));
    mapState.trail.pop();
    mapState.selected = null;
    setStatus("Mapa borrado.");
    renderAll();
    if (current()) fitView();
    loadMapHistory();
  } catch (error) {
    showError("map-error", error.message);
  }
}

// --- saving --------------------------------------------------------------------------------

async function saveMap() {
  const entry = current();
  try {
    const response = await postJSON("/api/maps/save", { id: entry.id });
    if (!response.ok) throw new Error(await errorText(response));
    const data = await response.json();
    setStatus(`Guardado en ${data.markdown} y ${data.canvas.split(/[\\/]/).pop()} (ábrelo en Obsidian copiándolo a tu bóveda).`);
  } catch (error) {
    showError("map-error", error.message);
  }
}

function downloadSvg() {
  const entry = current();
  const source = $("map-svg-root");
  const copy = source.cloneNode(true);
  // inline the computed colours: the CSS variables do not travel with the file
  const originals = source.querySelectorAll("*");
  copy.querySelectorAll("*").forEach((node, i) => {
    const style = getComputedStyle(originals[i]);
    for (const prop of ["fill", "stroke", "stroke-width", "stroke-dasharray", "font-size", "font-family", "font-weight", "opacity"]) {
      node.style.setProperty(prop, style.getPropertyValue(prop));
    }
  });
  const box = $("map-viewport").getBBox();
  copy.setAttribute("viewBox", `${box.x - 20} ${box.y - 20} ${box.width + 40} ${box.height + 40}`);
  copy.querySelector("#map-viewport").removeAttribute("transform");
  copy.setAttribute("xmlns", svgNs());
  const blob = new Blob([new XMLSerializer().serializeToString(copy)], { type: "image/svg+xml" });
  const link = el("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${entry.topic.replace(/[^\p{L}\p{N}]+/gu, "-").toLowerCase() || "mapa"}.svg`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

// --- wiring --------------------------------------------------------------------------------

function bindMapas() {
  $("map-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const topic = $("map-topic").value.trim();
    if (topic) createMap(topic);
  });
  for (const radio of document.querySelectorAll('input[name="map-view"]')) {
    radio.addEventListener("change", () => {
      mapState.view = radio.value;
      renderAll();
      if (mapState.view === "map") fitView();
    });
  }
  $("map-zoom-in").addEventListener("click", () => zoomBy(1.25));
  $("map-zoom-out").addEventListener("click", () => zoomBy(0.8));
  $("map-fit").addEventListener("click", fitView);
  $("map-save").addEventListener("click", saveMap);
  $("map-svg").addEventListener("click", downloadSvg);
  $("map-delete").addEventListener("click", deleteMap);
  let timer = null;
  $("map-history-search").addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(loadMapHistory, 250);
  });

  const stage = $("map-stage");
  stage.addEventListener("wheel", (event) => {
    event.preventDefault();
    const box = stage.getBoundingClientRect();
    zoomBy(event.deltaY < 0 ? 1.1 : 0.9, event.clientX - box.left, event.clientY - box.top);
  }, { passive: false });
  let drag = null;
  stage.addEventListener("pointerdown", (event) => {
    if (event.target.closest(".map-node")) return;
    drag = { x: event.clientX, y: event.clientY, start: { ...mapState.zoom } };
    stage.setPointerCapture(event.pointerId);
  });
  stage.addEventListener("pointermove", (event) => {
    if (!drag) return;
    mapState.zoom = { ...drag.start, x: drag.start.x + event.clientX - drag.x, y: drag.start.y + event.clientY - drag.y };
    mapState.moved = true;
    applyZoom();
  });
  stage.addEventListener("pointerup", () => (drag = null));
}

let mapasStarted = false;

pageHooks.mapas = () => {
  if (!mapasStarted) {
    mapasStarted = true;
    bindMapas();
    loadMapHistory();
    renderAll();
  }
  if (current() && mapState.view === "map") fitView();
};
