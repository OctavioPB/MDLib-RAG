// Biblioteca — page Mapas (ADR-022, ADR-036): concept maps and diagrams drawn from the server's layout.
// Uses the helpers of app.js ($, el, store, streamEvents, postJSON, errorText, showError, app).
// Library and model text is only ever inserted with textContent, never as HTML.
"use strict";

const mapState = {
  trail: [], // [{id, kind, topic, map, fragments}] — expanded maps, the last one shown (id: ADR-024)
  selected: null,
  view: "map",
  zoom: { x: 0, y: 0, k: 1 },
  busy: false,
  controller: null,
  moved: false, // the user panned or zoomed: stop re-fitting the growing map
  mastery: false, // colour the nodes by how well they are remembered today (ADR-026)
  kind: "concepts", // the type chosen for the next map (ADR-036)
};

const MAP_KINDS = {
  concepts: { label: "Conceptos", unit: ["concepto", "conceptos"], aria: "Mapa de conceptos" },
  novak: { label: "Mapa conceptual", unit: ["concepto", "conceptos"], aria: "Mapa conceptual" },
  flow: { label: "Flujo", unit: ["paso", "pasos"], aria: "Diagrama de flujo", none: "un proceso" },
  sequence: { label: "Secuencia", unit: ["mensaje", "mensajes"], aria: "Diagrama de secuencia", none: "una interacción entre participantes" },
  fishbone: { label: "Causa-efecto", unit: ["causa", "causas"], aria: "Diagrama de causa-efecto", none: "causas" },
};

const current = () => mapState.trail[mapState.trail.length - 1] || null;
const isDiagram = (entry) => Boolean(entry) && ["flow", "sequence", "fishbone"].includes(entry.kind);
const mapUnits = (kind, n) => `${n} ${MAP_KINDS[kind || "concepts"].unit[n === 1 ? 0 : 1]}`;
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

/** Arrowheads for the Novak map and the diagrams; inside the viewport so a saved SVG keeps them. */
function arrowDefs() {
  const defs = svgEl("defs");
  for (const [id, cls] of [["map-arrow", "map-arrowhead"], ["map-arrow-strong", "map-arrowhead strong"]]) {
    const marker = svgEl("marker", { id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 5, markerHeight: 5, orient: "auto-start-reverse" });
    marker.append(svgEl("path", { d: "M0,0 L10,5 L0,10 z", class: cls }));
    defs.append(marker);
  }
  return defs;
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
  const kind = expand && current() ? current().kind : mapState.kind;
  const entry = { id: null, kind, topic, map: null, fragments: [] };
  if (!expand) mapState.trail = [];
  mapState.trail.push(entry);
  mapState.selected = null;
  mapState.moved = false;
  mapState.busy = true;
  mapState.controller = new AbortController();
  $("map-topic").value = topic;
  $("map-suggestion").hidden = true;
  setStatus("Buscando en tus notas…");
  renderAll();
  const stages = { grouping: "Agrupando las causas en categorías…" };
  try {
    await streamEvents(
      "/api/maps",
      { topic, kind },
      (name, data) => {
        if (name === "start") {
          entry.fragments = data.fragments;
          setStatus(`Leyendo ${data.fragments.length} fragmentos de tus notas…`);
        } else if (name === "map") {
          entry.map = data.map;
          setStatus(`Fragmento ${data.processed} de ${entry.fragments.length} · ${mapUnits(kind, data.map.nodes.length - 1)}`);
          renderAll();
          if (!mapState.moved) fitView();
        } else if (name === "diagram") {
          entry.map = data.map;
          setStatus(stages[data.stage] || `Fragmento ${data.processed} de ${entry.fragments.length} · ${mapUnits(kind, data.map.count)}`);
          renderAll();
          if (!mapState.moved) fitView();
        } else if (name === "done") {
          if (isDiagram(entry) && !data.map_id) {
            mapState.trail.pop(); // nothing of that kind in the notes: nothing to keep
            setStatus(`Tus notas no describen ${MAP_KINDS[kind].none} sobre «${topic}». Prueba otro tipo de diagrama.`);
            return;
          }
          entry.id = data.map_id;
          if (!isDiagram(entry)) applyPositions(entry, {});
          loadMapHistory();
          const count = isDiagram(entry) ? entry.map.count : entry.map.nodes.length - 1;
          const left = data.omitted ? ` · ${data.omitted} ${isDiagram(entry) ? "piezas" : "conceptos sin relación clara"} se dejaron fuera` : "";
          const skipped = data.skipped ? ` · ${data.skipped} fragmentos no se pudieron leer` : "";
          setStatus(`${mapUnits(kind, count)} de ${entry.fragments.length} fragmentos${left}${skipped}`);
        } else if (name === "gap") {
          entry.gap = data; // a thin map: the topic is pending in Guías (ADR-029)
          renderGap();
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
    if (current()) fitView();
  }
}

async function suggestKind() {
  const topic = $("map-topic").value.trim();
  const box = $("map-suggestion");
  if (!topic) {
    showError("map-error", "Escribe primero el tema.");
    return;
  }
  if (mapState.busy) return;
  showError("map-error", "");
  mapState.busy = true;
  $("map-suggest").textContent = "Pensando…";
  renderAll();
  try {
    const response = await postJSON("/api/maps/suggest", { topic });
    if (!response.ok) throw new Error(await errorText(response));
    const data = await response.json();
    setMapKind(data.kind);
    box.replaceChildren(
      el("strong", null, `Sugerencia: ${MAP_KINDS[data.kind].label}. `),
      document.createTextNode(`${data.reason ? `${data.reason} ` : ""}Pulsa «Crear» para hacerlo, o elige otro tipo.`),
    );
    box.hidden = false;
  } catch (error) {
    showError("map-error", error.message);
  } finally {
    mapState.busy = false;
    $("map-suggest").textContent = "Sugerir tipo";
    renderAll();
  }
}

function setMapKind(kind) {
  mapState.kind = MAP_KINDS[kind] ? kind : "concepts";
  store.set("biblio.mapKind", mapState.kind);
  for (const radio of document.querySelectorAll('input[name="map-kind"]')) radio.checked = radio.value === mapState.kind;
}

function renderGap() {
  const entry = current();
  const box = $("map-gap");
  box.hidden = !(entry && entry.gap);
  const why = entry && entry.gap && entry.gap.absent
    ? "Tus notas no tratan este tema: los conceptos vienen de notas parecidas. "
    : "Tus notas tratan poco este tema. ";
  box.replaceChildren(...(entry && entry.gap ? [el("span", null, why), gapNote(entry.gap)] : []));
}

function setStatus(text) {
  $("map-status").textContent = text;
}

// --- drawing a concept map -----------------------------------------------------------------

/** Where the line from `a` towards `b` leaves the box of `a` (the Novak map's straight arrows). */
function boxEdge(a, b) {
  const { width, height } = nodeSize(a);
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  if (!dx && !dy) return { x: a.x, y: a.y };
  const t = Math.min(dx ? width / 2 / Math.abs(dx) : Infinity, dy ? height / 2 / Math.abs(dy) : Infinity);
  return { x: a.x + dx * t, y: a.y + dy * t };
}

function edgePath(a, b) {
  if (current() && current().kind === "novak") {
    const from = boxEdge(a, b);
    const to = boxEdge(b, a);
    return `M${from.x},${from.y} L${to.x},${to.y}`;
  }
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
  mapState.drawn = { nodes: new Map(), edges: [] };
  if (!entry || !entry.map) return;
  $("map-svg-root").setAttribute("aria-label", MAP_KINDS[entry.kind || "concepts"].aria);
  if (isDiagram(entry)) {
    drawDiagram(entry);
    return;
  }
  const novak = entry.kind === "novak";
  const byId = new Map(entry.map.nodes.map((n) => [n.id, n]));
  const edges = svgEl("g", { class: "map-edges" });
  const labels = svgEl("g", { class: "map-edge-labels" });
  for (const e of entry.map.edges) {
    const a = byId.get(e.source);
    const b = byId.get(e.target);
    if (!a || !b) continue;
    const touches = mapState.selected && (e.source === mapState.selected || e.target === mapState.selected);
    const path = svgEl("path", { d: edgePath(a, b), class: `map-edge ${e.kind}${touches ? " active" : ""}` });
    if (novak) path.setAttribute("marker-end", "url(#map-arrow)"); // a proposition reads along the arrow
    edges.append(path);
    const drawn = { a, b, path, text: null, halo: null };
    if (e.label && (e.kind === "tree" || touches || novak)) {
      drawn.text = svgEl("text", { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, class: "map-edge-label" });
      drawn.text.textContent = e.label;
      labels.append(drawn.text);
    }
    mapState.drawn.edges.push(drawn);
  }
  const nodes = svgEl("g", { class: "map-nodes" });
  for (const n of entry.map.nodes) {
    const { width, height, font } = nodeSize(n);
    const known = mapState.mastery && entry.mastery && entry.mastery.nodes[n.id];
    const g = svgEl("g", {
      class: `map-node level-${n.level}${known ? ` m-${known.state}` : ""}${n.id === mapState.selected ? " selected" : ""}`,
      transform: `translate(${n.x},${n.y})`,
      tabindex: 0,
      role: "button",
      "aria-label": `${n.label}, ${n.sources.length} ${n.sources.length === 1 ? "fuente" : "fuentes"}. Flechas para moverlo.`,
    });
    g.append(svgEl("rect", { x: -width / 2, y: -height / 2, width, height, rx: novak ? 10 : height / 2 }));
    const text = svgEl("text", { y: 1, "font-size": font, "dominant-baseline": "middle", "text-anchor": "middle" });
    text.textContent = n.label.length > 34 ? `${n.label.slice(0, 33)}…` : n.label;
    g.append(text);
    // a click selects, a drag moves the node (decided on release: see bindNodeDragging)
    g.addEventListener("pointerdown", (event) => startNodeDrag(event, n));
    g.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectNode(n.id);
        return;
      }
      const step = event.shiftKey ? 50 : 10;
      const delta = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] }[event.key];
      if (!delta || mapState.busy) return;
      event.preventDefault(); // move the node, not the page
      moveNode(n, n.x + delta[0], n.y + delta[1]);
      rememberPosition(n);
    });
    nodes.append(g);
    mapState.drawn.nodes.set(n.id, g);
  }
  viewport.append(arrowDefs(), edges, labels, nodes);
  applyZoom();
  addLabelHalos(labels, mapState.drawn.edges.map((d) => d.text), (i, halo) => { mapState.drawn.edges.filter((d) => d.text)[i].halo = halo; });
}

/** A halo behind each label so it stays readable over lines (needs the drawing on screen). */
function addLabelHalos(group, texts, keep = () => {}) {
  if ($("map-stage").hidden) return;
  texts.filter(Boolean).forEach((text, i) => {
    const box = text.getBBox();
    const halo = svgEl("rect", { x: box.x - 4, y: box.y - 1, width: box.width + 8, height: box.height + 2, rx: 4, class: "map-edge-halo" });
    group.insertBefore(halo, text);
    keep(i, halo);
  });
}

// --- drawing a flowchart, a sequence or a fishbone (ADR-036) --------------------------------

function drawDiagram(entry) {
  const drawing = entry.map.drawing;
  const viewport = $("map-viewport");
  const frames = svgEl("g", { class: "dg-frames" });
  for (const f of drawing.frames) {
    const g = svgEl("g", { class: "dg-frame" });
    g.append(svgEl("rect", { x: f.x, y: f.y, width: f.w, height: f.h, rx: 6 }));
    const tabWidth = Math.min(f.w, f.label.length * 7 + 46);
    g.append(svgEl("path", { d: `M${f.x},${f.y + 22} H${f.x + tabWidth - 10} L${f.x + tabWidth},${f.y + 12} V${f.y}`, class: "dg-frame-tab" }));
    const label = svgEl("text", { x: f.x + 8, y: f.y + 15, class: "dg-frame-label" });
    label.textContent = `si ${f.label.replace(/^si\s+/i, "")}`;
    g.append(label);
    frames.append(g);
  }
  const lines = svgEl("g", { class: "dg-lines" });
  const labels = svgEl("g", { class: "dg-labels" });
  const texts = [];
  for (const line of drawing.lines) {
    const d = line.points.map((p, i) => `${i ? "L" : "M"}${p[0]},${p[1]}`).join(" ");
    const selected = line.id && line.id === mapState.selected;
    const path = svgEl("path", { d, class: `dg-line ${line.role}${line.dashed ? " dashed" : ""}${selected ? " selected" : ""}` });
    if (line.arrow) path.setAttribute("marker-end", `url(#${line.role === "spine" ? "map-arrow-strong" : "map-arrow"})`);
    lines.append(path);
    if (line.id) { // a message: the whole line and its label are clickable
      const hit = svgEl("path", { d, class: "dg-hit dg-pick", tabindex: 0, role: "button", "aria-label": `Mensaje: ${line.label}` });
      bindPick(hit, line.id);
      lines.append(hit);
    }
    if (line.label) {
      const text = svgEl("text", {
        x: line.label_at[0],
        y: line.label_at[1],
        "text-anchor": line.anchor,
        class: `dg-label${line.id ? " dg-pick" : ""}${line.role === "flow" ? " branch" : ""}`,
      });
      text.textContent = line.label;
      if (line.id) bindPick(text, line.id);
      labels.append(text);
      texts.push(text);
    }
  }
  const boxes = svgEl("g", { class: "dg-boxes" });
  const quiet = new Set(["start", "end"]);
  for (const b of drawing.boxes) {
    const g = svgEl("g", {
      class: `dg-box ${b.role}${b.id === mapState.selected ? " selected" : ""}${quiet.has(b.role) ? "" : " dg-pick"}`,
      transform: `translate(${b.x},${b.y})`,
    });
    if (b.role === "decision") {
      g.append(svgEl("polygon", { points: `${b.w / 2},0 ${b.w},${b.h / 2} ${b.w / 2},${b.h} 0,${b.h / 2}` }));
    } else {
      const rx = quiet.has(b.role) ? b.h / 2 : b.role === "effect" ? 14 : 8;
      g.append(svgEl("rect", { width: b.w, height: b.h, rx }));
    }
    const lineHeight = b.role === "cause" ? 18 : 20;
    const text = svgEl("text", { x: b.w / 2, y: b.h / 2 - ((b.lines.length - 1) * lineHeight) / 2, "text-anchor": "middle", "dominant-baseline": "middle" });
    b.lines.forEach((part, i) => {
      const span = svgEl("tspan", { x: b.w / 2, dy: i ? lineHeight : 0 });
      span.textContent = part;
      text.append(span);
    });
    g.append(text);
    if (!quiet.has(b.role)) {
      g.setAttribute("tabindex", 0);
      g.setAttribute("role", "button");
      g.setAttribute("aria-label", b.label);
      bindPick(g, b.id);
    }
    boxes.append(g);
  }
  viewport.append(arrowDefs(), frames, lines, labels, boxes);
  applyZoom();
  addLabelHalos(labels, texts);
}

function bindPick(node, id) {
  node.addEventListener("click", () => selectNode(id));
  node.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      selectNode(id);
    }
  });
}

/** «Concepto»: show or hide the panel of the selected concept; the choice is remembered. */
function setConceptPanelOpen(open) {
  $("map-panel").hidden = !open;
  $("map-panel-toggle").setAttribute("aria-expanded", String(open));
  $("map-panel-toggle").classList.toggle("active", open);
  store.set("biblio.mapPanelOpen", open);
}

function selectNode(id) {
  mapState.selected = id;
  renderAll();
}

// --- moving nodes: the relations stay, only the drawing follows -----------------------------

/** Put a node at (x, y) and bring its lines and their labels along, without redrawing. */
function moveNode(node, x, y) {
  node.x = Math.round(x);
  node.y = Math.round(y);
  const g = mapState.drawn.nodes.get(node.id);
  if (g) g.setAttribute("transform", `translate(${node.x},${node.y})`);
  for (const drawn of mapState.drawn.edges) {
    if (drawn.a !== node && drawn.b !== node) continue;
    drawn.path.setAttribute("d", edgePath(drawn.a, drawn.b));
    if (!drawn.text) continue;
    const mx = (drawn.a.x + drawn.b.x) / 2;
    const my = (drawn.a.y + drawn.b.y) / 2;
    drawn.text.setAttribute("x", mx);
    drawn.text.setAttribute("y", my);
    if (drawn.halo) {
      drawn.halo.setAttribute("x", mx - Number(drawn.halo.getAttribute("width")) / 2);
      drawn.halo.setAttribute("y", my - Number(drawn.halo.getAttribute("height")) / 2);
    }
  }
}

let layoutTimer = null;

/** Keep where the user left a node: saved with the map in the history (ADR-024). */
function rememberPosition(node) {
  const entry = current();
  entry.positions = entry.positions || {};
  entry.positions[node.id] = [node.x, node.y];
  $("map-reset").disabled = false;
  clearTimeout(layoutTimer);
  layoutTimer = setTimeout(() => saveLayout(entry), 400);
}

async function saveLayout(entry) {
  if (!entry.id) return; // still being built: kept in the page, saved once the map is stored
  try {
    const response = await postJSON("/api/maps/positions", { id: entry.id, positions: entry.positions || {} });
    if (!response.ok) throw new Error(await errorText(response));
  } catch (error) {
    showError("map-error", `No se pudo guardar la posición de los nodos: ${error.message}`);
  }
}

function resetLayout() {
  const entry = current();
  if (!entry || !entry.map || !entry.layout) return;
  for (const n of entry.map.nodes) {
    const original = entry.layout[n.id];
    if (original) [n.x, n.y] = original;
  }
  entry.positions = {};
  saveLayout(entry);
  renderAll();
  fitView();
}

/** The automatic layout of a map, kept to undo the user's moves; then their moves applied. */
function applyPositions(entry, positions) {
  entry.layout = Object.fromEntries(entry.map.nodes.map((n) => [n.id, [n.x, n.y]]));
  entry.positions = positions || {};
  for (const n of entry.map.nodes) {
    const moved = entry.positions[n.id];
    if (moved) [n.x, n.y] = moved;
  }
}

function startNodeDrag(event, node) {
  event.stopPropagation(); // the stage would start panning
  mapState.nodeDrag = { node, x: event.clientX, y: event.clientY, ox: node.x, oy: node.y, moved: false };
  $("map-stage").setPointerCapture(event.pointerId);
}

function applyZoom() {
  const { x, y, k } = mapState.zoom;
  $("map-viewport").setAttribute("transform", `translate(${x},${y}) scale(${k})`);
}

/** The area to fit: node centres with room for their boxes, or a diagram's own size. */
function mapBounds(entry) {
  if (isDiagram(entry)) {
    const d = entry.map.drawing;
    return { x0: 0, y0: 0, x1: d.width, y1: d.height, padX: 20, padY: 20 };
  }
  const xs = entry.map.nodes.map((n) => n.x);
  const ys = entry.map.nodes.map((n) => n.y);
  return { x0: Math.min(...xs), y0: Math.min(...ys), x1: Math.max(...xs), y1: Math.max(...ys), padX: 160, padY: 80 };
}

function fitView() {
  const entry = current();
  const stage = $("map-stage");
  if (!entry || !entry.map) return;
  const b = mapBounds(entry);
  if (!Number.isFinite(b.x0) || b.x1 - b.x0 + b.y1 - b.y0 < 0) return;
  const w = b.x1 - b.x0 + b.padX * 2;
  const h = b.y1 - b.y0 + b.padY * 2;
  const box = stage.getBoundingClientRect();
  if (!box.width || !box.height) return; // hidden: fitted when shown
  const k = Math.min(1.2, Math.max(0.2, Math.min(box.width / w, box.height / h)));
  const cx = (b.x0 + b.x1) / 2;
  const cy = (b.y0 + b.y1) / 2;
  const readable = Math.min(1, box.width / w);
  if ((isDiagram(entry) || entry.kind === "novak") && k < 0.75 && readable > k) {
    // a tall flow, fishbone or Novak tree: fit the width and start at the top, so the text stays readable
    mapState.zoom = { k: readable, x: box.width / 2 - cx * readable, y: b.padY * readable };
  } else {
    mapState.zoom = { k, x: box.width / 2 - cx * k, y: box.height / 2 - cy * k };
  }
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

function renderMapOutline() {
  const entry = current();
  const box = $("map-outline");
  if (!entry || !entry.map) {
    box.replaceChildren();
    return;
  }
  if (isDiagram(entry)) {
    box.replaceChildren(...diagramOutline(entry));
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
    li.append(pickButton(n1.label, n1.id), document.createTextNode(` ${cites(n1)}${relation("root", n1.id)}`));
    const kids = map.nodes.filter((n) => n.parent === n1.id);
    if (kids.length) {
      const sub = el("ul");
      for (const n2 of kids) {
        const item = el("li");
        item.append(pickButton(n2.label, n2.id), document.createTextNode(` ${cites(n2)}${relation(n1.id, n2.id)}`));
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
  if (entry.kind === "novak") nodes.push(...mermaidBlock(novakMermaid(map)));
  box.replaceChildren(...nodes);
}

function pickButton(label, id) {
  const button = el("button", "link-button", label);
  button.type = "button";
  button.addEventListener("click", () => { mapState.selected = id; renderAll(); });
  return button;
}

/** The Novak map as Mermaid text, written in the page from the map it shows (ADR-036). */
function novakMermaid(map) {
  const quoted = (text) => `"${text.replace(/"/g, "#quot;")}"`;
  const lines = ["flowchart TD", ...map.nodes.map((n) => `  ${n.id}[${quoted(n.label)}]`)];
  for (const e of map.edges) {
    const arrow = e.kind === "cross" ? "-.->" : "-->";
    lines.push(`  ${e.source} ${arrow}${e.label ? `|${quoted(e.label)}|` : ""} ${e.target}`);
  }
  return `${lines.join("\n")}\n`;
}

function mapMermaid(entry) {
  if (!entry || !entry.map) return "";
  return isDiagram(entry) ? entry.map.mermaid : entry.kind === "novak" ? novakMermaid(entry.map) : "";
}

function mermaidBlock(text) {
  const pre = el("pre", "map-mermaid");
  pre.append(el("code", null, `\`\`\`mermaid\n${text}\`\`\``));
  return [el("h3", null, "Mermaid"), el("p", "small muted", "Pega este bloque en una nota de Obsidian y lo dibuja."), pre];
}

function diagramOutline(entry) {
  const map = entry.map;
  const out = [el("h2", null, entry.topic)];
  if (entry.kind === "flow") {
    const list = el("ol");
    for (const s of map.data.versions[map.data.current].steps) {
      const li = el("li");
      li.append(pickButton(s.label, s.id), document.createTextNode(` ${cites(s)}`));
      list.append(li);
    }
    out.push(list, el("p", "small muted", "Las flechas punteadas marcan un orden que tus notas no escriben."));
  } else if (entry.kind === "sequence") {
    const version = map.data.versions[map.data.current];
    const list = el("ol");
    for (const m of version.messages) {
      const li = el("li");
      li.append(document.createTextNode(`${m.sender} → ${m.receiver}: `), pickButton(m.label, m.id));
      li.append(document.createTextNode(` [${version.source}]${m.reply ? " (respuesta)" : ""}${m.condition ? ` — si ${m.condition.replace(/^si\s+/i, "")}` : ""}`));
      list.append(li);
    }
    out.push(el("p", null, `Participantes: ${version.participants.join(", ")}`), list);
  } else {
    const causes = new Map(map.data.causes.map((c) => [c.id, c]));
    out.push(el("p", null, `Efecto: ${map.data.effect}`));
    for (const group of map.data.categories) {
      out.push(el("h3", null, group.name));
      const list = el("ul");
      for (const id of group.causes) {
        const c = causes.get(id);
        if (!c) continue;
        const li = el("li");
        li.append(pickButton(c.label, c.id), document.createTextNode(` ${cites(c)}`));
        list.append(li);
      }
      out.push(list);
    }
  }
  out.push(...mermaidBlock(map.mermaid));
  return out;
}

/** The source fragments of a piece, each with its text and «Leer». */
function sourceList(entry, numbers, title) {
  const parts = [el("h3", null, title || `Aparece en ${numbers.length} ${numbers.length === 1 ? "fragmento" : "fragmentos"}`)];
  const fragments = new Map(entry.fragments.map((f) => [f.number, f]));
  for (const number of numbers) {
    const f = fragments.get(number);
    if (!f) continue;
    const item = el("details", "map-source");
    item.append(el("summary", null, `[${number}] ${f.sections.at(-1) || f.guide} · ${f.guide}`));
    item.append(el("div", "source-text", f.text));
    const links = el("div", "row");
    links.append(readLink(f.doc_path, f.lines[0]));
    if (f.open_url) {
      const open = el("a", "btn", "Abrir en Obsidian");
      open.href = f.open_url;
      links.append(open);
    }
    item.append(links);
    parts.push(item);
  }
  return parts;
}

function renderPanel() {
  const entry = current();
  if (isDiagram(entry) && entry.map) {
    renderDiagramPanel(entry);
    return;
  }
  const node = entry && entry.map && entry.map.nodes.find((n) => n.id === mapState.selected);
  $("map-panel-empty").hidden = Boolean(node);
  const view = $("map-panel-view");
  view.hidden = !node;
  if (!node) return;
  const parts = [el("span", "eyebrow", node.level === 0 ? "Tema" : "Concepto"), el("h2", null, node.label)];
  const related = entry.map.edges.filter((e) => e.label && (e.source === node.id || e.target === node.id));
  if (related.length) {
    parts.push(el("h3", null, entry.kind === "novak" ? "Proposiciones" : "Relaciones"));
    const list = el("ul", "map-relations");
    for (const e of related) list.append(el("li", null, sentence(entry.map, e)));
    parts.push(list);
  }
  const known = entry.mastery && entry.mastery.nodes[node.id];
  if (known && node.level > 0) {
    parts.push(el("p", "mastery-line", known.recall === null
      ? "Aún no has practicado sus secciones."
      : `Lo recuerdas hoy en un ${Math.round(known.recall * 100)} % (${known.practiced} de ${known.sections.length} ${known.sections.length === 1 ? "sección practicada" : "secciones practicadas"}).`));
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
  let practice;
  if (known && known.sections.length && node.level > 0) {
    // the node's own sections, not a new search on its name (ADR-026)
    practice = el("button", "btn", "Practicar esto");
    practice.type = "button";
    practice.title = "Preguntas sobre las secciones donde aparece este concepto";
    practice.addEventListener("click", () => {
      app.pendingPractice = {
        scope: { kind: "sections", value: JSON.stringify(known.sections), label: node.label },
        count: Math.min(known.sections.length, 10),
      };
      location.hash = "#/practicar";
    });
  } else {
    practice = el("a", "btn", "Practicar");
    practice.href = `#/practicar?tema=${encodeURIComponent(node.label)}`;
  }
  const feynmanLink = el("a", "btn", "Explícamelo tú");
  feynmanLink.href = `#/practicar?explicar=${encodeURIComponent(node.label)}`;
  feynmanLink.title = "Explícalo con tus palabras y lo comparo con tus notas";
  actions.append(explain, practice, feynmanLink);
  parts.push(actions);
  parts.push(...sourceList(entry, node.sources));
  view.replaceChildren(...parts);
}

// --- the panel of a diagram's piece (ADR-036) -----------------------------------------------

function quoteFigure(entry, quote, source) {
  const fragment = entry.fragments.find((f) => f.number === source);
  const figure = el("figure", "map-quote");
  figure.append(el("blockquote", null, `«${quote}»`));
  if (fragment) figure.append(el("figcaption", null, `[${source}] ${fragment.citation}`));
  return figure;
}

function pieceActions(entry, label, source) {
  const row = el("div", "row");
  const fragment = entry.fragments.find((f) => f.number === source);
  if (fragment) row.append(readLink(fragment.doc_path, fragment.lines[0], "btn-soft", "Leer en la nota"));
  const explain = el("a", "btn", "Explicar");
  explain.href = `#/conversar?q=${encodeURIComponent(`Explícame: ${label}`)}`;
  const feynmanLink = el("a", "btn", "Explícamelo tú");
  feynmanLink.href = `#/practicar?explicar=${encodeURIComponent(label)}`;
  row.append(explain, feynmanLink);
  return row;
}

function editButton(label, title, disabled, edit) {
  const button = el("button", "btn", label);
  button.type = "button";
  button.title = title;
  button.disabled = disabled || mapState.busy;
  button.addEventListener("click", () => editDiagram(edit));
  return button;
}

function renderDiagramPanel(entry) {
  const map = entry.map;
  const id = mapState.selected;
  let parts = null;
  if (entry.kind === "flow") {
    const steps = map.data.versions[map.data.current].steps;
    const i = steps.findIndex((s) => s.id === id);
    if (i >= 0) {
      const s = steps[i];
      const into = map.drawing.lines.find((ln) => ln.target === s.id && ln.role === "flow" && ln.points.length === 2);
      parts = [el("span", "eyebrow", `${s.decision ? "Decisión" : "Paso"} ${i + 1} de ${steps.length}`), el("h2", null, s.label)];
      parts.push(quoteFigure(entry, s.quote, s.sources[0]));
      if (into && into.dashed) parts.push(el("p", "map-note", "Flecha de entrada inferida: tus notas describen el paso, pero no escriben que vaya después del anterior. Puedes moverlo."));
      const move = el("div", "row");
      move.append(
        editButton("↑ Subir", "Mover este paso antes del anterior", i === 0, { op: "move_step", step: s.id, delta: -1 }),
        editButton("↓ Bajar", "Mover este paso después del siguiente", i === steps.length - 1, { op: "move_step", step: s.id, delta: 1 }),
      );
      parts.push(move, pieceActions(entry, s.label, s.sources[0]), ...sourceList(entry, s.sources));
    }
  } else if (entry.kind === "sequence") {
    const version = map.data.versions[map.data.current];
    const i = version.messages.findIndex((m) => m.id === id);
    const box = map.drawing.boxes.find((b) => b.id === id);
    if (i >= 0) {
      const m = version.messages[i];
      parts = [el("span", "eyebrow", `${m.reply ? "Respuesta" : "Mensaje"} ${i + 1} de ${version.messages.length}`), el("h2", null, m.label)];
      parts.push(el("p", null, `De ${m.sender} a ${m.receiver}${m.condition ? `, si ${m.condition.replace(/^si\s+/i, "")}` : ""}.`));
      parts.push(quoteFigure(entry, m.quote, version.source), pieceActions(entry, m.label, version.source), ...sourceList(entry, [version.source]));
    } else if (box) {
      const own = version.messages.filter((m) => m.sender === box.label || m.receiver === box.label);
      parts = [el("span", "eyebrow", "Participante"), el("h2", null, box.label)];
      const list = el("ul", "map-relations");
      for (const m of own) list.append(el("li", null, `${m.sender} → ${m.receiver}: ${m.label}`));
      parts.push(el("h3", null, `${own.length} ${own.length === 1 ? "mensaje" : "mensajes"}`), list, pieceActions(entry, box.label, version.source));
    }
  } else {
    const causes = new Map(map.data.causes.map((c) => [c.id, c]));
    const cause = causes.get(id);
    const group = map.data.categories.find((g) => g.causes.includes(id));
    const box = map.drawing.boxes.find((b) => b.id === id);
    if (cause) {
      parts = [el("span", "eyebrow", `Causa${group ? ` · ${group.name}` : ""}`), el("h2", null, cause.label)];
      parts.push(el("p", null, `Contribuye a: ${map.data.effect}`), quoteFigure(entry, cause.quote, cause.sources[0]));
      if (map.data.categories.length > 1) {
        parts.push(el("h3", null, "Mover a otra categoría"));
        const chips = el("div", "row map-chips");
        for (const g of map.data.categories) {
          const chip = editButton(g.name, `Mover a «${g.name}»`, g === group, { op: "move_cause", cause: cause.id, category: g.name });
          if (g === group) chip.classList.add("active");
          chips.append(chip);
        }
        parts.push(chips);
      }
      parts.push(pieceActions(entry, cause.label, cause.sources[0]), ...sourceList(entry, cause.sources));
    } else if (box && box.role === "category") {
      const members = (map.data.categories.find((g) => g.name === box.label) || { causes: [] }).causes.map((c) => causes.get(c)).filter(Boolean);
      parts = [el("span", "eyebrow", "Categoría"), el("h2", null, box.label)];
      const list = el("ul", "map-relations");
      for (const c of members) {
        const li = el("li");
        li.append(pickButton(c.label, c.id));
        list.append(li);
      }
      parts.push(el("h3", null, mapUnits("fishbone", members.length)), list);
    } else if (box && box.role === "effect") {
      parts = [el("span", "eyebrow", "Efecto"), el("h2", null, box.label)];
      parts.push(el("p", null, `${mapUnits("fishbone", map.data.causes.length)} en ${map.data.categories.length} ${map.data.categories.length === 1 ? "categoría" : "categorías"}.`));
    }
  }
  $("map-panel-empty").hidden = Boolean(parts);
  $("map-panel-view").hidden = !parts;
  if (parts) $("map-panel-view").replaceChildren(...parts);
}

async function editDiagram(edit) {
  const entry = current();
  if (!entry || !entry.id || mapState.busy) return;
  showError("map-error", "");
  try {
    const response = await postJSON("/api/maps/edit", { id: entry.id, ...edit });
    if (!response.ok) throw new Error(await errorText(response));
    entry.map = (await response.json()).map;
    renderAll();
    if (edit.op === "version") fitView();
  } catch (error) {
    showError("map-error", error.message);
  }
}

async function regroupCauses(grouping, categories) {
  const entry = current();
  if (!entry || !entry.id || mapState.busy) return;
  showError("map-error", "");
  mapState.busy = true;
  setStatus(grouping === "6m" ? "Repartiendo las causas en las 6M…" : "Agrupando las causas…");
  renderAll();
  try {
    const response = await postJSON("/api/maps/regroup", { id: entry.id, grouping, categories });
    if (!response.ok) throw new Error(await errorText(response));
    entry.map = (await response.json()).map;
    setStatus(`${mapUnits("fishbone", entry.map.count)} en ${entry.map.data.categories.length} categorías.`);
  } catch (error) {
    showError("map-error", error.message);
    setStatus("");
  } finally {
    mapState.busy = false;
    renderAll();
    fitView();
  }
}

/** Above the drawing: what only one type has (categories, versions, the dashed-arrow note). */
function renderDiagramBar() {
  const entry = current();
  const bar = $("map-diagram-bar");
  const show = isDiagram(entry) && entry.map && entry.id;
  bar.hidden = !show;
  if (!show) {
    bar.replaceChildren();
    return;
  }
  const map = entry.map;
  const parts = [];
  if (entry.kind === "fishbone") {
    const by = { model: "propuestas por el modelo a partir de las causas", "6m": "las 6M", custom: "las tuyas", none: "sin agrupar" }[map.data.grouping];
    parts.push(el("span", null, `Categorías: ${by}.`));
    const model = el("button", "btn", map.data.grouping === "model" ? "Proponer otras" : "Que las proponga el modelo");
    model.type = "button";
    model.addEventListener("click", () => regroupCauses("model"));
    const six = el("button", "btn", "Usar las 6M");
    six.type = "button";
    six.title = "Método, Máquina, Materiales, Mano de obra, Medición y Medio ambiente";
    six.addEventListener("click", () => regroupCauses("6m"));
    const own = el("button", "btn", "Mis categorías…");
    own.type = "button";
    const form = el("form", "map-own-categories");
    form.hidden = true;
    const input = el("input");
    input.type = "text";
    input.maxLength = 300;
    input.placeholder = "Categorías separadas por comas: Datos, Modelo, Despliegue";
    input.setAttribute("aria-label", "Tus categorías, separadas por comas");
    input.value = map.data.grouping === "custom" ? map.data.categories.map((g) => g.name).filter((n) => n !== "Otras").join(", ") : "";
    const apply = el("button", "btn-primary", "Agrupar");
    apply.type = "submit";
    form.append(input, apply);
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const names = input.value.split(",").map((n) => n.trim()).filter(Boolean);
      regroupCauses("custom", names);
    });
    own.addEventListener("click", () => {
      form.hidden = !form.hidden;
      if (!form.hidden) input.focus();
    });
    for (const button of [model, six, own]) button.disabled = mapState.busy;
    parts.push(model, six, own, form, el("span", "small muted", "Reagrupar no vuelve a leer tus notas."));
  } else {
    const versions = map.data.versions;
    if (versions.length > 1) {
      // each note's process or interaction is a version of its own (ADR-037)
      const what = entry.kind === "flow" ? "procesos" : "interacciones";
      parts.push(el("span", null, `${versions.length} ${what} en tus notas:`));
      versions.forEach((v, i) => {
        const fragment = entry.fragments.find((f) => f.number === v.source);
        const name = fragment ? fragment.sections.at(-1) || fragment.guide : `[${v.source}]`;
        const size = entry.kind === "flow" ? v.steps.length : v.messages.length;
        const button = el("button", `btn${i === map.data.current ? " active" : ""}`, `${name} (${size})`);
        button.type = "button";
        button.title = fragment ? fragment.citation : "";
        button.setAttribute("aria-pressed", String(i === map.data.current));
        button.disabled = mapState.busy;
        button.addEventListener("click", () => editDiagram({ op: "version", index: i }));
        parts.push(button);
      });
    }
    if (entry.kind === "sequence") {
      parts.push(el("span", "small muted", "Línea continua: mensaje · punteada: respuesta. Los recuadros son los casos en que ocurre."));
    } else {
      const inferred = map.drawing.lines.filter((ln) => ln.dashed).length;
      parts.push(el("span", "small muted", inferred
        ? `${inferred} ${inferred === 1 ? "flecha punteada marca" : "flechas punteadas marcan"} un orden que la nota no escribe. Selecciona un paso para moverlo.`
        : "Las flechas siguen el orden en que la nota escribe los pasos."));
    }
  }
  bar.replaceChildren(...parts);
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
  const concept = hasMap && !isDiagram(entry);
  $("map-empty").hidden = Boolean(entry);
  $("map-stage").hidden = !entry || mapState.view !== "map";
  $("map-outline").hidden = !entry || mapState.view !== "outline";
  $("map-create").disabled = mapState.busy;
  $("map-suggest").disabled = mapState.busy;
  $("map-save").disabled = !hasMap || !entry.id || mapState.busy;
  $("map-delete").disabled = !hasMap || !entry.id || mapState.busy;
  $("map-reset").disabled = !concept || mapState.busy || !entry.positions || !Object.keys(entry.positions).length;
  markActiveMap();
  $("map-svg").disabled = !hasMap || mapState.busy;
  $("map-mermaid").disabled = !hasMap || mapState.busy || !mapMermaid(entry);
  const masteryReady = concept && Boolean(entry.id) && !mapState.busy;
  $("map-mastery").disabled = !masteryReady;
  $("map-mastery").setAttribute("aria-pressed", String(mapState.mastery));
  $("map-legend").hidden = !(mapState.mastery && masteryReady && entry.mastery);
  if (masteryReady && !entry.mastery && !entry.masteryLoading) loadMastery(entry);
  renderTrail();
  renderGap();
  renderDiagramBar();
  drawMap();
  renderMapOutline();
  renderPanel();
}

// --- mastery (ADR-026) --------------------------------------------------------------------

async function loadMastery(entry) {
  entry.masteryLoading = true;
  try {
    const response = await fetch(`/api/maps/history/${entry.id}/mastery`);
    if (response.ok) entry.mastery = await response.json();
  } catch {
    // without mastery the map is drawn as before
  } finally {
    entry.masteryLoading = false;
  }
  if (entry.mastery && current() === entry && !mapState.busy) renderAll();
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
    const kind = MAP_KINDS[m.kind] ? m.kind : "concepts";
    button.append(el("span", null, `${MAP_KINDS[kind].label} · ${formatWhen(m.created)} · ${mapUnits(kind, m.concepts)}`));
    if (m.snippet) button.append(el("span", "history-snippet", m.snippet));
    button.addEventListener("click", () => openStoredMap(m.id));
    li.append(button);
    return li;
  });
  $("map-history-list").replaceChildren(...items);
  $("map-history-empty").hidden = items.length > 0;
  $("map-history-empty").textContent = query
    ? "Ningún mapa trata de eso."
    : "Los mapas y diagramas que crees aparecerán aquí.";
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
  const kind = MAP_KINDS[data.kind] ? data.kind : "concepts";
  const entry = { id: data.id, kind, topic: data.topic, map: data.map, fragments: data.fragments };
  if (!isDiagram(entry)) applyPositions(entry, data.positions);
  mapState.trail = [entry];
  mapState.selected = null;
  mapState.moved = false;
  $("map-topic").value = data.topic;
  setMapKind(kind);
  $("map-suggestion").hidden = true;
  setStatus(`${MAP_KINDS[kind].label} · ${mapUnits(kind, data.concepts)} · creado el ${formatWhen(data.created)}`);
  renderAll();
  fitView();
}

async function deleteMap() {
  const entry = current();
  if (!entry || !entry.id || !confirm(`¿Borrar «${entry.topic}» (${MAP_KINDS[entry.kind].label})? No se puede deshacer.`)) return;
  try {
    const response = await postJSON("/api/maps/delete", { id: entry.id });
    if (!response.ok) throw new Error(await errorText(response));
    mapState.trail.pop();
    mapState.selected = null;
    setStatus("Borrado.");
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

/** The clipboard API needs a secure page; from another device (LAN mode, plain HTTP) it is not. */
async function copyMapText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const area = el("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.append(area);
    area.select();
    let ok = false;
    try {
      ok = document.execCommand("copy");
    } catch {
      ok = false;
    }
    area.remove();
    return ok;
  }
}

async function copyMermaid() {
  const text = mapMermaid(current());
  if (!text) return;
  const ok = await copyMapText(`\`\`\`mermaid\n${text}\`\`\`\n`);
  setStatus(ok
    ? "Mermaid copiado: pégalo en una nota de Obsidian y lo dibuja."
    : "No se pudo copiar: está en la vista «Esquema», al final.");
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
  setMapKind(store.get("biblio.mapKind", "concepts"));
  for (const radio of document.querySelectorAll('input[name="map-kind"]')) {
    radio.addEventListener("change", () => {
      setMapKind(radio.value);
      $("map-suggestion").hidden = true;
    });
  }
  $("map-suggest").addEventListener("click", suggestKind);
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
  $("map-mermaid").addEventListener("click", copyMermaid);
  $("map-delete").addEventListener("click", deleteMap);
  $("map-reset").addEventListener("click", resetLayout);
  mapState.mastery = store.get("biblio.mapMastery", false);
  $("map-mastery").addEventListener("click", () => {
    mapState.mastery = !mapState.mastery;
    store.set("biblio.mapMastery", mapState.mastery);
    renderAll();
  });
  setConceptPanelOpen(store.get("biblio.mapPanelOpen", true));
  $("map-panel-toggle").addEventListener("click", () => {
    setConceptPanelOpen($("map-panel").hidden);
    fitView(); // the drawing gains or gives back the panel's width
  });
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
    if (event.target.closest(".map-node, .dg-pick")) return; // a click on a piece selects it
    drag = { x: event.clientX, y: event.clientY, start: { ...mapState.zoom } };
    stage.setPointerCapture(event.pointerId);
  });
  stage.addEventListener("pointermove", (event) => {
    const nd = mapState.nodeDrag;
    if (nd) {
      const dx = event.clientX - nd.x;
      const dy = event.clientY - nd.y;
      if (!nd.moved && Math.hypot(dx, dy) < 4) return; // still a click
      if (mapState.busy) return; // the map is still growing: it would be redrawn under the pointer
      nd.moved = true;
      stage.classList.add("dragging-node");
      moveNode(nd.node, nd.ox + dx / mapState.zoom.k, nd.oy + dy / mapState.zoom.k);
      return;
    }
    if (!drag) return;
    mapState.zoom = { ...drag.start, x: drag.start.x + event.clientX - drag.x, y: drag.start.y + event.clientY - drag.y };
    mapState.moved = true;
    applyZoom();
  });
  const release = () => {
    const nd = mapState.nodeDrag;
    mapState.nodeDrag = null;
    drag = null;
    stage.classList.remove("dragging-node");
    if (!nd) return;
    if (nd.moved) {
      rememberPosition(nd.node);
      drawMap(); // label halos re-measured at their new place
    } else {
      selectNode(nd.node.id);
    }
  };
  stage.addEventListener("pointerup", release);
  stage.addEventListener("pointercancel", release);
}

let mapasStarted = false;

pageHooks.mapas = () => {
  if (!mapasStarted) {
    mapasStarted = true;
    bindMapas();
    loadMapHistory();
    renderAll();
  }
  for (const entry of mapState.trail) entry.mastery = null; // practice may have changed it
  if (mapasStarted && current()) renderAll();
  if (current() && mapState.view === "map") fitView();
};
