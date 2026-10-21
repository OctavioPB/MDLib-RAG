// Biblioteca — page Guías: its hub and the study plan per folder (ADR-028).
// Uses the helpers of app.js ($, el, store, postJSON, errorText, streamEvents, showError, app) and
// requireLLM of guias.js. Library and model text is only ever inserted with textContent.
"use strict";

const PLAN_STATES = {
  unread: tx("Por leer"),
  read: tx("Leída"),
  practiced: tx("Practicada"),
  mastered: tx("Dominada"),
};

const plan = { view: null, plans: [], busy: false, pendingGap: null };

const GUIAS_SCREENS = { home: "guias-home", plan: "guias-plan", write: "guias-write", spotter: "guias-spotter" };

function showGuiasScreen(name) {
  for (const [screen, id] of Object.entries(GUIAS_SCREENS)) $(id).hidden = name !== screen;
  if (name === "home") renderGuiasHome();
}

// --- the hub ---------------------------------------------------------------------------------

async function loadPlans() {
  try {
    plan.plans = (await (await fetch("/api/plans")).json()).plans;
  } catch {
    plan.plans = [];
  }
  return plan.plans;
}

async function renderGuiasHome() {
  renderGaps();
  const plans = await loadPlans();
  const current = plans.find((p) => p.next) || plans[0];
  const hero = $("guias-next");
  hero.hidden = !current;
  if (!current) return;
  const done = current.counts.practiced + current.counts.mastered;
  $("guias-next-eyebrow").textContent = tx("Tu plan · {0}", current.folder);
  const open = $("guias-next-open");
  if (current.next) {
    $("guias-next-title").textContent = `${current.next.why === "review" ? tx("Repasa") : tx("Sigue con")}: ${current.next.title}`;
    $("guias-next-summary").textContent = tx("{0} de {1} guías practicadas · {2} dominadas.", done, current.total, current.counts.mastered);
    open.hidden = !current.next.open_url;
    open.href = current.next.open_url || "#";
  } else {
    $("guias-next-title").textContent = tx("Dominaste todo el plan");
    $("guias-next-summary").textContent = tx("Las {0} guías de {1}.", current.total, current.folder);
    open.hidden = true;
  }
  hero.classList.toggle("calm", !current.next);
  $("guias-next-plan").onclick = () => openPlan(current.folder);
}

// --- pending topics (ADR-029) ---------------------------------------------------------------

async function renderGaps() {
  let gaps = [];
  try {
    gaps = (await (await fetch("/api/gaps")).json()).gaps;
  } catch {
    // the hub works without them
  }
  $("guias-gaps").hidden = !gaps.length;
  $("gap-list").replaceChildren(...gaps.map(gapRow));
}

function gapWhy(gap) {
  const last = gap.events[0];
  if (!last) return "";
  if (last.source === "map") return tx("Mapa de «{0}»: {1}", last.text, last.detail || tx("pocos conceptos"));
  if (last.source === "spotter") return tx("Gap Spotter sobre «{0}»: {1}", last.text, last.detail); // ADR-047
  return `«${last.text}»`;
}

function gapRow(gap) {
  const li = el("li", "plan-row");
  const main = el("div", "plan-main");
  const meta = el("div", "plan-meta");
  if (gap.status === "draft") meta.append(el("span", "state-pill state-read", tx("En borrador")));
  meta.append(el("span", "small muted", gap.count > 1 ? tx("{0} veces", gap.count) : tx("1 vez")));
  main.append(meta, el("strong", null, gap.topic), el("span", "small muted", gapWhy(gap)));
  const actions = el("div", "plan-actions");
  const write = el("button", "btn-soft", gap.status === "draft" && gap.draft ? tx("Seguir el borrador") : tx("Escribir guía"));
  write.type = "button";
  write.addEventListener("click", () => {
    if (gap.status === "draft" && gap.draft) {
      location.hash = `#/guias/${gap.draft}`;
      return;
    }
    plan.pendingGap = gap.id;
    showGuiasScreen("write");
    newGuide();
    $("guide-topic").value = gap.topic;
    $("guide-topic").focus();
  });
  const status = async (value) => {
    const response = await postJSON("/api/gaps/status", { id: gap.id, status: value });
    if (response.ok) renderGaps();
  };
  const done = el("button", "btn", tx("Resuelto"));
  done.type = "button";
  done.title = tx("Ya tienes una guía sobre esto en tu biblioteca");
  done.addEventListener("click", () => status("resolved"));
  const dismiss = el("button", "btn", tx("Descartar"));
  dismiss.type = "button";
  dismiss.addEventListener("click", () => status("dismissed"));
  actions.append(write, done, dismiss);
  li.append(main, actions);
  return li;
}

document.addEventListener("guide-outlined", async (event) => {
  if (!plan.pendingGap) return;
  const id = plan.pendingGap;
  plan.pendingGap = null;
  await postJSON("/api/gaps/status", { id, status: "draft", draft: event.detail.slug });
});

// --- the plan screen -------------------------------------------------------------------------

async function loadPlanFolders() {
  try {
    const data = await (await fetch("/api/folders")).json();
    $("plan-folder-options").replaceChildren(
      ...data.folders.map((f) => {
        const option = document.createElement("option");
        option.value = f.path;
        option.label = `${f.documents} ${f.documents === 1 ? tx("guía") : tx("guías")}`;
        return option;
      }),
    );
  } catch {
    // the folder can still be typed
  }
}

function renderSavedPlans() {
  const box = $("plan-saved");
  box.replaceChildren(
    ...plan.plans.map((p) => {
      const chip = el("button", "plan-chip");
      chip.type = "button";
      chip.setAttribute("aria-current", String(Boolean(plan.view && plan.view.folder === p.folder)));
      chip.append(el("strong", null, p.folder), el("span", null, tx("{0}/{1} practicadas", p.counts.practiced + p.counts.mastered, p.total)));
      chip.addEventListener("click", () => openPlan(p.folder));
      return chip;
    }),
  );
}

async function openPlan(folder) {
  showGuiasScreen("plan");
  showError("plan-error", "");
  if (!folder) {
    plan.view = null;
    await loadPlans();
    renderSavedPlans();
    renderPlan();
    return;
  }
  const response = await fetch(`/api/plans/view?folder=${encodeURIComponent(folder)}`);
  plan.view = response.ok ? await response.json() : null;
  store.set("biblio.plan", plan.view ? folder : null);
  $("plan-folder").value = folder;
  await loadPlans();
  renderSavedPlans();
  renderPlan();
}

function setPlanStatus(text) {
  $("plan-status").hidden = !text;
  $("plan-status").textContent = text || "";
}

async function generatePlan(folder) {
  if (plan.busy) return;
  showError("plan-error", "");
  try {
    await requireLLM();
  } catch (error) {
    showError("plan-error", error.message);
    return;
  }
  plan.busy = true;
  $("plan-create").disabled = true;
  setPlanStatus(tx("Leyendo las guías de la carpeta…"));
  try {
    await streamEvents("/api/plans/generate", { folder }, (name, data) => {
      if (name === "progress") {
        setPlanStatus(data.total > 1 ? tx("Ordenando {0} de {1} · {2}", data.done, data.total, data.label) : tx("Ordenando las guías…"));
      } else if (name === "plan") {
        plan.view = data;
        store.set("biblio.plan", folder);
      }
    });
    setPlanStatus("");
    await loadPlans();
    renderSavedPlans();
    renderPlan();
  } catch (error) {
    setPlanStatus("");
    showError("plan-error", error.message);
  } finally {
    plan.busy = false;
    $("plan-create").disabled = false;
  }
}

function stateBar(view) {
  const bar = el("div", "plan-bar");
  bar.setAttribute("role", "img");
  const total = view.items.length || 1;
  const parts = ["mastered", "practiced", "read"].map((s) => `${view.counts[s]} ${PLAN_STATES[s].toLowerCase()}${view.counts[s] === 1 ? "" : "s"}`);
  bar.setAttribute("aria-label", parts.join(", "));
  for (const s of ["mastered", "practiced", "read", "unread"]) {
    const part = el("span", `s-${s}`);
    part.style.width = `${(100 * view.counts[s]) / total}%`;
    bar.append(part);
  }
  return bar;
}

function renderPlan() {
  const box = $("plan-view");
  const view = plan.view;
  if (!view) {
    box.replaceChildren(el("p", "muted", plan.plans.length ? tx("Elige uno de tus planes o crea otro.") : tx("Escribe una carpeta de tu temario y propongo el orden en que estudiar sus guías (~10 s; más en carpetas grandes).")));
    return;
  }
  const head = el("div", "plan-head");
  const title = el("div");
  title.append(el("h2", null, view.folder), el("span", "small muted", tx("{0} guías · orden propuesto por el modelo; puedes moverlas", view.items.length)));
  const actions = el("div", "row");
  const again = el("button", "btn", tx("Proponer otro orden"));
  again.type = "button";
  again.addEventListener("click", () => {
    if (confirm(tx("Se reemplazará el orden actual, también tus cambios. Lo que leíste y practicaste se mantiene. ¿Continuar?"))) generatePlan(view.folder);
  });
  const remove = el("button", "btn danger-text", tx("Borrar plan"));
  remove.type = "button";
  remove.addEventListener("click", async () => {
    if (!confirm(tx("¿Borrar el plan de «{0}»? Tus lecturas y tu práctica no se borran.", view.folder))) return;
    await postJSON("/api/plans/delete", { folder: view.folder });
    store.set("biblio.plan", null);
    openPlan(null);
  });
  actions.append(again, remove);
  head.append(title, actions);

  const legend = el("ul", "plan-legend");
  for (const s of ["unread", "read", "practiced", "mastered"]) {
    const li = el("li");
    li.append(el("span", `state-pill state-${s}`, PLAN_STATES[s]), el("span", "mono", String(view.counts[s])));
    legend.append(li);
  }

  const nodes = [head, stateBar(view), legend];
  let module = null;
  let list = null;
  view.items.forEach((item, index) => {
    const group = item.new ? tx("Nuevas en la carpeta") : item.module;
    if (group !== module || !list) {
      module = group;
      if (group) nodes.push(el("h3", "plan-module", group));
      list = el("ol", "plan-list");
      nodes.push(list);
    }
    list.append(planRow(view, item, index));
  });
  box.replaceChildren(...nodes);
}

function planRow(view, item, index) {
  const isNext = view.next && view.next.doc_path === item.doc_path;
  const li = el("li", `plan-row${isNext ? " is-next" : ""}`);
  li.append(el("span", "plan-num", String(index + 1)));
  const main = el("div", "plan-main");
  const meta = el("div", "plan-meta");
  meta.append(el("span", `state-pill state-${item.state}`, PLAN_STATES[item.state]));
  if (isNext) meta.append(el("span", "plan-next-tag", view.next.why === "review" ? tx("Repasa esta") : tx("Sigue aquí")));
  if (item.new) meta.append(el("span", "badge warn", tx("Nueva")));
  else if (!item.placed) meta.append(el("span", "small muted", tx("sin ordenar por el modelo")));
  main.append(meta, el("strong", null, item.title));
  if (item.reason) main.append(el("span", "small muted", item.reason));
  if (item.practiced) {
    const recall = item.recall === null ? "" : tx(" · la recuerdas en un {0} %", Math.round(item.recall * 100));
    main.append(el("span", "small", tx("{0} de {1} secciones practicadas{2}", item.practiced, item.sections, recall)));
  }
  li.append(main);

  const actions = el("div", "plan-actions");
  actions.append(readLink(item.doc_path, null, "btn"));
  if (item.open_url) {
    const open = el("a", "btn", "Obsidian");
    open.href = item.open_url;
    open.title = tx("Abrir en Obsidian");
    actions.append(open);
  }
  if (item.state === "unread" || item.state === "read") {
    const read = el("button", "btn", item.state === "read" ? tx("No leída") : tx("Marcar leída"));
    read.type = "button";
    read.addEventListener("click", async () => {
      const response = await postJSON("/api/plans/read", { doc_path: item.doc_path, read: item.state !== "read" });
      if (response.ok) openPlan(view.folder);
    });
    actions.append(read);
  }
  const practise = el("button", "btn-soft", tx("Practicar"));
  practise.type = "button";
  practise.addEventListener("click", () => {
    app.pendingPractice = { scope: { kind: "guide", value: item.doc_path, label: item.title }, count: 10 };
    location.hash = "#/practicar";
  });
  const explain = el("a", "btn-soft", tx("Explícamelo"));
  explain.href = `#/practicar?explicar=${encodeURIComponent(item.title)}`;
  actions.append(practise, explain);
  for (const [delta, label, name] of [[-1, "↑", tx("Subir")], [1, "↓", tx("Bajar")]]) {
    const move = el("button", "btn plan-move", label);
    move.type = "button";
    move.setAttribute("aria-label", `${name} ${item.title}`);
    move.disabled = index + delta < 0 || index + delta >= view.items.length;
    move.addEventListener("click", () => movePlanItem(index, delta));
    actions.append(move);
  }
  li.append(actions);
  return li;
}

async function movePlanItem(index, delta) {
  const view = plan.view;
  const items = [...view.items];
  const [moved] = items.splice(index, 1);
  items.splice(index + delta, 0, moved);
  const response = await postJSON("/api/plans/order", { folder: view.folder, order: items.map((i) => i.doc_path) });
  if (!response.ok) {
    showError("plan-error", await errorText(response));
    return;
  }
  await openPlan(view.folder);
  const rows = $("plan-view").querySelectorAll(".plan-move");
  const target = [...rows].find((b) => b.getAttribute("aria-label") === `${delta < 0 ? tx("Subir") : tx("Bajar")} ${moved.title}`);
  if (target && !target.disabled) target.focus();
}

// --- wiring ----------------------------------------------------------------------------------

function bindPlan() {
  $("tile-plan").addEventListener("click", () => openPlan(store.get("biblio.plan", null)));
  $("tile-write").addEventListener("click", () => showGuiasScreen("write"));
  for (const back of document.querySelectorAll("#page-guias [data-guias-back]")) {
    back.addEventListener("click", () => showGuiasScreen("home"));
  }
  $("plan-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const folder = $("plan-folder").value.trim().replace(/^\/+|\/+$/g, "");
    if (!folder) {
      showError("plan-error", tx("Escribe una carpeta de tu biblioteca."));
      return;
    }
    const known = plan.plans.find((p) => p.folder === folder);
    if (known) openPlan(folder);
    else generatePlan(folder);
  });
  loadPlanFolders();
}

let planBound = false;
const guiasHook = pageHooks.guias;

pageHooks.guias = () => {
  if (!planBound) {
    planBound = true;
    bindPlan();
  }
  guiasHook();
  if (/^#\/guias\/[a-z0-9-]+$/.test(location.hash)) showGuiasScreen("write"); // a draft's link
  else if (!$("guias-plan").hidden && plan.view) openPlan(plan.view.folder); // states may have moved
  else if ($("guias-plan").hidden && $("guias-write").hidden && $("guias-spotter").hidden) showGuiasScreen("home");
};
