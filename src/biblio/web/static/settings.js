// Biblioteca — «Ajustes» (the gear in the top bar): appearance, the library's location and
// «Actualizar», devices and «Reset», on every page. The theme switch and «Actualizar» keep their
// ids, so app.js and devices.js still drive them. Text from the server is inserted with
// textContent only.
"use strict";

function openSettingsPanel() {
  loadLibraryLocation();
  $("settings-devices").hidden = $("lan-connect").hidden;
  $("settings-dialog").showModal();
}

function bindSettingsPanel() {
  $("settings-open").addEventListener("click", openSettingsPanel);
  $("settings-close").addEventListener("click", () => $("settings-dialog").close());
  // a click on the backdrop (outside the panel) closes it
  $("settings-dialog").addEventListener("click", (event) => {
    if (event.target === $("settings-dialog")) $("settings-dialog").close();
  });
  bindLibraryLocation();
  bindAppReset();
  $("llm-toggle").addEventListener("click", settingsLlmToggle);
}

// --- where the library is (ADR-039): shown on this PC only ----------------------------------

const libloc = { busy: false, checked: null };

async function loadLibraryLocation() {
  let data = null;
  try {
    data = await (await fetch("/api/library")).json();
  } catch {
    return;
  }
  $("libloc").hidden = !data.local || !data.path;
  $("app-reset").hidden = !data.local; // this PC only (ADR-040)
  $("settings-llm").hidden = !data.local; // this PC only (ADR-043)
  if (data.local) settingsLlmLoad();
  $("settings-devices").hidden = $("lan-connect").hidden;
  $("libloc-path").textContent = data.path || "";
}

function liblocOpen(open) {
  $("libloc-panel").hidden = !open;
  $("libloc-open").setAttribute("aria-expanded", String(open));
  if (open) {
    $("libloc-input").value = $("libloc-path").textContent;
    $("libloc-input").focus();
  } else {
    $("libloc-browser").hidden = true;
    $("libloc-result").hidden = true;
  }
}

async function liblocBrowse(path) {
  const box = $("libloc-browser");
  box.hidden = false;
  let data;
  try {
    const response = await fetch(`/api/library/browse${path ? `?path=${encodeURIComponent(path)}` : ""}`);
    if (!response.ok) throw new Error(await errorText(response));
    data = await response.json();
  } catch (error) {
    box.replaceChildren(el("p", "error", error.message));
    return;
  }
  const head = el("div", "libloc-browser-head");
  if (data.parent !== null) {
    const up = el("button", "btn-ghost", "↑ Subir");
    up.type = "button";
    up.addEventListener("click", () => liblocBrowse(data.parent));
    head.append(up);
  }
  head.append(el("strong", null, data.path || "Unidades"));
  if (data.path) {
    const use = el("button", "btn-primary", "Usar esta carpeta");
    use.type = "button";
    use.addEventListener("click", () => {
      $("libloc-input").value = data.path;
      box.hidden = true;
      liblocCheck();
    });
    head.append(use);
  }
  const list = el("ul", "libloc-folders");
  for (const folder of data.folders) {
    const li = el("li");
    const button = el("button", "link-button", `📁 ${folder.name}`);
    button.type = "button";
    button.addEventListener("click", () => {
      $("libloc-input").value = folder.path;
      liblocBrowse(folder.path);
    });
    li.append(button);
    list.append(li);
  }
  const parts = [head, list];
  if (!data.folders.length) parts.push(el("p", "small muted", "No hay carpetas dentro."));
  if (data.truncated) parts.push(el("p", "small muted", "Hay más carpetas: escribe la ruta para llegar a ellas."));
  box.replaceChildren(...parts);
}

function liblocMinutes(m) {
  return m >= 90 ? `${Math.round(m / 60)} h` : `${Math.max(1, m)} min`;
}

async function liblocCheck() {
  const path = $("libloc-input").value.trim();
  const box = $("libloc-result");
  if (!path || libloc.busy) return;
  box.hidden = false;
  box.replaceChildren(el("p", "muted", "Comprobando las notas de esa carpeta…"));
  let data;
  try {
    const response = await postJSON("/api/library/check", { path });
    if (!response.ok) throw new Error(await errorText(response));
    data = await response.json();
  } catch (error) {
    box.replaceChildren(el("p", "error", error.message));
    return;
  }
  libloc.checked = data;
  const parts = [];
  if (data.plan === "invalid") {
    parts.push(el("p", "error", `${data.path}: ${data.reason}`));
  } else if (data.plan === "same") {
    parts.push(el("p", null, "Esa carpeta ya es tu biblioteca."));
  } else {
    const relocate = data.plan === "relocate";
    parts.push(el("p", null, relocate
      ? `La misma biblioteca en otro sitio: ${data.matching} de ${data.notes} notas coinciden con el índice. Se reutiliza: solo se procesan las notas que cambiaron, unos segundos.`
      : `Otra biblioteca: ${data.notes} notas, ${data.matching} de ellas ya en el índice. Hay que indexarla entera: ~${liblocMinutes(data.minutes_gpu)} en la GPU, o ~${liblocMinutes(data.minutes_cpu)} en CPU mientras el modelo de lenguaje esté encendido${app.llmRunning ? " (lo está ahora)" : ""}.`));
    if (!relocate) {
      parts.push(el("p", "small muted", "Tu biblioteca actual sigue funcionando hasta que termine. El historial de práctica, los mapas, los diccionarios y los planes se refieren a notas de la biblioteca anterior."));
    }
    const go = el("button", "btn-primary", relocate ? "Cambiar a esta carpeta" : "Indexar y cambiar");
    go.type = "button";
    go.addEventListener("click", () => liblocApply(data.path));
    parts.push(go);
  }
  box.replaceChildren(...parts);
}

async function liblocApply(path) {
  if (libloc.busy) return;
  libloc.busy = true;
  const box = $("libloc-result");
  const status = el("p", null, "Preparando…");
  box.replaceChildren(status);
  try {
    await streamEvents("/api/library/set", { path }, (name, data) => {
      if (name === "start") {
        status.textContent = `${data.full ? "Indexando" : "Revisando"} ${data.files.toLocaleString("es")} notas${data.device === "cpu" ? " en CPU" : ""}…`;
      } else if (name === "progress") {
        status.textContent = `${data.files.toLocaleString("es")} de ${data.total.toLocaleString("es")} notas · ${data.chunks.toLocaleString("es")} fragmentos nuevos`;
      } else if (name === "done") {
        status.textContent = `Índice listo: ${data.documents.toLocaleString("es")} notas (${data.added} nuevas, ${data.updated} cambiadas, ${data.deleted} quitadas) en ${data.seconds} s.`;
      } else if (name === "library") {
        $("libloc-path").textContent = data.path;
        box.append(el("p", "ok", `Tu biblioteca ahora es ${data.path}.`));
      }
    });
    library.signature = ""; // the Biblioteca page lists the new notes next time it is shown
    if (currentPage() === "biblioteca") loadLibrary();
    loadStatus();
  } catch (error) {
    box.append(el("p", "error", `No se cambió la biblioteca: ${error.message}`));
  } finally {
    libloc.busy = false;
  }
}

function bindLibraryLocation() {
  $("libloc-open").addEventListener("click", () => liblocOpen($("libloc-panel").hidden));
  $("libloc-cancel").addEventListener("click", () => liblocOpen(false));
  $("libloc-browse").addEventListener("click", () => liblocBrowse($("libloc-input").value.trim()));
  $("libloc-form").addEventListener("submit", (event) => {
    event.preventDefault();
    liblocCheck();
  });
}

// --- «Reset»: the app's memory, never the library (ADR-040) --------------------------------------

const RESET_LABELS = {
  conversations: ["conversación", "conversaciones"],
  maps: ["mapa o diagrama", "mapas y diagramas"],
  practice: ["pregunta de práctica (con sus repasos)", "preguntas de práctica (con sus repasos)"],
  drafts: ["archivo de guías y exportaciones", "archivos de guías y exportaciones"],
};

async function openResetDialog() {
  const dialog = $("reset-dialog");
  $("reset-error").hidden = true;
  $("reset-confirm").disabled = false;
  $("reset-confirm").hidden = false;
  $("reset-cancel").textContent = "Cancelar";
  $("reset-body").hidden = false;
  try {
    const response = await fetch("/api/reset");
    if (!response.ok) throw new Error(await errorText(response));
    const items = (await response.json()).items;
    $("reset-items").replaceChildren(...items.map((i) => {
      const [one, many] = RESET_LABELS[i.key];
      return el("li", null, `${i.count.toLocaleString("es")} ${i.count === 1 ? one : many}`);
    }), el("li", null, "planes de estudio, notas marcadas como leídas, temas pendientes y diccionarios"));
  } catch (error) {
    $("reset-items").replaceChildren(el("li", "error", error.message));
  }
  dialog.showModal();
}

async function confirmReset() {
  $("reset-confirm").disabled = true;
  $("reset-error").hidden = true;
  try {
    const response = await postJSON("/api/reset", { confirm: true });
    if (!response.ok) throw new Error(await errorText(response));
    const data = await response.json();
    for (const key of Object.keys(localStorage)) {
      if (key.startsWith("biblio.") && key !== "biblio.theme") localStorage.removeItem(key); // keep the look
    }
    $("reset-title").textContent = "Listo";
    $("reset-body").replaceChildren(
      el("p", null, "La app empieza de cero. Tu biblioteca y su índice siguen igual."),
      el("p", "small muted", `La memoria anterior quedó en ${data.backup}.`),
    );
    $("reset-confirm").hidden = true;
    $("reset-cancel").textContent = "Empezar";
    $("reset-cancel").onclick = () => { location.hash = "#/conversar"; location.reload(); };
  } catch (error) {
    $("reset-error").textContent = error.message;
    $("reset-error").hidden = false;
    $("reset-confirm").disabled = false;
  }
}

function bindAppReset() {
  $("app-reset-open").addEventListener("click", openResetDialog);
  $("reset-confirm").addEventListener("click", confirmReset);
  $("reset-cancel").addEventListener("click", () => $("reset-dialog").close());
}

bindSettingsPanel();

// --- the language model, switched off and on (ADR-043): this PC only ----------------------------

const llmSwitch = { control: null, busy: false };

async function settingsLlmLoad() {
  try {
    llmSwitch.control = (await (await fetch("/api/status")).json()).llm_control;
  } catch {
    llmSwitch.control = null;
  }
  settingsLlmRender();
}

function settingsLlmRender() {
  const c = llmSwitch.control;
  const toggle = $("llm-toggle");
  $("settings-llm").classList.toggle("is-on", Boolean(c && c.running));
  if (!c || !c.available) {
    $("llm-state-text").textContent = c && c.running ? "Encendido" : "Apagado";
    toggle.hidden = true;
    $("llm-help").textContent = "Esta sesión no controla el modelo: ábrela con Biblioteca.cmd o biblio start.";
    return;
  }
  if (c.running) {
    $("llm-state-text").textContent = c.owned ? "Encendido" : "Encendido (Ollama abierto fuera de la app)";
    toggle.hidden = false;
    toggle.textContent = "Apagar el modelo";
    $("llm-help").textContent = "Libera la memoria de la GPU. Sin el modelo puedes buscar en tus notas, pero no conversar, crear mapas ni practicar.";
  } else {
    $("llm-state-text").textContent = "Apagado";
    toggle.hidden = !c.can_start;
    toggle.textContent = "Encender el modelo";
    $("llm-help").textContent = c.can_start
      ? "Tarda unos segundos. Con el modelo encendido, la búsqueda deja la GPU al modelo."
      : "Ollama no está instalado en este PC.";
  }
  toggle.disabled = llmSwitch.busy;
}

async function settingsLlmToggle() {
  const c = llmSwitch.control;
  if (!c || llmSwitch.busy) return;
  const on = !c.running;
  llmSwitch.busy = true;
  const result = $("llm-result");
  result.hidden = false;
  result.classList.remove("error");
  result.textContent = on ? "Encendiendo el modelo…" : "Apagando el modelo…";
  settingsLlmRender();
  try {
    const response = await postJSON(on ? "/api/llm/start" : "/api/llm/stop", {});
    if (!response.ok) throw new Error(await errorText(response));
    const data = await response.json();
    llmSwitch.control = data.control;
    result.textContent = {
      started: "Modelo encendido.",
      stopped: "Modelo apagado: la memoria de la GPU quedó libre.",
      unloaded: "El modelo salió de la memoria y la GPU quedó libre. Ollama se abrió fuera de la app y sigue abierto: volverá a cargar el modelo con la próxima pregunta.",
    }[data.result] || "Listo.";
  } catch (error) {
    result.classList.add("error");
    result.textContent = error.message;
  } finally {
    llmSwitch.busy = false;
    settingsLlmRender();
    loadStatus(); // the status in the top bar
  }
}
