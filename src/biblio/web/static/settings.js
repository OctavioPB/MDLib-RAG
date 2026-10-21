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
  // the interface's language (ADR-046): kept in this browser; the page reloads in it
  for (const radio of document.querySelectorAll('input[name="lang"]')) {
    radio.checked = radio.value === I18N.lang;
    radio.addEventListener("change", () => {
      store.set("biblio.lang", radio.value);
      location.reload();
    });
  }
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
    const up = el("button", "btn-ghost", tx("↑ Subir"));
    up.type = "button";
    up.addEventListener("click", () => liblocBrowse(data.parent));
    head.append(up);
  }
  head.append(el("strong", null, data.path || tx("Unidades")));
  if (data.path) {
    const use = el("button", "btn-primary", tx("Usar esta carpeta"));
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
  if (!data.folders.length) parts.push(el("p", "small muted", tx("No hay carpetas dentro.")));
  if (data.truncated) parts.push(el("p", "small muted", tx("Hay más carpetas: escribe la ruta para llegar a ellas.")));
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
  box.replaceChildren(el("p", "muted", tx("Comprobando las notas de esa carpeta…")));
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
    parts.push(el("p", null, tx("Esa carpeta ya es tu biblioteca.")));
  } else {
    const relocate = data.plan === "relocate";
    parts.push(el("p", null, relocate
      ? tx("La misma biblioteca en otro sitio: {0} de {1} notas coinciden con el índice. Se reutiliza: solo se procesan las notas que cambiaron, unos segundos.", data.matching, data.notes)
      : tx("Otra biblioteca: {0} notas, {1} de ellas ya en el índice. Hay que indexarla entera: ~{2} en la GPU, o ~{3} en CPU mientras el modelo de lenguaje esté encendido{4}.", data.notes, data.matching, liblocMinutes(data.minutes_gpu), liblocMinutes(data.minutes_cpu), app.llmRunning ? tx(" (lo está ahora)") : "")));
    if (!relocate) {
      parts.push(el("p", "small muted", tx("Tu biblioteca actual sigue funcionando hasta que termine. El historial de práctica, los mapas, los diccionarios y los planes se refieren a notas de la biblioteca anterior.")));
    }
    const go = el("button", "btn-primary", relocate ? tx("Cambiar a esta carpeta") : tx("Indexar y cambiar"));
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
  const status = el("p", null, tx("Preparando…"));
  box.replaceChildren(status);
  try {
    await streamEvents("/api/library/set", { path }, (name, data) => {
      if (name === "start") {
        status.textContent = tx("{0} {1} notas{2}…", data.full ? tx("Indexando") : tx("Revisando"), data.files.toLocaleString(i18nLocale()), data.device === "cpu" ? tx(" en CPU") : "");
      } else if (name === "progress") {
        status.textContent = tx("{0} de {1} notas · {2} fragmentos nuevos", data.files.toLocaleString(i18nLocale()), data.total.toLocaleString(i18nLocale()), data.chunks.toLocaleString(i18nLocale()));
      } else if (name === "done") {
        status.textContent = tx("Índice listo: {0} notas ({1} nuevas, {2} cambiadas, {3} quitadas) en {4} s.", data.documents.toLocaleString(i18nLocale()), data.added, data.updated, data.deleted, data.seconds);
      } else if (name === "library") {
        $("libloc-path").textContent = data.path;
        box.append(el("p", "ok", tx("Tu biblioteca ahora es {0}.", data.path)));
      }
    });
    library.signature = ""; // the Biblioteca page lists the new notes next time it is shown
    if (currentPage() === "biblioteca") loadLibrary();
    loadStatus();
  } catch (error) {
    box.append(el("p", "error", tx("No se cambió la biblioteca: {0}", error.message)));
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
  conversations: [tx("conversación"), tx("conversaciones")],
  maps: [tx("mapa o diagrama"), tx("mapas y diagramas")],
  practice: [tx("pregunta de práctica (con sus repasos)"), tx("preguntas de práctica (con sus repasos)")],
  drafts: [tx("archivo de guías y exportaciones"), tx("archivos de guías y exportaciones")],
};

async function openResetDialog() {
  const dialog = $("reset-dialog");
  $("reset-error").hidden = true;
  $("reset-confirm").disabled = false;
  $("reset-confirm").hidden = false;
  $("reset-cancel").textContent = tx("Cancelar");
  $("reset-body").hidden = false;
  try {
    const response = await fetch("/api/reset");
    if (!response.ok) throw new Error(await errorText(response));
    const items = (await response.json()).items;
    $("reset-items").replaceChildren(...items.map((i) => {
      const [one, many] = RESET_LABELS[i.key];
      return el("li", null, `${i.count.toLocaleString(i18nLocale())} ${i.count === 1 ? one : many}`);
    }), el("li", null, tx("planes de estudio, notas marcadas como leídas, temas pendientes y diccionarios")));
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
    $("reset-title").textContent = tx("Listo");
    $("reset-body").replaceChildren(
      el("p", null, tx("La app empieza de cero. Tu biblioteca y su índice siguen igual.")),
      el("p", "small muted", tx("La memoria anterior quedó en {0}.", data.backup)),
    );
    $("reset-confirm").hidden = true;
    $("reset-cancel").textContent = tx("Empezar");
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
    $("llm-state-text").textContent = c && c.running ? tx("Encendido") : tx("Apagado");
    toggle.hidden = true;
    $("llm-help").textContent = tx("Esta sesión no controla el modelo: ábrela con Biblioteca.cmd o biblio start.");
    return;
  }
  if (c.running) {
    $("llm-state-text").textContent = c.owned ? tx("Encendido") : tx("Encendido (Ollama abierto fuera de la app)");
    toggle.hidden = false;
    toggle.textContent = tx("Apagar el modelo");
    $("llm-help").textContent = tx("Libera la memoria de la GPU. Sin el modelo puedes buscar en tus notas, pero no conversar, crear mapas ni practicar.");
  } else {
    $("llm-state-text").textContent = tx("Apagado");
    toggle.hidden = !c.can_start;
    toggle.textContent = tx("Encender el modelo");
    $("llm-help").textContent = c.can_start
      ? tx("Tarda unos segundos. Con el modelo encendido, la búsqueda deja la GPU al modelo.")
      : tx("Ollama no está instalado en este PC.");
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
  result.textContent = on ? tx("Encendiendo el modelo…") : tx("Apagando el modelo…");
  settingsLlmRender();
  try {
    const response = await postJSON(on ? "/api/llm/start" : "/api/llm/stop", {});
    if (!response.ok) throw new Error(await errorText(response));
    const data = await response.json();
    llmSwitch.control = data.control;
    result.textContent = {
      started: tx("Modelo encendido."),
      stopped: tx("Modelo apagado: la memoria de la GPU quedó libre."),
      unloaded: tx("El modelo salió de la memoria y la GPU quedó libre. Ollama se abrió fuera de la app y sigue abierto: volverá a cargar el modelo con la próxima pregunta."),
    }[data.result] || tx("Listo.");
  } catch (error) {
    result.classList.add("error");
    result.textContent = error.message;
  } finally {
    llmSwitch.busy = false;
    settingsLlmRender();
    loadStatus(); // the status in the top bar
  }
}
