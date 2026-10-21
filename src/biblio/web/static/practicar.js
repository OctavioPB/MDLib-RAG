// Biblioteca — page Practicar (ADR-020): quizzes generated from library sections.
// Uses the helpers of app.js ($, el, store, postJSON, errorText, streamEvents, markdownNode, app).
// The page never receives an answer before the user answers: the server grades and records.
"use strict";

const KIND_LABELS = {
  mc: tx("Opción múltiple"),
  tf: tx("Verdadero o falso"),
  open_self: tx("Respuesta abierta · te calificas tú"),
  open_model: tx("Respuesta abierta · la califica el modelo (experimental)"),
};
const RESULT_LABELS = { correct: tx("Correcta"), partial: tx("A medias"), wrong: tx("Incorrecta") };
const RESULT_MARKS = { correct: "✓", partial: "~", wrong: "✗" };

const practice = {
  guides: new Map(), // datalist value → path
  folders: new Set(),
  quiz: null,
};

function newQuiz(label) {
  return {
    label,
    planned: 0,
    questions: [], // in arrival order
    skipped: 0,
    generating: true,
    error: null,
    index: 0,
    answers: new Map(), // question id → {result, feedback, attempt, step, …}
    controller: new AbortController(),
  };
}

const SCREENS = {
  setup: "practice-setup", // the hub
  form: "practice-form-screen",
  "feynman-intro": "practice-feynman-intro",
  progress: "practice-progress-screen",
  dictionary: "practice-dictionary", // dictionary.js (ADR-034)
};

function showScreen(name) {
  for (const [screen, id] of Object.entries(SCREENS)) $(id).hidden = name !== screen;
  $("practice-quiz").hidden = name !== "quiz";
  $("practice-results").hidden = name !== "results";
  $("practice-feynman").hidden = name !== "feynman"; // «Explícamelo tú», feynman.js
}

function plainText(text) {
  return markdownNode(text, (n) => el("span", null, `[${n}]`), "md");
}

// --- setup ---------------------------------------------------------------------------------

async function loadScopes() {
  try {
    const guides = (await (await fetch("/api/practice/guides")).json()).guides;
    const options = guides.map((g) => {
      const value = `${g.title} (${g.path})`;
      practice.guides.set(value, g.path);
      practice.guides.set(g.path, g.path);
      const option = el("option");
      option.value = value;
      return option;
    });
    $("guide-options").replaceChildren(...options);
  } catch {
    /* typing a path still works */
  }
  const folders = app.folders.length ? app.folders : (await (await fetch("/api/folders")).json()).folders;
  $("folder-options").replaceChildren(
    ...folders.map((f) => {
      practice.folders.add(f.path);
      const option = el("option");
      option.value = f.path;
      option.label = tx("{0} notas", f.documents);
      return option;
    }),
  );
}

function selectedScope() {
  const kind = document.querySelector('input[name="practice-scope"]:checked').value;
  const raw = $(`scope-${kind}`).value.trim();
  if (!raw) throw new Error(tx("Escribe la guía, la carpeta o el tema que quieres practicar."));
  if (kind === "guide") {
    const path = practice.guides.get(raw);
    if (!path) throw new Error(tx("Elige una guía de la lista que aparece al escribir."));
    return { kind, value: path, label: raw.replace(/ \([^()]*\)$/, "") };
  }
  if (kind === "folder") {
    const folder = raw.replace(/^\/+|\/+$/g, "");
    if (!practice.folders.has(folder)) throw new Error(tx("Esa carpeta no existe en tu biblioteca."));
    return { kind, value: folder, label: folder };
  }
  return { kind, value: raw, label: raw };
}

async function renderProgress() {
  const box = $("practice-progress");
  let data;
  try {
    data = await (await fetch("/api/practice/progress")).json();
  } catch {
    box.replaceChildren(el("p", "muted", tx("No se pudo leer tu progreso.")));
    renderHubProgress(null);
    return;
  }
  renderHubProgress(data);
  const nodes = [el("h2", null, tx("Tu progreso"))];
  if (!data.guides.length) {
    nodes.push(el("p", "muted", tx("Aún no has practicado. Tus resultados aparecerán aquí.")));
  }
  for (const g of data.guides.slice(0, 6)) {
    const row = el("div", "progress-row");
    const head = el("div", "progress-head");
    head.append(el("span", null, g.title));
    head.append(el("span", "mono", `${formatScore(g.score)}/${g.answered}`));
    const bar = el("div", "bar");
    const fill = el("div", "bar-fill");
    fill.style.width = `${Math.round((100 * g.score) / Math.max(g.answered, 1))}%`;
    bar.append(fill);
    row.append(head, bar);
    if (g.to_review.length) {
      const more = g.to_review.length > 2 ? tx(" y {0} más", g.to_review.length - 2) : "";
      row.append(el("span", "small muted", tx("Repasar: {0}{1}", g.to_review.slice(0, 2).join(", "), more)));
    }
    nodes.push(row);
  }
  const notes = el("div", "progress-notes");
  if (data.grader.reviewed) {
    notes.append(el("p", null, tx("Calificación del modelo: coincidió contigo en {0} de {1} respuestas.", data.grader.agreed, data.grader.reviewed)));
    if (data.grader.unreliable) {
      notes.append(el("p", "flag warn", tx("Sus notas coinciden poco con las tuyas: no te fíes de ellas.")));
    }
  }
  if (data.feynman && data.feynman.votes) {
    const f = data.feynman;
    notes.append(el("p", null, tx("«Explícamelo tú»: te pareció justo {0} de {1} puntos señalados.", f.fair, f.votes)));
    if (f.unreliable) {
      notes.append(el("p", "flag warn", tx("Lo que señala coincide poco con tu criterio: tómalo con cautela.")));
    }
  }
  if (data.questions) {
    notes.append(el("p", null, tx("Preguntas marcadas como mal hechas: {0} de {1}.", data.flagged, data.questions)));
  }
  notes.append(el("p", null, tx("Tu historial se guarda solo en este equipo.")));
  nodes.push(notes);
  box.replaceChildren(...nodes);
}

function formatScore(score) {
  return Number.isInteger(score) ? String(score) : score.toFixed(1);
}

// --- running a quiz ------------------------------------------------------------------------

// `origin`: the screen the quiz was started from, where its errors are shown
async function startQuiz(scope, count, origin = "setup") {
  const errorBox = origin === "form" ? "practice-error" : "hub-error";
  const kinds = [...document.querySelectorAll('input[name="practice-kind"]:checked')].map((i) => i.value);
  showError("practice-error", "");
  showError("hub-error", "");
  if (!kinds.length) {
    showScreen(origin);
    showError(errorBox, tx("Elige al menos un tipo de pregunta en Cuestionario."));
    return;
  }
  if (practice.quiz && practice.quiz.generating) practice.quiz.controller.abort();
  const quiz = newQuiz(scope.label);
  practice.quiz = quiz;
  $("practice-start").disabled = true;
  try {
    await loadStatus();
    // a review can ask its stored questions again without the model (ADR-025)
    if (!app.llmRunning && scope.kind !== "srs") {
      throw new Error(tx("El modelo de lenguaje no está encendido. Inícialo con scripts/serve-llm.ps1."));
    }
    showScreen("quiz");
    renderQuiz();
    await streamEvents(
      "/api/practice/start",
      { scope: { kind: scope.kind, value: scope.value }, count, kinds },
      (name, data) => {
        if (practice.quiz !== quiz) return;
        if (name === "quiz") {
          quiz.planned = data.planned;
          quiz.reused = data.reused;
        }
        else if (name === "question") quiz.questions.push(data);
        else if (name === "skipped") quiz.skipped += 1;
        else if (name === "done") quiz.generating = false;
        // while the user answers, a question arriving behind must not redraw (and blur) the card
        const current = quiz.questions[quiz.index];
        if (current && $("practice-quiz").dataset.qid === String(current.id)) updateQuizHead();
        else renderQuiz();
      },
      quiz.controller.signal,
    );
  } catch (error) {
    if (error.name === "AbortError") return;
    if (!quiz.questions.length) {
      showScreen(origin);
      showError(errorBox, error.message);
    } else {
      quiz.error = error.message;
    }
  } finally {
    quiz.generating = false;
    $("practice-start").disabled = false;
    if (practice.quiz === quiz && !$("practice-quiz").hidden) renderQuiz();
  }
}

// --- today's reviews (ADR-025) --------------------------------------------------------------

async function fetchToday() {
  try {
    const response = await fetch("/api/practice/today");
    return response.ok ? await response.json() : null;
  } catch {
    return null;
  }
}

function showTodayBadge(data) {
  const badge = $("today-badge");
  badge.hidden = !data || !data.today;
  badge.textContent = data && data.today ? String(data.today) : "";
  badge.title = data && data.today ? tx("{0} repasos pendientes hoy", data.today) : "";
}

// The hub's hero: today's reviews first (ADR-025); with none due, the way to start.
async function renderToday() {
  const data = await fetchToday();
  showTodayBadge(data);
  const start = $("today-start");
  let title, summary, note = "", mode = null;
  if (data && data.today) {
    const word = data.today === 1 ? tx("sección") : tx("secciones");
    title = tx("{0} {1} que empiezas a olvidar", data.today, word);
    summary = tx("Unos {0} {1}. Primero las que más se te escapan.", data.minutes, data.minutes === 1 ? tx("minuto") : tx("minutos"));
    const later = data.due - data.today;
    if (later > 0) note = tx("Quedan {0} más para otro día: el límite es {1} al día.", later, data.limit);
    mode = "srs";
  } else if (data && data.done) {
    title = tx("Repaso de hoy terminado");
    summary = `${data.done} ${data.done === 1 ? tx("pregunta") : tx("preguntas")}. ${data.due ? tx("Quedan {0} para mañana.", data.due) : tx("Mañana te diré qué toca.")}`;
  } else {
    title = tx("Nada que repasar hoy");
    summary = tx("Las secciones que practiques volverán aquí justo cuando empieces a olvidarlas.");
    mode = "quiz";
  }
  $("today-title").textContent = title;
  $("today-summary").textContent = summary;
  $("today-note").textContent = note;
  $("today-note").hidden = !note;
  $("practice-today").classList.toggle("calm", mode !== "srs");
  start.hidden = mode === null;
  start.dataset.mode = mode || "";
  start.textContent = mode === "srs" ? tx("Empezar repaso") : tx("Hacer un cuestionario");
}

function renderHubProgress(data) {
  const box = $("hub-progress");
  const review = $("tile-review");
  review.disabled = !data || !data.to_review;
  $("tile-review-text").textContent = data && data.to_review
    ? tx("{0} {1} la última vez.", data.to_review, data.to_review === 1 ? tx("sección que fallaste") : tx("secciones que fallaste"))
    : tx("No tienes fallos pendientes.");
  practice.toReview = data ? data.to_review : 0;
  const nodes = [el("h2", null, tx("Tu progreso"))];
  if (!data || !data.guides.length) {
    nodes.push(el("p", "muted hub-progress-empty", data ? tx("Aún no has practicado. Tus resultados aparecerán aquí.") : tx("No se pudo leer tu progreso.")));
  } else {
    for (const g of data.guides.slice(0, 2)) {
      const row = el("div", "hub-progress-row");
      const head = el("div", "progress-head");
      head.append(el("span", null, g.title), el("span", "mono", `${formatScore(g.score)}/${g.answered}`));
      const bar = el("div", "bar");
      const fill = el("div", "bar-fill");
      fill.style.width = `${Math.round((100 * g.score) / Math.max(g.answered, 1))}%`;
      bar.append(fill);
      row.append(head, bar);
      nodes.push(row);
    }
  }
  const all = el("button", "link-button hub-progress-all", tx("Ver todo →"));
  all.type = "button";
  all.addEventListener("click", () => showScreen("progress"));
  if (data && data.guides.length) nodes.push(all);
  box.replaceChildren(...nodes);
}

function finishQuiz() {
  const quiz = practice.quiz;
  if (quiz.generating) quiz.controller.abort(); // stop generating questions nobody will see
  quiz.generating = false;
  renderResults();
  showScreen("results");
}

function quizTotal(quiz) {
  return quiz.generating ? Math.max(quiz.planned - quiz.skipped, quiz.questions.length) : quiz.questions.length;
}

function updateQuizHead() {
  const quiz = practice.quiz;
  const total = quizTotal(quiz);
  $("quiz-where").textContent = total
    ? tx("Pregunta {0} de {1} · {2}", Math.min(quiz.index + 1, total), total, quiz.label)
    : quiz.label;
  const answered = [...quiz.answers.values()].filter((a) => a.result).length;
  $("quiz-bar").style.width = `${total ? Math.round((100 * answered) / total) : 0}%`;
}

function renderQuiz() {
  const quiz = practice.quiz;
  const box = $("practice-quiz");
  const question = quiz.questions[quiz.index];
  box.dataset.qid = question ? String(question.id) : "";
  const head = el("div", "quiz-head");
  const where = el("span");
  where.id = "quiz-where";
  const stop = el("button", "btn-ghost", tx("Terminar"));
  stop.type = "button";
  stop.addEventListener("click", finishQuiz);
  head.append(where, stop);
  const bar = el("div", "bar");
  const fill = el("div", "bar-fill");
  fill.id = "quiz-bar";
  bar.append(fill);
  const nodes = [head, bar];
  if (!question) {
    if (quiz.generating) {
      const wait = el("section", "practice-card waiting");
      wait.append(el("p", null, quiz.questions.length ? tx("Preparando la siguiente pregunta…") : tx("Leyendo tus notas y preparando la primera pregunta…")));
      wait.append(el("p", "small muted", tx("Cada pregunta sale de una sección de tus notas y tarda unos segundos.")));
      nodes.push(wait);
    } else if (!quiz.questions.length) {
      nodes.push(el("p", "error", quiz.error || tx("No se pudo generar ninguna pregunta. Prueba con otra guía o tema.")));
    } else {
      finishQuiz();
      return;
    }
  } else {
    nodes.push(questionCard(question));
  }
  if (quiz.skipped) {
    nodes.push(el("p", "small muted", tx("{0} {1}: el modelo no escribió una pregunta válida.", quiz.skipped, quiz.skipped === 1 ? tx("sección se omitió") : tx("secciones se omitieron"))));
  }
  if (quiz.error && question) nodes.push(el("p", "error", quiz.error));
  box.replaceChildren(...nodes);
  updateQuizHead();
}

function questionCard(q) {
  const quiz = practice.quiz;
  const state = quiz.answers.get(q.id) || { step: "answering", choice: null, text: "" };
  const card = el("section", "practice-card question-card");
  card.append(el("span", "tag", KIND_LABELS[q.kind]));
  card.append(el("h1", "question-text", q.text));

  const answered = state.step === "done";
  const working = state.step === "working";
  let action = null; // the submit button, enabled once there is an answer

  if (q.kind === "mc") {
    const group = el("div", "options");
    group.setAttribute("role", "radiogroup");
    group.setAttribute("aria-label", tx("Opciones"));
    q.options.forEach((option, i) => {
      const button = el("button", "option");
      button.type = "button";
      button.setAttribute("role", "radio");
      button.setAttribute("aria-checked", String(state.choice === i));
      if (answered && state.feedback.correct === i) button.classList.add("is-correct");
      else if (answered && state.choice === i) button.classList.add("is-wrong");
      button.append(el("span", "option-letter", "ABCD"[i]), el("span", null, option));
      button.disabled = answered || working;
      button.addEventListener("click", () => {
        state.choice = i;
        quiz.answers.set(q.id, state);
        renderQuiz();
      });
      group.append(button);
    });
    card.append(group);
  } else if (q.kind === "tf") {
    const group = el("div", "tf-options");
    for (const [value, label] of [[true, tx("Verdadero")], [false, tx("Falso")]]) {
      const button = el("button", "option tf", label);
      button.type = "button";
      button.setAttribute("aria-pressed", String(state.choice === value));
      if (answered && state.feedback.truth === value) button.classList.add("is-correct");
      else if (answered && state.choice === value) button.classList.add("is-wrong");
      button.disabled = answered || working;
      button.addEventListener("click", () => {
        state.choice = value;
        quiz.answers.set(q.id, state);
        renderQuiz();
      });
      group.append(button);
    }
    card.append(group);
  } else {
    const label = el("label", "small", tx("Tu respuesta"));
    label.htmlFor = `answer-${q.id}`;
    const area = el("textarea", "answer-box");
    area.id = `answer-${q.id}`;
    area.rows = 4;
    area.maxLength = 2000;
    area.value = state.text;
    area.disabled = state.step !== "answering";
    area.addEventListener("input", () => {
      state.text = area.value;
      quiz.answers.set(q.id, state);
      if (action) action.disabled = !area.value.trim();
    });
    card.append(label, area);
  }

  if (state.step === "answering" || working) {
    const row = el("div", "row");
    action = el("button", "btn-primary", working ? tx("Calificando…") : actionLabel(q.kind));
    action.type = "button";
    action.disabled = working || (q.kind === "mc" || q.kind === "tf" ? state.choice === null : !state.text.trim());
    action.addEventListener("click", () => submitAnswer(q, state));
    const skip = el("button", "btn-ghost", tx("Saltar"));
    skip.type = "button";
    skip.disabled = working;
    skip.addEventListener("click", () => nextQuestion());
    row.append(action, skip);
    card.append(row);
  } else {
    card.append(feedbackNode(q, state));
  }
  if (state.error) card.append(el("p", "error", state.error));
  return card;
}

function actionLabel(kind) {
  return { mc: tx("Comprobar"), tf: tx("Comprobar"), open_self: tx("Mostrar respuesta"), open_model: tx("Calificar") }[kind];
}

async function postPractice(path, body) {
  const response = await postJSON(path, body);
  if (!response.ok) throw new Error(await errorText(response));
  return response.json();
}

async function submitAnswer(q, state) {
  const quiz = practice.quiz;
  state.error = null;
  state.step = "working";
  quiz.answers.set(q.id, state);
  renderQuiz();
  try {
    if (q.kind === "mc" || q.kind === "tf") {
      const data = await postPractice("/api/practice/check", { question_id: q.id, choice: state.choice });
      Object.assign(state, { step: "done", feedback: data, result: data.result });
    } else if (q.kind === "open_self") {
      const data = await postPractice("/api/practice/reveal", { question_id: q.id });
      Object.assign(state, { step: "self", feedback: data });
    } else {
      const data = await postPractice("/api/practice/grade", { question_id: q.id, answer: state.text });
      if (data.graded) {
        Object.assign(state, { step: "done", feedback: data, result: data.result, attempt: data.attempt_id, review: "pending" });
      } else {
        Object.assign(state, { step: "self", feedback: data, gradeFailed: true });
      }
    }
  } catch (error) {
    state.step = "answering";
    state.error = error.message;
  }
  renderQuiz();
}

async function selfGrade(q, state, result) {
  try {
    await postPractice("/api/practice/self", { question_id: q.id, answer: state.text, result });
    Object.assign(state, { step: "done", result, selfGraded: true });
  } catch (error) {
    state.error = error.message;
  }
  renderQuiz();
}

async function reviewGrade(state, result) {
  try {
    await postPractice("/api/practice/review", { attempt_id: state.attempt, result });
    state.review = result === state.feedback.result ? "agreed" : "changed";
    state.result = result;
  } catch (error) {
    state.error = error.message;
  }
  renderQuiz();
}

function choiceButtons(onPick, labels) {
  const row = el("div", "row");
  for (const [result, label] of labels) {
    const button = el("button", "btn", label);
    button.type = "button";
    button.addEventListener("click", () => onPick(result));
    row.append(button);
  }
  return row;
}

function feedbackNode(q, state) {
  const f = state.feedback;
  const box = el("div", "feedback");
  const result = state.result;
  if (state.step === "self") {
    box.classList.add("neutral");
    box.append(el("strong", null, state.gradeFailed ? tx("El modelo no pudo calificarla: compárala tú.") : tx("Compara con tu respuesta")));
  } else if (q.kind === "open_model") {
    box.classList.add(result === "correct" ? "good" : result === "partial" ? "mid" : "bad");
    box.append(el("strong", null, tx("{0}, según el modelo", RESULT_LABELS[result])));
    if (f.reason) box.append(el("p", null, f.reason));
  } else if (state.selfGraded) {
    box.classList.add("neutral");
    box.append(el("strong", null, tx("Te calificaste: {0}", RESULT_LABELS[result].toLowerCase())));
  } else {
    box.classList.add(result === "correct" ? "good" : "bad");
    box.append(el("strong", null, result === "correct" ? tx("Correcto") : tx("No es correcto")));
  }
  if (f.reference) {
    const ref = el("div", "reference");
    ref.append(el("span", "small muted", tx("Respuesta de referencia")));
    ref.append(plainText(f.reference));
    box.append(ref);
  }
  if (f.explanation) box.append(plainText(f.explanation));
  box.append(el("p", "small", tx("Fuente: {0} (líneas {1}-{2})", f.source.citation, f.source.lines[0], f.source.lines[1])));

  if (state.step === "self") {
    box.append(el("p", "small strong", tx("¿Cómo te fue?")));
    box.append(choiceButtons((r) => selfGrade(q, state, r), [["correct", tx("La sabía")], ["partial", tx("A medias")], ["wrong", tx("No la sabía")]]));
  } else if (q.kind === "open_model" && state.review === "pending") {
    box.append(el("p", "small strong", tx("¿Estás de acuerdo con la nota?")));
    const row = choiceButtons((r) => reviewGrade(state, r === "yes" ? f.result : r), [["yes", tx("Sí")]]);
    const change = el("button", "btn", tx("No, cámbiala"));
    change.type = "button";
    change.addEventListener("click", () => {
      state.review = "choosing";
      renderQuiz();
    });
    row.append(change);
    box.append(row);
  } else if (q.kind === "open_model" && state.review === "choosing") {
    box.append(el("p", "small strong", tx("¿Cuál es la nota correcta?")));
    box.append(choiceButtons((r) => reviewGrade(state, r), [["correct", tx("Correcta")], ["partial", tx("A medias")], ["wrong", tx("Incorrecta")]]));
  } else if (state.review === "changed") {
    box.append(el("p", "small muted", tx("Nota cambiada a «{0}». Esto mide cuánto acierta el modelo.", RESULT_LABELS[result].toLowerCase())));
  }

  const source = el("details", "source-details");
  source.append(el("summary", null, tx("Ver la sección de tus notas")));
  source.append(el("div", "source-text", f.source.text));
  const links = el("div", "row");
  links.append(readLink(f.source.doc_path, f.source.lines[0]));
  if (f.source.open_url) {
    const open = el("a", "btn", tx("Abrir en Obsidian"));
    open.href = f.source.open_url;
    links.append(open);
  }
  source.append(links);
  box.append(source);

  const actions = el("div", "row-between");
  const flag = el("button", "btn-ghost", state.flagged ? tx("Marcada: no se volverá a preguntar") : tx("La pregunta está mal"));
  flag.type = "button";
  flag.disabled = Boolean(state.flagged);
  flag.addEventListener("click", async () => {
    try {
      await postPractice("/api/practice/flag", { question_id: q.id });
      state.flagged = true;
    } catch (error) {
      state.error = error.message;
    }
    renderQuiz();
  });
  const pendingGrade = state.step === "self" || (q.kind === "open_model" && ["pending", "choosing"].includes(state.review));
  const next = el("button", "btn-dark", isLast() ? tx("Ver resultado") : tx("Siguiente"));
  next.type = "button";
  next.disabled = pendingGrade;
  next.addEventListener("click", nextQuestion);
  actions.append(flag, next);
  box.append(actions);
  return box;
}

function isLast() {
  const quiz = practice.quiz;
  return !quiz.generating && quiz.index >= quiz.questions.length - 1;
}

function nextQuestion() {
  const quiz = practice.quiz;
  if (isLast()) {
    finishQuiz();
    return;
  }
  quiz.index += 1;
  renderQuiz();
  $("practice-quiz").scrollIntoView({ block: "start" });
}

// --- results -------------------------------------------------------------------------------

function renderResults() {
  fetchToday().then(showTodayBadge); // answers may have cleared some of today's reviews
  const quiz = practice.quiz;
  const answered = quiz.questions.filter((q) => quiz.answers.get(q.id)?.result);
  const score = answered.reduce((sum, q) => sum + { correct: 1, partial: 0.5, wrong: 0 }[quiz.answers.get(q.id).result], 0);
  const summary = el("section", "practice-card results-summary");
  const big = el("div", "score");
  big.append(el("span", "score-number", `${formatScore(score)} de ${answered.length}`));
  big.append(el("span", "muted", quiz.label));
  summary.append(big);
  const failed = answered.filter((q) => quiz.answers.get(q.id).result !== "correct");
  summary.append(el("p", null, !answered.length
    ? tx("No respondiste ninguna pregunta.")
    : failed.length
      ? tx("Te conviene repasar {0}: aparecen en «Repasar mis fallos».", failed.length === 1 ? tx("una sección") : tx("{0} secciones", failed.length))
      : tx("Todas correctas.")));
  const list = el("ol", "results-list");
  for (const q of answered) {
    const state = quiz.answers.get(q.id);
    const li = el("li", `result-row ${state.result}`);
    li.append(el("span", "result-mark", RESULT_MARKS[state.result]));
    li.lastChild.setAttribute("aria-label", RESULT_LABELS[state.result]);
    const body = el("div");
    body.append(el("span", null, q.text));
    const how = q.kind === "open_model" ? tx(" · nota del modelo") : state.selfGraded ? tx(" · te calificaste tú") : "";
    body.append(el("span", "small muted", `${state.feedback.source.sections.at(-1) || state.feedback.source.guide}${how}`));
    li.append(body);
    list.append(li);
  }
  const actions = el("div", "row");
  const again = el("button", "btn-primary", tx("Volver a Practicar"));
  again.type = "button";
  again.addEventListener("click", () => {
    showScreen("setup");
    renderProgress();
    renderToday();
  });
  const ask = el("a", "btn", tx("Preguntar sobre este tema"));
  ask.href = "#/conversar";
  actions.append(again, ask);
  $("practice-results").replaceChildren(summary, ...(answered.length ? [list] : []), actions);
}

// --- wiring --------------------------------------------------------------------------------

function bindPracticar() {
  $("practice-form").addEventListener("submit", (event) => {
    event.preventDefault();
    try {
      const scope = selectedScope();
      const count = Number(document.querySelector('input[name="practice-count"]:checked').value);
      startQuiz(scope, count, "form");
    } catch (error) {
      showError("practice-error", error.message);
    }
  });
  const openForm = () => {
    showError("hub-error", "");
    showScreen("form");
    $("scope-guide").focus();
  };
  $("tile-quiz").addEventListener("click", openForm);
  $("tile-feynman").addEventListener("click", () => {
    showError("hub-error", "");
    showScreen("feynman-intro");
    $("feynman-concept").focus();
  });
  $("tile-review").addEventListener("click", () =>
    startQuiz({ kind: "review", value: "", label: tx("Repaso de fallos") }, Math.min(10, practice.toReview)),
  );
  for (const back of document.querySelectorAll("#page-practicar [data-back]")) {
    back.addEventListener("click", () => {
      showScreen("setup");
      renderProgress();
      renderToday();
    });
  }
  $("today-start").addEventListener("click", async () => {
    if ($("today-start").dataset.mode === "quiz") {
      openForm();
      return;
    }
    const data = await fetchToday();
    if (!data || !data.today) {
      renderToday();
      return;
    }
    startQuiz({ kind: "srs", value: "", label: tx("Repaso de hoy") }, data.today);
  });
  for (const kind of ["guide", "folder", "topic"]) {
    $(`scope-${kind}`).addEventListener("focus", () => {
      document.querySelector(`input[name="practice-scope"][value="${kind}"]`).checked = true;
    });
  }
}

let practicarStarted = false;

pageHooks.practicar = () => {
  if (!practicarStarted) {
    practicarStarted = true;
    bindPracticar();
    loadScopes();
  }
  const topic = location.hash.match(/[?&]tema=([^&]*)/); // from Conversar's «Practicar este tema»
  if (topic && (!practice.quiz || $("practice-quiz").hidden)) {
    document.querySelector('input[name="practice-scope"][value="topic"]').checked = true;
    $("scope-topic").value = decodeURIComponent(topic[1]);
    showScreen("form");
  }
  const pending = app.pendingPractice; // from Mapas' «Practicar esto» (ADR-026)
  app.pendingPractice = null;
  if (pending && (!practice.quiz || !practice.quiz.generating)) {
    startQuiz(pending.scope, pending.count);
    return;
  }
  if (!$("practice-setup").hidden) {
    renderProgress();
    renderToday();
  }
};

fetchToday().then(showTodayBadge); // the nav badge shows on every page
