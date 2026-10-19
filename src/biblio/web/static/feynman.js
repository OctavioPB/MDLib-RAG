// Biblioteca — «Explícamelo tú» in Practicar (ADR-027): explain a concept, compared with your notes.
// Uses the helpers of app.js ($, el, postJSON, errorText, streamEvents, showError, app) and
// showScreen of practicar.js. Library and model text is only ever inserted with textContent.
"use strict";

const FEYNMAN_GRADES = [["correct", "Bien"], ["partial", "A medias"], ["wrong", "Mal"]];
const COVERAGE_MARKS = { covered: ["✓", "correct"], partial: ["~", "partial"], missing: ["✗", "wrong"] };
const CLAIM_MARKS = { supported: ["✓", "correct"], contradicts: ["✗", "wrong"], unsupported: ["?", "partial"] };

const feynman = { session: null };

async function startFeynman(concept) {
  showError("feynman-error", "");
  if (feynman.session) feynman.session.controller.abort();
  $("feynman-concept").value = concept;
  const s = {
    concept,
    id: null,
    notes: 0,
    error: null,
    text: "",
    checking: false,
    review: null, // {ideas, claims, claimsFailed, score, proposed, sources}
    votes: new Map(), // "idea:0" → true | false
    result: null,
    controller: new AbortController(),
  };
  feynman.session = s;
  showScreen("feynman");
  renderFeynman();
  try {
    await streamEvents(
      "/api/feynman/start",
      { concept },
      (name, data) => {
        if (name !== "ready") return;
        s.id = data.session_id;
        s.notes = data.notes;
      },
      s.controller.signal,
    );
  } catch (error) {
    if (error.name === "AbortError") return;
    s.error = error.message;
  }
  if (feynman.session === s) updateFeynmanControls();
}

function minChars() {
  return 40; // as feynman.min_explanation_chars; the server says so if it changes
}

function renderFeynman() {
  const s = feynman.session;
  const card = el("section", "practice-card question-card");
  const tags = el("div", "row");
  tags.append(el("span", "tag", "Explícamelo tú"), el("span", "badge warn", "Experimental"));
  card.append(tags);
  card.append(el("h1", "question-text", s.concept));
  card.append(el("p", "muted", "Explícalo con tus palabras, como si se lo contaras a alguien que no sabe nada del tema. Después lo comparo con tus notas."));
  const area = el("textarea", "answer-box");
  area.id = "feynman-text";
  area.rows = 9;
  area.maxLength = 4000;
  area.placeholder = "Escribe aquí tu explicación…";
  area.setAttribute("aria-label", "Tu explicación");
  area.value = s.text;
  area.addEventListener("input", () => {
    s.text = area.value;
    updateFeynmanControls();
  });
  card.append(area);
  const status = el("p", "small muted");
  status.id = "feynman-status";
  status.setAttribute("aria-live", "polite");
  card.append(status);
  const row = el("div", "row");
  const check = el("button", "btn-primary btn-big", "Comprobar");
  check.id = "feynman-check";
  check.type = "button";
  check.addEventListener("click", checkExplanation);
  const back = el("button", "btn", "Volver");
  back.type = "button";
  back.addEventListener("click", leaveFeynman);
  row.append(check, back);
  card.append(row);
  const review = el("div", "feynman-review");
  review.id = "feynman-review";
  $("practice-feynman").replaceChildren(card, review);
  updateFeynmanControls();
  renderReview();
  area.focus();
}

function updateFeynmanControls() {
  const s = feynman.session;
  if (!s || !$("feynman-status")) return;
  const length = s.text.trim().length;
  let status;
  if (s.error) status = s.error;
  else if (s.checking) status = "Comparando tu explicación con tus notas…";
  else if (!s.id) status = "Mientras escribes, leo lo que dicen tus notas…";
  else if (s.review) status = s.result ? "Guardado en tu práctica." : "";
  else if (length < minChars()) status = `Listo: lo compararé con ${s.notes} ${s.notes === 1 ? "sección" : "secciones"} de tus notas. Escribe al menos ${minChars()} caracteres.`;
  else status = `Listo: lo compararé con ${s.notes} ${s.notes === 1 ? "sección" : "secciones"} de tus notas.`;
  $("feynman-status").textContent = status;
  $("feynman-status").hidden = !status;
  $("feynman-status").classList.toggle("error", Boolean(s.error));
  const area = $("feynman-text");
  area.readOnly = s.checking || Boolean(s.review && s.review.proposed);
  const check = $("feynman-check");
  check.hidden = Boolean(s.review && s.review.proposed);
  check.disabled = !s.id || s.checking || length < minChars() || Boolean(s.error && !s.id);
}

function leaveFeynman() {
  const s = feynman.session;
  if (s) s.controller.abort();
  feynman.session = null;
  showScreen("setup");
  renderProgress();
  renderToday();
}

async function checkExplanation() {
  const s = feynman.session;
  s.error = null;
  s.checking = true;
  s.review = { ideas: null, claims: null };
  s.votes = new Map();
  updateFeynmanControls();
  renderReview();
  try {
    await streamEvents(
      "/api/feynman/check",
      { session_id: s.id, explanation: s.text },
      (name, data) => {
        if (name === "coverage") s.review.ideas = data.ideas;
        else if (name === "claims") {
          s.review.claims = data.claims;
          s.review.claimsFailed = data.failed;
        } else if (name === "done") Object.assign(s.review, data);
        if (feynman.session === s) renderReview();
      },
      s.controller.signal,
    );
  } catch (error) {
    if (error.name === "AbortError") return;
    s.error = error.message;
    s.review = null;
  } finally {
    s.checking = false;
    if (feynman.session === s) {
      updateFeynmanControls();
      renderReview();
    }
  }
}

function citationOf(review, number) {
  const source = (review.sources || []).find((f) => f.number === number);
  return source ? source.citation : `fragmento ${number}`;
}

function voteControl(s, point) {
  const box = el("span", "vote");
  box.append(el("span", "small muted", "¿Es justo?"));
  for (const [fair, label] of [[true, "Sí"], [false, "No"]]) {
    const button = el("button", "btn-soft btn-small", label);
    button.type = "button";
    button.setAttribute("aria-pressed", String(s.votes.get(point) === fair));
    button.addEventListener("click", async () => {
      const response = await postJSON("/api/feynman/vote", { session_id: s.id, point, fair });
      if (!response.ok) return;
      s.votes.set(point, fair);
      for (const b of box.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b === button));
    });
    box.append(button);
  }
  return box;
}

function pointRow(mark, title, lines, vote) {
  const li = el("li", `result-row ${mark[1]}`);
  li.append(el("span", "result-mark", mark[0]));
  const body = el("div");
  body.append(el("strong", null, title));
  for (const [cls, text] of lines) body.append(el("span", cls, text));
  if (vote) body.append(vote);
  li.append(body);
  return li;
}

function renderReview() {
  const s = feynman.session;
  const box = $("feynman-review");
  if (!s || !box) return;
  const r = s.review;
  if (!r) {
    box.replaceChildren();
    return;
  }
  const nodes = [];
  if (r.proposed) {
    const covered = r.ideas.filter((i) => i.verdict === "covered").length;
    const partial = r.ideas.filter((i) => i.verdict === "partial").length;
    const summary = el("div", `feedback ${{ correct: "good", partial: "mid", wrong: "bad" }[r.proposed]}`);
    summary.append(el("strong", null, `Cubriste ${covered} de ${r.ideas.length} ideas clave${partial ? ` y ${partial} a medias` : ""}`));
    const bar = el("div", "bar");
    const fill = el("div", "bar-fill");
    fill.style.width = `${Math.round(r.score * 100)}%`;
    bar.append(fill);
    summary.append(bar);
    nodes.push(summary);
  }
  if (r.ideas) {
    nodes.push(el("h2", "feynman-heading", "Ideas clave de tus notas"));
    const list = el("ul", "results-list");
    r.ideas.forEach((idea, i) => {
      const said = idea.phrase ? ["small", `Lo dijiste: «${idea.phrase}»`] : ["small", "No lo encontré en tu explicación."];
      const source = ["small muted", `${citationOf(r, idea.fragment)}: «${idea.quote}»`];
      list.append(pointRow(COVERAGE_MARKS[idea.verdict], idea.text, [said, source], r.proposed ? voteControl(s, `idea:${i}`) : null));
    });
    nodes.push(list);
  } else if (s.checking) {
    nodes.push(el("p", "muted", "Buscando las ideas clave en tu explicación…"));
  }
  if (r.claims) {
    nodes.push(el("h2", "feynman-heading", "Lo que afirmaste"));
    if (r.claimsFailed) nodes.push(el("p", "muted", "Esta vez no pude revisar tus afirmaciones."));
    else if (!r.claims.length) nodes.push(el("p", "muted", "No encontré afirmaciones que contrastar."));
    const list = el("ul", "results-list");
    r.claims.forEach((claim, i) => {
      const lines = {
        supported: [["small", "Tus notas lo respaldan."], ["small muted", `${citationOf(r, claim.fragment)}: «${claim.quote}»`]],
        contradicts: [["small", "Tus notas dicen otra cosa."], ["small muted", `${citationOf(r, claim.fragment)}: «${claim.quote}»`]],
        unsupported: [["small", "Tus notas no lo mencionan (no quiere decir que sea falso)."]],
      }[claim.verdict];
      list.append(pointRow(CLAIM_MARKS[claim.verdict], `«${claim.text}»`, lines, r.proposed ? voteControl(s, `claim:${i}`) : null));
    });
    nodes.push(list);
  } else if (r.ideas && s.checking) {
    nodes.push(el("p", "muted", "Revisando lo que afirmaste…"));
  }
  if (r.proposed) nodes.push(gradeCard(s));
  if (r.sources) {
    nodes.push(el("h2", "feynman-heading", "Tus notas"));
    for (const f of r.sources) {
      const item = el("details", "map-source");
      item.append(el("summary", null, `[${f.number}] ${f.citation}`));
      item.append(el("div", "source-text", f.text));
      if (f.open_url) {
        const open = el("a", "btn-soft", "Abrir en Obsidian");
        open.href = f.open_url;
        item.append(open);
      }
      nodes.push(item);
    }
  }
  box.replaceChildren(...nodes);
}

function gradeCard(s) {
  const card = el("div", "feedback neutral");
  const proposed = Object.fromEntries(FEYNMAN_GRADES)[s.review.proposed];
  if (s.result) {
    card.append(el("strong", null, `Guardado: ${Object.fromEntries(FEYNMAN_GRADES)[s.result].toLowerCase()}`));
    card.append(el("p", "small", "Cuenta para tus repasos y para el dominio de los mapas."));
  } else {
    card.append(el("strong", null, `Nota propuesta: ${proposed.toLowerCase()}`));
    card.append(el("p", "small", "Acéptala o cámbiala; se guarda como práctica de las secciones de tus notas que tratan esto."));
    const row = el("div", "row");
    for (const [result, label] of FEYNMAN_GRADES) {
      const button = el("button", result === s.review.proposed ? "btn-primary" : "btn", label);
      button.type = "button";
      button.addEventListener("click", async () => {
        const response = await postJSON("/api/feynman/confirm", { session_id: s.id, result });
        if (!response.ok) {
          s.error = await errorText(response);
          updateFeynmanControls();
          return;
        }
        s.result = result;
        updateFeynmanControls();
        renderReview();
        fetchToday().then(showTodayBadge);
      });
      row.append(button);
    }
    card.append(row);
  }
  const more = el("div", "row");
  if (!s.result) {
    const retry = el("button", "btn", "Corregir y comprobar otra vez");
    retry.type = "button";
    retry.addEventListener("click", () => {
      s.review = null;
      updateFeynmanControls();
      renderReview();
      $("feynman-text").focus();
    });
    more.append(retry);
  }
  const another = el("button", "btn", "Explicar otro concepto");
  another.type = "button";
  another.addEventListener("click", leaveFeynman);
  more.append(another);
  card.append(more);
  return card;
}

function bindFeynman() {
  $("feynman-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const concept = $("feynman-concept").value.trim();
    if (!concept) {
      showError("feynman-error", "Escribe el concepto que vas a explicar.");
      return;
    }
    await loadStatus();
    if (!app.llmRunning) {
      showError("feynman-error", "El modelo de lenguaje no está encendido. Inícialo con scripts/serve-llm.ps1.");
      return;
    }
    startFeynman(concept);
  });
}

let feynmanBound = false;
const practicarHook = pageHooks.practicar;

pageHooks.practicar = () => {
  if (!feynmanBound) {
    feynmanBound = true;
    bindFeynman();
  }
  const asked = location.hash.match(/[?&]explicar=([^&]*)/); // from Mapas' «Explícamelo tú»
  if (asked) {
    const concept = decodeURIComponent(asked[1]);
    history.replaceState(null, "", "#/practicar"); // a reload does not start it again
    practicarHook();
    $("feynman-concept").value = concept;
    startFeynman(concept);
    return;
  }
  practicarHook();
  if (feynman.session && !$("practice-feynman").hidden) updateFeynmanControls();
};
