// Biblioteca — the interface's language (ADR-046). The Spanish text is the key: `tx("Guardar")` gives
// «Save» in English and «Guardar» otherwise. The English dictionary is I18N_EN (i18n-en.js, loaded just
// before this file). Generated content (answers, maps, quizzes…) is never translated.
"use strict";

const I18N = {
  lang: (() => {
    try {
      return JSON.parse(localStorage.getItem("biblio.lang") || '"es"') === "en" ? "en" : "es";
    } catch {
      return "es";
    }
  })(),
  patterns: null, // keys with {0}…, as regular expressions, for server messages
};

/** The interface text in the current language; `{0}`, `{1}`… are replaced by the values. */
function tx(text, ...values) {
  const out = I18N.lang === "en" && Object.hasOwn(I18N_EN, text) ? I18N_EN[text] : text;
  return values.length ? out.replace(/\{(\d+)\}/g, (all, i) => (i < values.length ? String(values[i]) : all)) : out;
}

/** A message from the server: its exact text, else a dictionary key with {0}… that matches it. */
function txServer(message) {
  if (I18N.lang !== "en" || typeof message !== "string") return message;
  if (Object.hasOwn(I18N_EN, message)) return I18N_EN[message];
  if (!I18N.patterns) {
    const escape = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    I18N.patterns = Object.keys(I18N_EN)
      .filter((key) => /\{\d+\}/.test(key))
      .map((key) => [new RegExp(`^${escape(key).replace(/\\\{(\d+)\\\}/g, "(.+?)")}$`, "s"), key]);
  }
  for (const [pattern, key] of I18N.patterns) {
    const found = message.match(pattern);
    if (found) return tx(key, ...found.slice(1));
  }
  return message;
}

/** For dates and numbers. */
function i18nLocale() {
  return I18N.lang === "en" ? "en" : "es";
}

const I18N_ATTRIBUTES = ["placeholder", "title", "aria-label"];

/** Translate the static page: text nodes and the attributes people read, by exact text. */
function i18nApply(root) {
  if (I18N.lang !== "en") return;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) => (node.parentElement && node.parentElement.closest("script, style, svg, code, pre")
      ? NodeFilter.FILTER_REJECT
      : NodeFilter.FILTER_ACCEPT),
  });
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const text = node.nodeValue.replace(/\s+/g, " ").trim();
    if (text && Object.hasOwn(I18N_EN, text)) {
      const lead = node.nodeValue.match(/^\s*/)[0] ? " " : "";
      const trail = node.nodeValue.match(/\s*$/)[0] ? " " : "";
      node.nodeValue = `${lead}${I18N_EN[text]}${trail}`;
    }
  }
  for (const element of root.querySelectorAll(I18N_ATTRIBUTES.map((a) => `[${a}]`).join(","))) {
    for (const attribute of I18N_ATTRIBUTES) {
      const value = element.getAttribute(attribute);
      const text = value && value.replace(/\s+/g, " ").trim();
      if (text && Object.hasOwn(I18N_EN, text)) element.setAttribute(attribute, I18N_EN[text]);
    }
  }
}

// before every other script's DOMContentLoaded handler: this file is the first deferred one after
// the dictionary
document.addEventListener("DOMContentLoaded", () => {
  document.documentElement.lang = i18nLocale();
  document.title = tx(document.title);
  i18nApply(document.body);
});
