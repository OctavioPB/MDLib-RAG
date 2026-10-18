// Biblioteca — the chosen colour theme, applied before the page is drawn so it never flashes the
// other one. Loaded without `defer` on purpose; app.js holds the switch. "system" follows the OS.
"use strict";
(() => {
  let theme = "system";
  try {
    theme = JSON.parse(localStorage.getItem("biblio.theme")) || "system";
  } catch {
    /* storage blocked: follow the system */
  }
  if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;
})();
