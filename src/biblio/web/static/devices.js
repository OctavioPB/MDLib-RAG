// Biblioteca — LAN mode (ADR-035): on this PC, «Conectar un dispositivo» shows a one-time QR code
// and the paired devices; on a phone or tablet, links that only work on the PC are hidden.
// Uses the helpers of app.js ($, el, postJSON, errorText).
"use strict";

const lanState = { expires: null, timer: null };

function lanMinutesLeft() {
  if (!lanState.expires) return 0;
  return Math.max(0, Math.ceil((lanState.expires - Date.now()) / 60000));
}

function lanShowExpiry() {
  const left = lanMinutesLeft();
  $("lan-expiry").textContent = left
    ? tx("Vale para un solo dispositivo durante {0} {1}.", left, left === 1 ? tx("minuto") : tx("minutos"))
    : tx("El código caducó: genera otro.");
  $("lan-qr").classList.toggle("expired", !left);
}

async function lanNewCode() {
  $("lan-error").hidden = true;
  const response = await postJSON("/api/lan/pair", {});
  if (!response.ok) {
    $("lan-error").textContent = await errorText(response);
    $("lan-error").hidden = false;
    return;
  }
  const data = await response.json();
  lanState.expires = new Date(data.expires).getTime();
  $("lan-qr").src = `/api/lan/qr.svg?link=${encodeURIComponent(data.link)}`;
  $("lan-link").textContent = data.link;
  lanShowExpiry();
  clearInterval(lanState.timer);
  lanState.timer = setInterval(lanShowExpiry, 15000);
}

const LAN_MONTHS = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"];

function lanDate(iso) {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "" : `${d.getDate()} ${LAN_MONTHS[d.getMonth()]}`;
}

async function lanLoadDevices() {
  let data;
  try {
    data = await (await fetch("/api/lan")).json();
  } catch {
    return;
  }
  $("lan-address").textContent = data.url || tx("este PC no tiene dirección en una red privada");
  const list = $("lan-devices");
  if (!data.devices.length) {
    list.replaceChildren(el("li", "muted", tx("Ninguno todavía.")));
    return;
  }
  list.replaceChildren(
    ...data.devices.map((d) => {
      const li = el("li", "lan-device");
      const text = el("div");
      text.append(el("strong", null, d.name), el("span", "small muted", tx("Emparejado el {0} · último uso {1}", lanDate(d.created), lanDate(d.last_seen))));
      const remove = el("button", "btn danger-text", tx("Quitar"));
      remove.type = "button";
      remove.addEventListener("click", async () => {
        if (!confirm(tx("¿Quitar «{0}»? Tendrá que volver a escanear un código para entrar.", d.name))) return;
        await postJSON("/api/lan/revoke", { id: d.id });
        lanLoadDevices();
      });
      li.append(text, remove);
      return li;
    }),
  );
}

function lanOpen() {
  $("lan-dialog").showModal();
  lanNewCode();
  lanLoadDevices();
}

async function lanInit() {
  let status;
  try {
    status = await (await fetch("/api/status")).json();
  } catch {
    return;
  }
  const lan = status.lan || {};
  document.body.classList.toggle("remote-device", !lan.local);
  $("lan-connect").hidden = !(lan.enabled && lan.local);
  $("lan-connect").addEventListener("click", lanOpen);
  $("lan-new").addEventListener("click", lanNewCode);
  $("lan-close").addEventListener("click", () => $("lan-dialog").close());
  $("lan-dialog").addEventListener("close", () => clearInterval(lanState.timer));
}

lanInit();
