"""LAN mode (ADR-035): who may use the web UI from another device on the home network.

Off by default. When on, the server listens on every interface, but only loopback and private
addresses are answered, and a device that is not this PC needs a token it got by opening a
one-time pairing link shown on the PC. Only the tokens' SHA-256 are kept, in `web.devices`.
"""

from __future__ import annotations

import hashlib
import io
import ipaddress
import json
import secrets
import socket
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import segno

COOKIE = "biblio_device"
PAIR_PATH = "/pair"


def is_loopback(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def is_home_network(ip: str) -> bool:
    """Loopback, private (RFC 1918, IPv6 unique-local) or link-local: the networks LAN mode
    answers. A public address is never answered, whatever else it sends."""
    try:
        addr = ipaddress.ip_address(ip.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return (addr.is_private or addr.is_loopback or addr.is_link_local) and not (
        addr.is_multicast or addr.is_unspecified
    )


def lan_address() -> str | None:
    """This PC's address on the home network, for the pairing link. A UDP socket «connected» to a
    private address picks the interface the system would use; UDP connect sends no packet."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.254.254.254", 9))
            ip = str(probe.getsockname()[0])
    except OSError:
        return None
    return ip if is_home_network(ip) and not is_loopback(ip) else None


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _now() -> datetime:
    return datetime.now().astimezone()


@dataclass(frozen=True)
class Device:
    id: str
    name: str
    created: str
    last_seen: str
    expires: str


@dataclass(frozen=True)
class Access:
    allowed: bool
    reason: str = ""  # why not, for the 403 page


class LanAccess:
    """Pairing codes (in memory, one use, short-lived) and paired devices (on disk, hashed)."""

    def __init__(self, path: Path, pairing_minutes: int, device_days: int) -> None:
        self._path = path
        self._pairing = timedelta(minutes=pairing_minutes)
        self._lifetime = timedelta(days=device_days)
        self._lock = threading.Lock()
        self._codes: dict[str, datetime] = {}  # code hash → expiry
        self._devices: list[dict[str, str]] = self._load()

    def _load(self) -> list[dict[str, str]]:
        if not self._path.is_file():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._devices, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self._path)

    # --- pairing ---------------------------------------------------------------------------

    def new_code(self) -> tuple[str, datetime]:
        code = secrets.token_urlsafe(16)  # 128 bits
        expires = _now() + self._pairing
        with self._lock:
            now = _now()
            self._codes = {h: e for h, e in self._codes.items() if e > now}
            self._codes[_hash(code)] = expires
        return code, expires

    def redeem(self, code: str, name: str) -> str | None:
        """A device token for a valid, unused, unexpired code; the code is spent either way."""
        with self._lock:
            expires = self._codes.pop(_hash(code), None)
            if expires is None or expires <= _now():
                return None
            token = secrets.token_urlsafe(32)
            now = _now()
            self._devices.append(
                {
                    "id": secrets.token_hex(6),
                    "token_sha256": _hash(token),
                    "name": name[:120] or "Dispositivo",
                    "created": now.isoformat(timespec="seconds"),
                    "last_seen": now.isoformat(timespec="seconds"),
                    "expires": (now + self._lifetime).isoformat(timespec="seconds"),
                }
            )
            self._save()
        return token

    # --- devices ---------------------------------------------------------------------------

    def check(self, token: str | None) -> bool:
        if not token:
            return False
        digest = _hash(token)
        with self._lock:
            now = _now()
            for d in self._devices:
                if secrets.compare_digest(d.get("token_sha256", ""), digest):
                    if datetime.fromisoformat(d["expires"]) <= now:
                        return False
                    if now - datetime.fromisoformat(d["last_seen"]) > timedelta(minutes=10):
                        d["last_seen"] = now.isoformat(timespec="seconds")
                        self._save()
                    return True
        return False

    def devices(self) -> list[Device]:
        with self._lock:
            return [
                Device(d["id"], d["name"], d["created"], d["last_seen"], d["expires"])
                for d in self._devices
            ]

    def revoke(self, device_id: str) -> bool:
        with self._lock:
            kept = [d for d in self._devices if d.get("id") != device_id]
            if len(kept) == len(self._devices):
                return False
            self._devices = kept
            self._save()
        return True


def decide(
    client_ip: str, host: str, local_ip: str, port: int, path: str, token_ok: bool
) -> Access:
    """Whether a request is answered in LAN mode. Pure, so every rule is tested."""
    if not is_home_network(client_ip):
        return Access(False, "Solo se atiende a dispositivos de tu red local.")
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}
    if local_ip:
        allowed_hosts.add(f"[{local_ip}]:{port}" if ":" in local_ip else f"{local_ip}:{port}")
    if host not in allowed_hosts:
        return Access(False, "forbidden host")
    if is_loopback(client_ip):
        return Access(True)
    if path.split("?", 1)[0] == PAIR_PATH:
        return Access(True)  # the pairing link itself checks its code
    if not token_ok:
        return Access(False, "Este dispositivo no está emparejado.")
    return Access(True)


def device_name(user_agent: str) -> str:
    """A short label from the browser's User-Agent, for the list of devices."""
    ua = user_agent.lower()
    if "ipad" in ua:
        return "iPad"
    if "iphone" in ua:
        return "iPhone"
    if "android" in ua:
        return "Teléfono Android" if "mobile" in ua else "Tableta Android"
    for key, label in (("macintosh", "Mac"), ("windows", "Windows"), ("linux", "Linux")):
        if key in ua:
            return label
    return "Dispositivo"


def qr_svg(text: str) -> str:
    """A standalone SVG document (with its namespace, so it works as an <img>)."""
    out = io.BytesIO()
    segno.make(text, error="m").save(
        out, kind="svg", scale=6, border=2, dark="#111111", light="#ffffff", xmldecl=False
    )
    return out.getvalue().decode("utf-8")


def qr_text(text: str) -> str:
    """The QR code for a terminal, with half-block characters (two modules per line)."""
    qr = segno.make(text, error="m")
    rows = [list(r) for r in qr.matrix_iter(scale=1, border=2)]
    if len(rows) % 2:
        rows.append([0] * len(rows[0]))
    lines = []
    for top, bottom in zip(rows[0::2], rows[1::2], strict=True):
        line = []
        for a, b in zip(top, bottom, strict=True):
            line.append({(1, 1): "█", (1, 0): "▀", (0, 1): "▄", (0, 0): " "}[(int(a), int(b))])
        lines.append("".join(line))
    return "\n".join(lines)
