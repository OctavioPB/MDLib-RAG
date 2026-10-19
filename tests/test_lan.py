from __future__ import annotations

import http.client
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from test_web import engine  # noqa: F401  (the fixture is used by name)

from biblio.web.engine import Engine
from biblio.web.lan import LanAccess, decide, device_name, is_home_network, qr_svg, qr_text
from biblio.web.server import make_server

PORT = 8765
PC = "192.168.1.20"


@pytest.mark.parametrize(
    ("ip", "home"),
    [
        ("127.0.0.1", True),
        ("::1", True),
        ("192.168.1.34", True),
        ("10.0.0.7", True),
        ("172.20.3.4", True),
        ("fd12:3456::1", True),  # IPv6 unique-local
        ("fe80::1%12", True),  # link-local with a zone
        ("::ffff:192.168.1.34", True),  # IPv4-mapped
        ("8.8.8.8", False),
        ("2001:4860::8888", False),
        ("0.0.0.0", False),
        ("not an ip", False),
    ],
)
def test_only_the_home_network_is_answered(ip: str, home: bool) -> None:
    assert is_home_network(ip) is home


def test_access_rules() -> None:
    def ask(client: str, host: str = f"{PC}:{PORT}", path: str = "/", token: bool = False) -> str:
        access = decide(client, host, PC, PORT, path, token)
        return "ok" if access.allowed else access.reason

    assert ask("127.0.0.1", host=f"127.0.0.1:{PORT}") == "ok"  # this PC, as before
    assert ask("127.0.0.1", host=f"localhost:{PORT}") == "ok"
    assert ask("192.168.1.34") == "Este dispositivo no está emparejado."
    assert ask("192.168.1.34", token=True) == "ok"
    assert ask("192.168.1.34", path="/pair?code=abc") == "ok"  # the link checks its own code
    assert ask("192.168.1.34", path="/api/status") != "ok"
    assert ask("8.8.8.8", token=True).startswith("Solo se atiende")  # never a public address
    assert ask("192.168.1.34", host=f"evil.example:{PORT}", token=True) == "forbidden host"
    assert ask("192.168.1.34", host=f"{PC}:9999", token=True) == "forbidden host"


def test_pairing_codes_and_devices(tmp_path: Path) -> None:
    path = tmp_path / "devices.json"
    access = LanAccess(path, pairing_minutes=10, device_days=30)
    code, _ = access.new_code()
    token = access.redeem(code, "iPhone")
    assert token and access.check(token)
    assert access.redeem(code, "otro") is None  # one use
    assert not access.check("a made-up token") and not access.check(None)
    assert token not in path.read_text(encoding="utf-8")  # only its hash is kept
    again = LanAccess(path, 10, 30)  # survives a restart
    assert again.check(token)
    (device,) = again.devices()
    assert device.name == "iPhone"
    assert again.revoke(device.id) and not again.check(token) and not again.revoke(device.id)

    expired = LanAccess(tmp_path / "x.json", pairing_minutes=0, device_days=30)
    old, _ = expired.new_code()
    assert expired.redeem(old, "tarde") is None
    short = LanAccess(tmp_path / "y.json", pairing_minutes=10, device_days=0)
    c, _ = short.new_code()
    assert not short.check(short.redeem(c, "x"))  # a device whose days are over


def test_labels_and_qr() -> None:
    assert device_name("Mozilla/5.0 (iPhone; CPU iPhone OS 17_0)") == "iPhone"
    assert device_name("Mozilla/5.0 (Linux; Android 14; SM-X710)") == "Tableta Android"
    assert device_name("Mozilla/5.0 (Linux; Android 14; Pixel 8) Mobile") == "Teléfono Android"
    link = f"http://{PC}:{PORT}/pair?code=abc"
    svg = qr_svg(link)
    assert svg.startswith("<svg") and 'xmlns="http://www.w3.org/2000/svg"' in svg.split(">", 1)[0]
    lines = qr_text(link).splitlines()
    assert len(lines) > 10 and len({len(line) for line in lines}) == 1


# --- the server in LAN mode, reached from this PC (loopback) ---------------------------------


@pytest.fixture
def lan_server(engine: Engine, tmp_path: Path) -> Iterator[tuple[str, int, LanAccess]]:  # noqa: F811
    access = LanAccess(tmp_path / "devices.json", 10, 30)
    srv = make_server(engine, "127.0.0.1", 0, access, PC, lan_bind="127.0.0.1")
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield "127.0.0.1", srv.server_address[1], access
    srv.shutdown()
    srv.server_close()


def _req(
    server: tuple[str, int, Any],
    method: str,
    path: str,
    body: Any = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection(server[0], server[1], timeout=10)
    data = json.dumps(body).encode() if body is not None else None
    hdrs = {"Content-Type": "application/json"} if data is not None else {}
    conn.request(method, path, body=data, headers={**hdrs, **(headers or {})})
    resp = conn.getresponse()
    payload = resp.read()
    conn.close()
    return resp.status, {k.lower(): v for k, v in resp.getheaders()}, payload


def test_the_pc_manages_pairing(lan_server: tuple[str, int, LanAccess]) -> None:
    status, _, raw = _req(lan_server, "GET", "/api/status")
    assert status == 200 and json.loads(raw)["lan"] == {"enabled": True, "local": True}
    _, _, raw = _req(lan_server, "GET", "/api/lan")
    info = json.loads(raw)
    assert (
        info["enabled"] and info["url"] == f"http://{PC}:{lan_server[1]}" and info["devices"] == []
    )

    status, _, raw = _req(lan_server, "POST", "/api/lan/pair", {})
    link = json.loads(raw)["link"]
    assert status == 200 and link.startswith(f"http://{PC}:{lan_server[1]}/pair?code=")
    status, headers, svg = _req(lan_server, "GET", f"/api/lan/qr.svg?link={link}")
    assert status == 200 and headers["content-type"] == "image/svg+xml" and svg.startswith(b"<svg")
    assert _req(lan_server, "GET", "/api/lan/qr.svg?link=http://evil.example/")[0] == 404

    # the PC itself never needs to pair: the link just sends it home
    status, headers, _ = _req(lan_server, "GET", "/pair?code=whatever")
    assert status == 303 and headers["location"] == "/" and "set-cookie" not in headers

    code = link.split("code=", 1)[1]
    token = lan_server[2].redeem(code, "iPad")
    assert token
    _, _, raw = _req(lan_server, "GET", "/api/lan")
    (device,) = json.loads(raw)["devices"]
    assert device["name"] == "iPad"
    assert _req(lan_server, "POST", "/api/lan/revoke", {"id": device["id"]})[0] == 200
    assert not lan_server[2].check(token)


def test_cross_site_posts_are_refused(lan_server: tuple[str, int, LanAccess]) -> None:
    host = f"127.0.0.1:{lan_server[1]}"
    evil = {"Origin": "http://evil.example", "Host": host}
    assert _req(lan_server, "POST", "/api/lan/pair", {}, evil)[0] == 403
    same = {"Origin": f"http://{host}", "Host": host}
    assert _req(lan_server, "POST", "/api/lan/pair", {}, same)[0] == 200
    assert _req(lan_server, "GET", "/", headers={"Host": "evil.example"})[0] == 403
