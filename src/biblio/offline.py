"""Runtime network guards (ADR-002, `offline-guarantee` skill).

`apply_offline_env` must run before any ML library is imported: several of them read these variables
at import time.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlsplit

OFFLINE_ENV: dict[str, str] = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
}

# The subset that blocks downloads; `biblio models pull` lifts only these, never the telemetry ones.
DOWNLOAD_BLOCKING_ENV: tuple[str, ...] = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
)


class NetworkPolicyError(RuntimeError):
    """A configured endpoint would leave the machine."""


def apply_offline_env() -> None:
    os.environ.update(OFFLINE_ENV)


def allow_model_downloads() -> None:
    """Lift the download blocks for `biblio models pull`, the only networked command."""
    for key in DOWNLOAD_BLOCKING_ENV:
        os.environ.pop(key, None)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"


def missing_offline_env() -> list[str]:
    """Offline variables that are absent or set to a different value."""
    return [key for key, value in OFFLINE_ENV.items() if os.environ.get(key) != value]


def is_loopback_url(url: str) -> bool:
    """True if the URL's host is `localhost` or a loopback IP literal.

    Hostnames are not resolved: DNS resolution would itself be a network call.
    """
    host = urlsplit(url).hostname
    if host is None:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def require_loopback(url: str) -> None:
    if not is_loopback_url(url):
        raise NetworkPolicyError(
            f"Endpoint {url!r} is not on localhost; only loopback endpoints are allowed (ADR-002)."
        )


def loopback_port_open(url: str, timeout_s: float) -> bool:
    """True if something accepts TCP connections at a loopback URL's port.

    Refuses non-loopback URLs, so it can never open a connection that leaves the machine.
    """
    require_loopback(url)
    parts = urlsplit(url)
    host = parts.hostname or "localhost"
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False
