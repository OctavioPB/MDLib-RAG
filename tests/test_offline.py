from __future__ import annotations

import os
import socket

import pytest
from pytest_socket import SocketBlockedError, SocketConnectBlockedError

from biblio.offline import (
    OFFLINE_ENV,
    NetworkPolicyError,
    allow_model_downloads,
    apply_offline_env,
    is_loopback_url,
    missing_offline_env,
    require_loopback,
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in OFFLINE_ENV:
        monkeypatch.delenv(key, raising=False)


def test_external_connections_are_blocked() -> None:
    # An IP literal, so the test does not depend on DNS.
    with pytest.raises((SocketBlockedError, SocketConnectBlockedError)):
        socket.create_connection(("1.1.1.1", 443), timeout=1)


@pytest.mark.usefixtures("clean_env")
def test_apply_offline_env_sets_every_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    assert set(missing_offline_env()) == set(OFFLINE_ENV)
    apply_offline_env()
    assert missing_offline_env() == []
    for key in OFFLINE_ENV:
        monkeypatch.delenv(key)  # undo for other tests


@pytest.mark.usefixtures("clean_env")
def test_allow_model_downloads_keeps_telemetry_off(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_offline_env()
    allow_model_downloads()
    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert os.environ["DO_NOT_TRACK"] == "1"
    for key in OFFLINE_ENV:
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:11434/v1",
        "http://localhost:8080/v1",
        "http://LOCALHOST/v1",
        "http://[::1]:8080/v1",
        "http://127.0.0.2:8080",
    ],
)
def test_loopback_urls_are_accepted(url: str) -> None:
    assert is_loopback_url(url)
    require_loopback(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://api.openai.com/v1",
        "http://192.168.1.10:11434/v1",
        "http://0.0.0.0:11434/v1",
        "http://localhost.evil.com/v1",
        "http://127.0.0.1.nip.io/v1",
        "not a url",
        "",
    ],
)
def test_non_loopback_urls_are_rejected(url: str) -> None:
    assert not is_loopback_url(url)
    with pytest.raises(NetworkPolicyError):
        require_loopback(url)
