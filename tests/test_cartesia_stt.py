#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlparse

import pytest
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response
from websockets.protocol import State

from pipecat.services.cartesia.stt import CartesiaSTTService
from pipecat.utils.errors import ErrorCategory


class _FakeWebsocket:
    def __init__(self, *, state=State.OPEN, send_side_effect=None):
        self.state = state
        self.send = AsyncMock(side_effect=send_side_effect)


def _websocket_rejection(status_code: int) -> InvalidStatus:
    """Build the exception `websockets` raises when a handshake is rejected."""
    return InvalidStatus(Response(status_code, "", Headers()))


@pytest.mark.asyncio
async def test_cartesia_connect_failure_clears_stale_websocket(monkeypatch):
    async def fake_websocket_connect(*args, **kwargs):
        raise RuntimeError("connection failed")

    monkeypatch.setattr(
        "pipecat.services.websocket_service.websocket_connect", fake_websocket_connect
    )

    service = CartesiaSTTService(api_key="test-key", sample_rate=16000)
    service._websocket = _FakeWebsocket(state=State.CLOSED)

    await service._connect_websocket()

    assert service._websocket is None


def _capture_connect_url(monkeypatch) -> dict:
    """Stub out the websocket connect and capture the URL it's called with."""
    captured = {}

    async def fake_websocket_connect(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs.get("additional_headers")
        return _FakeWebsocket()

    monkeypatch.setattr(
        "pipecat.services.websocket_service.websocket_connect", fake_websocket_connect
    )
    return captured


def _connected_service(**kwargs) -> CartesiaSTTService:
    service = CartesiaSTTService(api_key="test-key", sample_rate=16000, **kwargs)
    # sample_rate is normally set from StartFrame, which these tests skip.
    service._sample_rate = 16000
    return service


@pytest.mark.asyncio
async def test_cartesia_connect_websocket_url_includes_keyterm(monkeypatch):
    captured = _capture_connect_url(monkeypatch)

    service = _connected_service(
        settings=CartesiaSTTService.Settings(model="ink-2", keyterm=["Cartesia", "Ink 2"]),
    )

    await service._connect_websocket()

    parsed = urlparse(captured["url"])
    query = parse_qs(parsed.query)
    assert parsed.scheme == "wss"
    assert parsed.netloc == "api.cartesia.ai"
    assert parsed.path == "/stt/websocket"
    assert query["model"] == ["ink-2"]
    assert query["sample_rate"] == ["16000"]
    assert query["keyterm"] == ["Cartesia", "Ink 2"]
    assert captured["headers"]["X-API-Key"] == "test-key"


@pytest.mark.asyncio
async def test_cartesia_connect_websocket_url_encodes_keyterm_spaces_as_percent_20(monkeypatch):
    captured = _capture_connect_url(monkeypatch)

    service = _connected_service(
        settings=CartesiaSTTService.Settings(model="ink-2", keyterm=["Ink 2"]),
    )

    await service._connect_websocket()

    assert "keyterm=Ink%202" in captured["url"]


@pytest.mark.asyncio
async def test_cartesia_connect_websocket_url_omits_keyterm_when_not_set(monkeypatch):
    captured = _capture_connect_url(monkeypatch)

    service = _connected_service()

    await service._connect_websocket()

    query = parse_qs(urlparse(captured["url"]).query)
    assert "keyterm" not in query


@pytest.mark.asyncio
async def test_cartesia_connect_websocket_url_omits_keyterm_for_non_ink_2_model(monkeypatch):
    captured = _capture_connect_url(monkeypatch)

    service = _connected_service(
        settings=CartesiaSTTService.Settings(keyterm=["Cartesia"]),
    )

    await service._connect_websocket()

    query = parse_qs(urlparse(captured["url"]).query)
    assert query["model"] == ["ink-whisper"]
    assert "keyterm" not in query


@pytest.mark.asyncio
async def test_cartesia_connect_websocket_url_clamps_keyterms_to_limits(monkeypatch):
    captured = _capture_connect_url(monkeypatch)

    service = _connected_service(
        settings=CartesiaSTTService.Settings(
            model="ink-2",
            keyterm=[f"term{i}" for i in range(150)],
        ),
    )

    await service._connect_websocket()

    query = parse_qs(urlparse(captured["url"]).query)
    assert query["keyterm"] == [f"term{i}" for i in range(100)]


@pytest.mark.asyncio
async def test_cartesia_connect_websocket_url_clamps_keyterms_to_character_budget(monkeypatch):
    captured = _capture_connect_url(monkeypatch)

    service = _connected_service(
        settings=CartesiaSTTService.Settings(model="ink-2", keyterm=["a" * 700, "b" * 700, "c"]),
    )

    await service._connect_websocket()

    query = parse_qs(urlparse(captured["url"]).query)
    assert query["keyterm"] == ["a" * 700]


@pytest.mark.asyncio
async def test_cartesia_update_keyterm_reconnects(monkeypatch):
    _capture_connect_url(monkeypatch)

    service = _connected_service(
        settings=CartesiaSTTService.Settings(model="ink-2", keyterm=["Cartesia"]),
    )
    reconnect = AsyncMock()
    monkeypatch.setattr(service, "_request_reconnect", reconnect)

    await service._update_settings(CartesiaSTTService.Settings(keyterm=["Ink 2"]))

    assert service._settings.keyterm == ["Ink 2"]
    reconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_cartesia_run_stt_logs_send_failure_without_clearing_websocket():
    service = CartesiaSTTService(api_key="test-key", sample_rate=16000)
    websocket = _FakeWebsocket(send_side_effect=RuntimeError("websocket closed"))
    service._websocket = websocket

    async for _ in service.run_stt(b"\x00" * 160):
        pass

    assert service._websocket is websocket


@pytest.mark.asyncio
async def test_cartesia_rejected_api_key_makes_the_service_unusable(monkeypatch):
    async def fake_websocket_connect(*args, **kwargs):
        raise _websocket_rejection(401)

    monkeypatch.setattr(
        "pipecat.services.websocket_service.websocket_connect", fake_websocket_connect
    )

    service = CartesiaSTTService(api_key="wrong-key", sample_rate=16000)

    await service._connect_websocket()

    assert not service.is_usable


@pytest.mark.asyncio
async def test_cartesia_server_error_leaves_the_service_usable(monkeypatch):
    async def fake_websocket_connect(*args, **kwargs):
        raise _websocket_rejection(503)

    monkeypatch.setattr(
        "pipecat.services.websocket_service.websocket_connect", fake_websocket_connect
    )

    service = CartesiaSTTService(api_key="test-key", sample_rate=16000)

    await service._connect_websocket()

    assert service.is_usable


@pytest.mark.asyncio
async def test_cartesia_classifies_the_rejection(monkeypatch):
    async def fake_websocket_connect(*args, **kwargs):
        raise _websocket_rejection(401)

    monkeypatch.setattr(
        "pipecat.services.websocket_service.websocket_connect", fake_websocket_connect
    )

    service = CartesiaSTTService(api_key="wrong-key", sample_rate=16000)

    errors = []
    service.push_frame = AsyncMock(side_effect=lambda frame, *a, **kw: errors.append(frame))

    await service._connect_websocket()

    assert errors[0].category == ErrorCategory.AUTHENTICATION
