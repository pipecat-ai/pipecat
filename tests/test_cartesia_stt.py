#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlparse

import pytest
from websockets.protocol import State

from pipecat.services.cartesia.stt import CartesiaSTTService


class _FakeWebsocket:
    def __init__(self, *, state=State.OPEN, send_side_effect=None):
        self.state = state
        self.send = AsyncMock(side_effect=send_side_effect)


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


@pytest.mark.asyncio
async def test_cartesia_connect_websocket_url_includes_keyterm(monkeypatch):
    captured = {}

    async def fake_websocket_connect(url, *, additional_headers):
        captured["url"] = url
        captured["headers"] = additional_headers
        return _FakeWebsocket()

    monkeypatch.setattr("pipecat.services.cartesia.stt.websocket_connect", fake_websocket_connect)

    service = CartesiaSTTService(
        api_key="test-key",
        sample_rate=16000,
        settings=CartesiaSTTService.Settings(keyterm=["Cartesia", "Ink Whisper"]),
    )
    # sample_rate is normally set from StartFrame; poke it directly since this
    # test calls _connect_websocket() without running a full pipeline.
    service._sample_rate = 16000

    await service._connect_websocket()

    parsed = urlparse(captured["url"])
    query = parse_qs(parsed.query)
    assert parsed.scheme == "wss"
    assert parsed.netloc == "api.cartesia.ai"
    assert parsed.path == "/stt/websocket"
    assert query["model"] == ["ink-whisper"]
    assert query["sample_rate"] == ["16000"]
    assert query["keyterm"] == ["Cartesia", "Ink Whisper"]
    assert captured["headers"]["X-API-Key"] == "test-key"


@pytest.mark.asyncio
async def test_cartesia_connect_websocket_url_omits_keyterm_when_not_set(monkeypatch):
    captured = {}

    async def fake_websocket_connect(url, *, additional_headers):
        captured["url"] = url
        return _FakeWebsocket()

    monkeypatch.setattr("pipecat.services.cartesia.stt.websocket_connect", fake_websocket_connect)

    service = CartesiaSTTService(api_key="test-key", sample_rate=16000)
    service._sample_rate = 16000

    await service._connect_websocket()

    query = parse_qs(urlparse(captured["url"]).query)
    assert "keyterm" not in query


@pytest.mark.asyncio
async def test_cartesia_run_stt_logs_send_failure_without_clearing_websocket():
    service = CartesiaSTTService(api_key="test-key", sample_rate=16000)
    websocket = _FakeWebsocket(send_side_effect=RuntimeError("websocket closed"))
    service._websocket = websocket

    async for _ in service.run_stt(b"\x00" * 160):
        pass

    assert service._websocket is websocket
