#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from unittest.mock import AsyncMock

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

    monkeypatch.setattr("pipecat.services.cartesia.stt.websocket_connect", fake_websocket_connect)

    service = CartesiaSTTService(api_key="test-key", sample_rate=16000)
    service._websocket = _FakeWebsocket(state=State.CLOSED)

    await service._connect_websocket()

    assert service._websocket is None


@pytest.mark.asyncio
async def test_cartesia_run_stt_logs_send_failure_without_clearing_websocket():
    service = CartesiaSTTService(api_key="test-key", sample_rate=16000)
    websocket = _FakeWebsocket(send_side_effect=RuntimeError("websocket closed"))
    service._websocket = websocket

    async for _ in service.run_stt(b"\x00" * 160):
        pass

    assert service._websocket is websocket
