#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from unittest.mock import AsyncMock

import pytest
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response
from websockets.protocol import State

from pipecat.services.cartesia.stt import CartesiaSTTService
from pipecat.services.status import ServiceStatus
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


@pytest.mark.asyncio
async def test_cartesia_rejected_api_key_misconfigures_the_service(monkeypatch):
    async def fake_websocket_connect(*args, **kwargs):
        raise _websocket_rejection(401)

    monkeypatch.setattr("pipecat.services.cartesia.stt.websocket_connect", fake_websocket_connect)

    service = CartesiaSTTService(api_key="wrong-key", sample_rate=16000)

    await service._connect_websocket()

    assert service.status == ServiceStatus.MISCONFIGURED
    assert not service.status.is_usable


@pytest.mark.asyncio
async def test_cartesia_unavailable_service_stays_usable(monkeypatch):
    async def fake_websocket_connect(*args, **kwargs):
        raise _websocket_rejection(503)

    monkeypatch.setattr("pipecat.services.cartesia.stt.websocket_connect", fake_websocket_connect)

    service = CartesiaSTTService(api_key="test-key", sample_rate=16000)

    await service._connect_websocket()

    assert service.status.is_usable


@pytest.mark.asyncio
async def test_cartesia_classifies_the_rejection(monkeypatch):
    async def fake_websocket_connect(*args, **kwargs):
        raise _websocket_rejection(401)

    monkeypatch.setattr("pipecat.services.cartesia.stt.websocket_connect", fake_websocket_connect)

    service = CartesiaSTTService(api_key="wrong-key", sample_rate=16000)

    errors = []
    service.push_frame = AsyncMock(side_effect=lambda frame, *a, **kw: errors.append(frame))

    await service._connect_websocket()

    assert errors[0].category == ErrorCategory.AUTHENTICATION
