#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
from unittest.mock import AsyncMock

import pytest
from websockets.protocol import State

from pipecat.frames.frames import CancelFrame, EndFrame
from pipecat.services.gladia.stt import GladiaSTTService
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup


class _FakeWebsocket:
    def __init__(self, *, state=State.OPEN):
        self.state = state
        self.sent = []
        self.closed = False

    async def send(self, payload):
        self.sent.append(json.loads(payload))

    async def close(self):
        self.closed = True
        self.state = State.CLOSED

    def __aiter__(self):
        return self._iter_messages()

    async def _iter_messages(self):
        if False:
            yield None


def _connected_service():
    """Build a service holding an open fake socket, without touching the network."""
    service = GladiaSTTService(api_key="test-key")
    service._setup = frame_processor_setup(TaskManager())
    websocket = _FakeWebsocket()
    service._websocket = websocket
    service._connection_active = True
    return service, websocket


def _message_types(websocket):
    return [message["type"] for message in websocket.sent]


@pytest.mark.asyncio
async def test_stop_sends_stop_recording_while_the_socket_is_open():
    service, websocket = _connected_service()

    await service.stop(EndFrame())

    assert _message_types(websocket) == ["stop_recording"]
    assert websocket.closed


@pytest.mark.asyncio
async def test_stop_disconnects_once():
    service, _ = _connected_service()
    service._disconnect_websocket = AsyncMock(wraps=service._disconnect_websocket)

    await service.stop(EndFrame())

    service._disconnect_websocket.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancel_does_not_send_stop_recording():
    service, websocket = _connected_service()

    await service.cancel(CancelFrame())

    assert _message_types(websocket) == []
    assert websocket.closed
