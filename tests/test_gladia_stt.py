#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from unittest.mock import AsyncMock

import pytest
from websockets.protocol import State

from pipecat.frames.frames import EndFrame
from pipecat.services.gladia.stt import GladiaSTTService


@pytest.mark.asyncio
async def test_stop_sends_stop_recording_before_disconnect():
    service = GladiaSTTService(api_key="test-key")
    events = []

    service._send_stop_recording = AsyncMock(side_effect=lambda: events.append("send_stop"))
    service._disconnect = AsyncMock(side_effect=lambda: events.append("disconnect"))

    await service.stop(EndFrame())

    assert events == ["send_stop", "disconnect"]


@pytest.mark.asyncio
async def test_stop_waits_for_the_receive_task_to_drain():
    service = GladiaSTTService(api_key="test-key")
    drained = False

    async def receive():
        nonlocal drained
        await asyncio.sleep(0)
        drained = True

    service._send_stop_recording = AsyncMock()
    service._disconnect = AsyncMock()
    service._receive_task = asyncio.create_task(receive())

    await service.stop(EndFrame())

    assert drained


@pytest.mark.asyncio
async def test_stop_gives_up_on_a_receive_task_that_never_finishes():
    service = GladiaSTTService(api_key="test-key")

    async def never():
        await asyncio.Event().wait()

    service._send_stop_recording = AsyncMock()
    service._disconnect = AsyncMock()
    task = asyncio.create_task(never())
    service._receive_task = task

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("pipecat.services.gladia.stt._FINAL_TRANSCRIPT_TIMEOUT", 0.05)
        await service.stop(EndFrame())

    service._disconnect.assert_awaited()
    task.cancel()


@pytest.mark.asyncio
async def test_send_stop_recording_is_dropped_once_the_socket_is_gone():
    """The guard silently swallows the message after _disconnect_websocket()
    sets self._websocket = None, which is why ordering matters here."""
    service = GladiaSTTService(api_key="test-key")
    sent = []

    class FakeWebsocket:
        state = State.OPEN

        async def send(self, payload):
            sent.append(payload)

    service._websocket = FakeWebsocket()
    await service._send_stop_recording()
    assert len(sent) == 1

    service._websocket = None
    await service._send_stop_recording()
    assert len(sent) == 1
