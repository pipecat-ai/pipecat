#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Closing the Flux websocket when the disconnect is cancelled."""

import asyncio

import pytest

from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService


class _RecordingWebsocket:
    """A websocket that records whether close() ran."""

    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


def _make_service(websocket):
    """Build a Flux service with only the state and hooks _disconnect_websocket uses."""
    service = DeepgramFluxSTTService.__new__(DeepgramFluxSTTService)
    service._name = "FakeFlux"
    service._websocket = websocket
    service._connection_established_event = asyncio.Event()
    service._configure_in_flight = False
    service._configure_sent_at = None
    service._configure_pending_fields = None
    service.errors = []
    service.events = []

    async def send_close_stream():
        pass

    async def push_error(error_msg, exception=None):
        service.errors.append(error_msg)

    async def call_event_handler(name, *args):
        service.events.append(name)

    service._send_close_stream = send_close_stream
    service.push_error = push_error
    service._call_event_handler = call_event_handler
    return service


@pytest.mark.asyncio
async def test_cancelled_disconnect_closes_the_websocket():
    """A cancelled disconnect still closes the socket it was about to drop."""
    websocket = _RecordingWebsocket()
    service = _make_service(websocket)
    entered = asyncio.Event()

    async def hang():
        entered.set()
        await asyncio.Event().wait()

    service.stop_all_metrics = hang

    task = asyncio.create_task(service._disconnect_websocket())
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert websocket.closed
    assert service._websocket is None
    assert service.events == ["on_disconnected"]
