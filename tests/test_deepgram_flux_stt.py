#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import json
import unittest
from contextlib import suppress
from unittest.mock import AsyncMock

import pytest
from websockets.protocol import State

from pipecat.frames.frames import TranscriptionFrame
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.deepgram.flux.stt_base import (
    DeepgramFluxSTTBase,
    DeepgramFluxSTTSettings,
)


class _FakeFluxWebsocket:
    """Controllable websocket for exercising Flux disconnect ordering."""

    def __init__(self, close_response=None):
        self.state = State.OPEN
        self.close_response = close_response
        self.messages = asyncio.Queue()
        self.receive_started = asyncio.Event()
        self.receiver_cancelled = asyncio.Event()
        self.close_stream_sent = asyncio.Event()
        self.receive_alive_when_close_stream_sent = None
        self.close_called = False
        self.close_call_count = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.receive_started.set()
        try:
            message = await self.messages.get()
        except asyncio.CancelledError:
            self.receiver_cancelled.set()
            raise
        if message is None:
            self.state = State.CLOSED
            raise StopAsyncIteration
        if isinstance(message, Exception):
            raise message
        return message

    async def send(self, message):
        if isinstance(message, str) and json.loads(message) == {"type": "CloseStream"}:
            self.receive_alive_when_close_stream_sent = not self.receiver_cancelled.is_set()
            self.close_stream_sent.set()
            if self.close_response is not None:
                if isinstance(self.close_response, Exception):
                    await self.messages.put(self.close_response)
                else:
                    await self.messages.put(json.dumps(self.close_response))
                    await self.messages.put(None)

    async def close(self):
        self.close_called = True
        self.close_call_count += 1
        self.state = State.CLOSED


def _make_websocket_flux_service(websocket):
    service = DeepgramFluxSTTService(api_key="test-key", sample_rate=16000)
    service._websocket = websocket
    service._disconnecting = True
    service.stop_all_metrics = AsyncMock()
    service._call_event_handler = AsyncMock()
    service.push_error = AsyncMock()

    cancelled_tasks = []

    async def cancel_task(task, timeout=1.0):
        cancelled_tasks.append(task)
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    service.cancel_task = cancel_task
    return service, cancelled_tasks


async def _start_receive_task(service, websocket):
    task = asyncio.create_task(service._receive_task_handler(service._report_error))
    service._receive_task = task
    await asyncio.wait_for(websocket.receive_started.wait(), timeout=1.0)
    return task


def _make_fake_flux_service():
    """Build a minimal concrete Flux service for exercising the Configure logic.

    The subclass is defined lazily inside this factory (not at module level) so
    it never registers in ``AIService.__subclasses__()`` during import. That
    keeps it out of the auto-discovery in ``tests/test_service_init.py``, which
    walks every ``AIService`` subclass at collection time.

    The returned instance records every JSON message sent, so we can assert
    that Configure sends are serialized (never more than one in flight) and
    that bursts are coalesced rather than replayed one at a time.
    """

    class _FakeFluxService(DeepgramFluxSTTBase):
        def __init__(self):
            # Bypass STTService.__init__ (needs a pipeline); wire up only the
            # state _send_configure / _handle_message touch.
            self._name = "FakeFlux"
            self._settings = DeepgramFluxSTTSettings(model="flux-general-en")
            self._configure_in_flight = False
            self._configure_sent_at = None
            self._configure_pending_fields = None
            self._active = True
            self.sent_messages = []
            self.errors = []

        async def _transport_send_audio(self, audio: bytes):
            pass

        async def _transport_send_json(self, message: dict):
            self.sent_messages.append(message)

        def _transport_is_active(self) -> bool:
            return self._active

        async def _connect(self):
            pass

        async def _disconnect(self):
            pass

        async def run_stt(self, audio: bytes):
            yield None

        async def push_error(self, error_msg, exception=None):
            self.errors.append(error_msg)

    return _FakeFluxService()


@pytest.mark.asyncio
async def test_disconnect_drains_final_end_of_turn_before_closing_websocket():
    """CloseStream keeps the receiver alive long enough to process the final turn."""
    final_turn = {
        "type": "TurnInfo",
        "event": "EndOfTurn",
        "transcript": "the last words",
    }
    websocket = _FakeFluxWebsocket(close_response=final_turn)
    service, cancelled_tasks = _make_websocket_flux_service(websocket)
    service.emit_stt_usage_metrics = AsyncMock()
    service.push_frame = AsyncMock()
    service._handle_transcription = AsyncMock()
    service.broadcast_frame = AsyncMock()
    receive_task = await _start_receive_task(service, websocket)

    await service._disconnect_websocket()

    assert websocket.receive_alive_when_close_stream_sent is True
    transcription = service.push_frame.await_args.args[0]
    assert isinstance(transcription, TranscriptionFrame)
    assert transcription.text == "the last words"
    assert transcription.finalized
    assert transcription.result == final_turn
    assert receive_task.done()
    assert cancelled_tasks == []
    assert service._receive_task is None
    assert websocket.close_called


@pytest.mark.asyncio
async def test_disconnect_cancels_receiver_when_server_does_not_close():
    """A server that never closes cannot hold Flux teardown open indefinitely."""
    websocket = _FakeFluxWebsocket()
    service, cancelled_tasks = _make_websocket_flux_service(websocket)
    service._CLOSE_STREAM_DRAIN_TIMEOUT = 0
    receive_task = await _start_receive_task(service, websocket)

    await service._disconnect_websocket()

    assert websocket.receive_alive_when_close_stream_sent is True
    assert websocket.receiver_cancelled.is_set()
    assert receive_task.done()
    assert cancelled_tasks == [receive_task]
    assert service._receive_task is None
    assert websocket.close_called


@pytest.mark.asyncio
async def test_disconnect_with_completed_receiver_is_safe_and_idempotent():
    """An already-finished receiver is neither cancelled again nor left referenced."""
    websocket = _FakeFluxWebsocket()
    websocket.state = State.CLOSED
    service, cancelled_tasks = _make_websocket_flux_service(websocket)
    receive_task = asyncio.create_task(asyncio.sleep(0))
    await receive_task
    service._receive_task = receive_task

    await service._disconnect_websocket()
    await service._disconnect_websocket()

    assert cancelled_tasks == []
    assert service._receive_task is None
    assert websocket.close_called
    assert websocket.close_call_count == 1
    service._call_event_handler.assert_awaited_once_with("on_disconnected")


@pytest.mark.asyncio
async def test_disconnect_handles_receive_error_during_graceful_drain():
    """A receive-loop failure during drain still completes websocket cleanup."""
    websocket = _FakeFluxWebsocket(close_response=RuntimeError("receive failed"))
    service, cancelled_tasks = _make_websocket_flux_service(websocket)
    receive_task = await _start_receive_task(service, websocket)

    await service._disconnect_websocket()

    assert websocket.receive_alive_when_close_stream_sent is True
    assert receive_task.done()
    assert cancelled_tasks == []
    assert service._receive_task is None
    assert websocket.close_called


@pytest.mark.asyncio
async def test_cancelling_disconnect_still_cleans_up_receiver_and_websocket():
    """Cancellation of teardown propagates only after its resources are released."""
    websocket = _FakeFluxWebsocket()
    service, cancelled_tasks = _make_websocket_flux_service(websocket)
    watchdog_task = asyncio.create_task(asyncio.Event().wait())
    service._watchdog_task = watchdog_task
    receive_task = await _start_receive_task(service, websocket)
    disconnect_task = asyncio.create_task(service._disconnect_websocket())
    await asyncio.wait_for(websocket.close_stream_sent.wait(), timeout=1.0)

    disconnect_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await disconnect_task

    assert websocket.receiver_cancelled.is_set()
    assert receive_task.done()
    assert watchdog_task.done()
    assert cancelled_tasks == [watchdog_task, receive_task]
    assert service._watchdog_task is None
    assert service._receive_task is None
    assert service._websocket is None
    assert service._disconnecting is True
    assert websocket.close_called


@pytest.mark.asyncio
async def test_reconnect_disconnect_restores_disconnecting_state():
    """A reconnect teardown suppresses receive errors only while the old socket drains."""
    websocket = _FakeFluxWebsocket(
        close_response={"type": "TurnInfo", "event": "Update", "transcript": "pending"}
    )
    service, _ = _make_websocket_flux_service(websocket)
    service._disconnecting = False
    service._report_error = AsyncMock()
    await _start_receive_task(service, websocket)

    await service._disconnect_websocket()

    assert service._disconnecting is False
    service._report_error.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_configure_sends_immediately_when_idle():
    """With nothing in flight, _send_configure sends right away and marks in-flight."""
    service = _make_fake_flux_service()
    service._settings.eot_threshold = 0.8

    await service._send_configure({"eot_threshold"})

    assert service.sent_messages == [{"type": "Configure", "thresholds": {"eot_threshold": 0.8}}]
    assert service._configure_in_flight
    assert service._configure_sent_at is not None


@pytest.mark.asyncio
async def test_send_configure_coalesces_burst_while_in_flight():
    """A burst of sends while one is in flight is coalesced, not replayed one at a time."""
    service = _make_fake_flux_service()
    service._settings.eot_threshold = 0.1
    service._settings.eager_eot_threshold = 0.2

    await service._send_configure({"eot_threshold"})
    assert len(service.sent_messages) == 1

    # These arrive while the first is still in flight: coalesced, not sent.
    await service._send_configure({"eager_eot_threshold"})
    service._settings.eager_eot_threshold = 0.9
    await service._send_configure({"eager_eot_threshold"})

    assert len(service.sent_messages) == 1
    assert service._configure_pending_fields == {"eager_eot_threshold"}

    # Acking the first flushes the coalesced update, built from *current*
    # settings — not whatever value was in effect when it was queued.
    await service._handle_message({"type": "ConfigureSuccess"})

    assert service.sent_messages == [
        {"type": "Configure", "thresholds": {"eot_threshold": 0.1}},
        {"type": "Configure", "thresholds": {"eager_eot_threshold": 0.9}},
    ]
    assert service._configure_in_flight  # the flushed Configure is now in flight
    assert service._configure_pending_fields is None


@pytest.mark.asyncio
async def test_send_configure_failure_flushes_pending_and_pushes_error():
    """A ConfigureFailure still flushes any coalesced update and pushes an error."""
    service = _make_fake_flux_service()
    service._settings.eot_threshold = 0.5

    await service._send_configure({"eot_threshold"})
    await service._send_configure({"eager_eot_threshold"})  # coalesced

    await service._handle_message(
        {"type": "ConfigureFailure", "error_code": "bad", "description": "nope"}
    )

    assert service.errors == ["Configure rejected: [bad] nope"]
    assert len(service.sent_messages) == 2  # the coalesced update was still flushed
    assert service._configure_pending_fields is None


@pytest.mark.asyncio
async def test_send_configure_supersedes_stale_in_flight_instead_of_coalescing():
    """A Configure whose ack never arrives must not block later updates forever."""
    service = _make_fake_flux_service()
    service._CONFIGURE_ACK_TIMEOUT = 0.01

    await service._send_configure({"eot_threshold"})
    assert len(service.sent_messages) == 1

    await asyncio.sleep(0.02)  # let the in-flight Configure go stale

    # No ack ever arrived, but this must send now rather than coalesce forever.
    await service._send_configure({"eager_eot_threshold"})

    assert len(service.sent_messages) == 2
    assert service._configure_pending_fields is None


@pytest.mark.asyncio
async def test_on_configure_acked_drops_pending_fields_when_transport_inactive():
    """An ack shouldn't try to flush a pending update once the transport is dead.

    Otherwise the flush's send raises inside _handle_message, which both
    swallows whatever error/success handling comes after it and (without a
    reconnect to clean up afterwards) leaves _configure_in_flight stuck True.
    """
    service = _make_fake_flux_service()

    await service._send_configure({"eot_threshold"})
    await service._send_configure({"eager_eot_threshold"})  # coalesced

    service._active = False  # transport has gone away before the ack arrives
    await service._handle_message({"type": "ConfigureSuccess"})

    assert len(service.sent_messages) == 1  # the pending Configure was not sent
    assert not service._configure_in_flight
    assert service._configure_pending_fields is None


@pytest.mark.asyncio
async def test_stray_ack_is_ignored():
    """An ack with no Configure in flight should be ignored gracefully."""
    service = _make_fake_flux_service()

    # No send in flight; this must not raise.
    await service._handle_message({"type": "ConfigureSuccess"})

    assert not service._configure_in_flight
    assert service._configure_pending_fields is None


@pytest.mark.asyncio
async def test_reset_configure_state_clears_in_flight_and_pending():
    """Teardown must clear both the in-flight and any coalesced pending update."""
    service = _make_fake_flux_service()

    await service._send_configure({"eot_threshold"})
    await service._send_configure({"eager_eot_threshold"})  # coalesced

    service._reset_configure_state()

    assert not service._configure_in_flight
    assert service._configure_sent_at is None
    assert service._configure_pending_fields is None


def test_reset_configure_state_with_nothing_in_flight_is_safe():
    """Calling the reset helper with nothing in flight must not raise."""
    service = _make_fake_flux_service()

    service._reset_configure_state()

    assert not service._configure_in_flight
    assert service._configure_pending_fields is None


if __name__ == "__main__":
    unittest.main()
