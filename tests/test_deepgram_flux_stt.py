#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio

import pytest

from pipecat.services.deepgram.flux.base import (
    DeepgramFluxSTTBase,
    DeepgramFluxSTTSettings,
)


class _FakeFluxService(DeepgramFluxSTTBase):
    """Minimal concrete subclass exercising the shared Configure logic.

    Records every JSON message sent and lets a test resolve the Configure ack
    on demand, so we can assert that sends are serialized one-at-a-time.
    """

    def __init__(self):
        # Bypass STTService.__init__ (needs a pipeline); wire up only the state
        # _send_configure / _handle_message touch.
        self._name = "FakeFlux"
        self._settings = DeepgramFluxSTTSettings(model="flux-general-en")
        self._configure_lock = asyncio.Lock()
        self._configure_ack = None
        self.sent_messages = []
        self.errors = []

    async def _transport_send_audio(self, audio: bytes):
        pass

    async def _transport_send_json(self, message: dict):
        self.sent_messages.append(message)

    def _transport_is_active(self) -> bool:
        return True

    async def _connect(self):
        pass

    async def _disconnect(self):
        pass

    async def run_stt(self, audio: bytes):
        yield None

    async def push_error(self, error_msg, exception=None):
        self.errors.append(error_msg)


@pytest.mark.asyncio
async def test_send_configure_waits_for_ack():
    """_send_configure should block until a ConfigureSuccess ack arrives."""
    service = _FakeFluxService()
    service._settings.eot_threshold = 0.8

    send = asyncio.ensure_future(service._send_configure({"eot_threshold"}))
    await asyncio.sleep(0)  # let _send_configure write the message and start waiting

    assert service.sent_messages == [{"type": "Configure", "thresholds": {"eot_threshold": 0.8}}]
    assert not send.done()  # still waiting for the ack

    await service._handle_message({"type": "ConfigureSuccess"})
    await send

    assert service._configure_ack is None  # waiter cleaned up


@pytest.mark.asyncio
async def test_send_configure_serializes_bursts():
    """A burst of Configure sends must go out one ack at a time, never overlapping."""
    service = _FakeFluxService()

    first = asyncio.ensure_future(service._send_configure({"eot_threshold"}))
    second = asyncio.ensure_future(service._send_configure({"eager_eot_threshold"}))
    await asyncio.sleep(0)

    # Only the first Configure is in flight; the second is blocked on the lock.
    assert len(service.sent_messages) == 1

    await service._handle_message({"type": "ConfigureSuccess"})
    await first
    await asyncio.sleep(0)

    # Now the second Configure is sent.
    assert len(service.sent_messages) == 2

    await service._handle_message({"type": "ConfigureSuccess"})
    await second

    assert len(service.sent_messages) == 2


@pytest.mark.asyncio
async def test_send_configure_failure_signals_and_pushes_error():
    """A ConfigureFailure should unblock the waiter and still push an error."""
    service = _FakeFluxService()

    send = asyncio.ensure_future(service._send_configure({"eot_threshold"}))
    await asyncio.sleep(0)

    await service._handle_message(
        {"type": "ConfigureFailure", "error_code": "bad", "description": "nope"}
    )
    await send

    assert service.errors == ["Configure rejected: [bad] nope"]
    assert service._configure_ack is None


@pytest.mark.asyncio
async def test_send_configure_times_out_without_ack():
    """A missing ack must not deadlock: the wait is bounded and then proceeds."""
    service = _FakeFluxService()
    service._CONFIGURE_ACK_TIMEOUT = 0.01

    # No ack is ever delivered; the send should still complete via the timeout.
    await asyncio.wait_for(service._send_configure({"eot_threshold"}), timeout=1.0)

    assert len(service.sent_messages) == 1
    assert service._configure_ack is None


@pytest.mark.asyncio
async def test_stray_ack_is_ignored():
    """An ack with no Configure in flight should be ignored gracefully."""
    service = _FakeFluxService()

    # No send in flight; this must not raise.
    await service._handle_message({"type": "ConfigureSuccess"})

    assert service._configure_ack is None
