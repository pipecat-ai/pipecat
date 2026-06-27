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


def _make_fake_flux_service():
    """Build a minimal concrete Flux service for exercising the Configure logic.

    The subclass is defined lazily inside this factory (not at module level) so
    it never registers in ``AIService.__subclasses__()`` during import. That
    keeps it out of the auto-discovery in ``tests/test_service_init.py``, which
    walks every ``AIService`` subclass at collection time.

    The returned instance records every JSON message sent and lets a test
    resolve the Configure ack on demand, so we can assert that sends are
    serialized one-at-a-time.
    """

    class _FakeFluxService(DeepgramFluxSTTBase):
        def __init__(self):
            # Bypass STTService.__init__ (needs a pipeline); wire up only the
            # state _send_configure / _handle_message touch.
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

    return _FakeFluxService()


@pytest.mark.asyncio
async def test_send_configure_waits_for_ack():
    """_send_configure should block until a ConfigureSuccess ack arrives."""
    service = _make_fake_flux_service()
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
    service = _make_fake_flux_service()

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
    service = _make_fake_flux_service()

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
    service = _make_fake_flux_service()
    service._CONFIGURE_ACK_TIMEOUT = 0.01

    # No ack is ever delivered; the send should still complete via the timeout.
    await asyncio.wait_for(service._send_configure({"eot_threshold"}), timeout=1.0)

    assert len(service.sent_messages) == 1
    assert service._configure_ack is None


@pytest.mark.asyncio
async def test_stray_ack_is_ignored():
    """An ack with no Configure in flight should be ignored gracefully."""
    service = _make_fake_flux_service()

    # No send in flight; this must not raise.
    await service._handle_message({"type": "ConfigureSuccess"})

    assert service._configure_ack is None


@pytest.mark.asyncio
async def test_cancel_configure_ack_unblocks_in_flight_send():
    """Teardown/reconnect must unblock an in-flight send instead of leaving it hung."""
    service = _make_fake_flux_service()

    send = asyncio.ensure_future(service._send_configure({"eot_threshold"}))
    await asyncio.sleep(0)
    assert not send.done()  # waiting for an ack that will never arrive

    # Simulates the teardown/reconnect call sites unblocking the waiter.
    service._cancel_configure_ack()
    await send  # returns cleanly rather than raising or hanging

    assert service._configure_ack is None


def test_cancel_configure_ack_no_send_in_flight_is_safe():
    """Calling the unblock helper with nothing in flight must not raise."""
    service = _make_fake_flux_service()

    # No send in flight; this must be a no-op.
    service._cancel_configure_ack()

    assert service._configure_ack is None
