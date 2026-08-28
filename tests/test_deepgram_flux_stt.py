#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

import pytest

from pipecat.services.deepgram.flux.stt_base import (
    DeepgramFluxSTTBase,
    DeepgramFluxSTTSettings,
)


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
            self.reconnect_requests = 0
            self.connection_events = []

        async def _transport_send_audio(self, audio: bytes):
            pass

        async def _transport_send_json(self, message: dict):
            self.sent_messages.append(message)

        def _transport_is_active(self) -> bool:
            return self._active

        async def _connect(self):
            self.connection_events.append("connect")

        async def _disconnect(self):
            self.connection_events.append("disconnect")

        async def _request_reconnect(self):
            self.reconnect_requests += 1

        async def set_usable(self, usable: bool):
            pass

        async def run_stt(self, audio: bytes):
            yield None

        async def push_error(self, error_msg, exception=None):
            self.errors.append(error_msg)

    return _FakeFluxService()


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


@pytest.mark.asyncio
async def test_do_reconnect_tears_down_before_re_establishing():
    """A reconnect drops the current connection before opening a new one."""
    service = _make_fake_flux_service()

    await service._do_reconnect()

    assert service.connection_events == ["disconnect", "connect"]


@pytest.mark.asyncio
async def test_update_settings_reconnects_for_connection_only_field():
    """Fields Flux only reads from the connection URL are applied by reconnecting."""
    service = _make_fake_flux_service()

    await service._update_settings(DeepgramFluxSTTSettings(numerals=True))

    assert service.reconnect_requests == 1
    assert service.sent_messages == []


@pytest.mark.asyncio
async def test_update_settings_configures_without_reconnecting():
    """Configure-able fields reach the live connection without dropping it."""
    service = _make_fake_flux_service()

    await service._update_settings(DeepgramFluxSTTSettings(eot_threshold=0.9))

    assert service.sent_messages == [{"type": "Configure", "thresholds": {"eot_threshold": 0.9}}]
    assert service.reconnect_requests == 0


if __name__ == "__main__":
    unittest.main()
