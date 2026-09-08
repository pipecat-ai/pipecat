#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import dataclasses
import unittest

import pytest

from pipecat.frames.frames import (
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.deepgram.flux.stt_base import (
    DeepgramFluxSTTBase,
    DeepgramFluxSTTSettings,
    FluxConnectionNotConfirmedError,
    FluxFatalError,
    FluxTurnDetection,
)
from pipecat.services.stt_service import STTService
from pipecat.utils.errors import ErrorCategory

pytest.importorskip("aws_sdk_sagemaker_runtime_http2")

from pipecat.services.deepgram.flux.sagemaker.stt import (  # noqa: E402
    DeepgramFluxSageMakerSTTService,
)


def _make_fake_flux_service(turn_detection=FluxTurnDetection.AUTOMATIC):
    """Build a minimal concrete Flux service for exercising the protocol logic.

    The subclass is defined lazily inside this factory (not at module level) so
    it never registers in ``AIService.__subclasses__()`` during import. That
    keeps it out of the auto-discovery in ``tests/test_service_init.py``, which
    walks every ``AIService`` subclass at collection time.

    The returned instance records every JSON message sent, every frame pushed
    and every frame class broadcast, so tests can assert both what goes out to
    Flux and what reaches the pipeline.

    Args:
        turn_detection: Which turn detection mode the service runs in.
    """

    class _FakeFluxService(DeepgramFluxSTTBase):
        def __init__(self):
            # Bypass STTService.__init__ (needs a pipeline); wire up only the
            # state the methods under test touch.
            self._name = "FakeFlux"
            self._settings = DeepgramFluxSTTSettings(model="flux-general-en", min_confidence=None)
            self._configure_in_flight = False
            self._configure_sent_at = None
            self._configure_pending_fields = None
            self._active = True
            self._turn_detection = turn_detection
            self._user_is_speaking = False
            self._user_id = ""
            self._finalize_requested = False
            self._finalize_pending = False
            self.sent_messages = []
            self.errors = []
            self.reconnect_requests = 0
            self.connection_events = []
            self.pushed_frames = []
            self.broadcast_frames = []

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

        async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
            self.pushed_frames.append(frame)

        async def broadcast_frame(self, frame_cls, **kwargs):
            self.broadcast_frames.append(frame_cls)

        async def _call_event_handler(self, name, *args, **kwargs):
            pass

        async def emit_stt_usage_metrics(self):
            pass

        async def _handle_transcription(self, transcript, is_final, language=None):
            pass

    return _FakeFluxService()


async def _noop_process_frame(self, frame, direction):
    """Stand in for STTService.process_frame, which needs a live pipeline."""
    pass


def _turn_info(event, **extra):
    """Build a TurnInfo message payload."""
    return {"type": "TurnInfo", "event": event, "transcript": "hello there", **extra}


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


@pytest.mark.asyncio
async def test_fatal_error_reports_code_and_description():
    """A FatalError raises with the code and description Flux sends."""
    service = _make_fake_flux_service()

    with pytest.raises(Exception) as excinfo:
        await service._handle_fatal_error(
            {
                "code": "UNPARSABLE_CLIENT_MESSAGE",
                "description": "Could not deserialize last text message",
            }
        )

    assert "UNPARSABLE_CLIENT_MESSAGE" in str(excinfo.value)
    assert "Could not deserialize last text message" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Settings capabilities
#
# Every field is classified by how it reaches Flux: sent over the live
# connection (Configure), applied by reconnecting, applied locally, or
# reported as unsupported.
# ---------------------------------------------------------------------------


def _settings_fields():
    """Every declared Flux setting, minus the inherited overflow dict."""
    return {f.name for f in dataclasses.fields(DeepgramFluxSTTSettings)} - {"extra"}


# Flux has no `language` parameter; `language_hints` covers multilingual input.
_UNSUPPORTED_FIELDS = {"language"}


@pytest.mark.parametrize("service", [DeepgramFluxSTTService, DeepgramFluxSageMakerSTTService])
def test_every_setting_is_classified(service):
    """No setting on either transport is left unclassified.

    A field added to the settings without being classified would otherwise
    only show up as a log warning at runtime.
    """
    fields = _settings_fields()
    classified = (
        service._CONFIGURE_FIELDS
        | service._CONNECTION_FIELDS
        | service._LOCAL_FIELDS
        | _UNSUPPORTED_FIELDS
    )
    assert fields - classified == set()


def test_no_setting_is_classified_two_ways():
    """A field belongs to exactly one bucket, on either transport."""
    for service in (DeepgramFluxSTTService, DeepgramFluxSageMakerSTTService):
        configure = service._CONFIGURE_FIELDS
        connection = service._CONNECTION_FIELDS
        local = service._LOCAL_FIELDS
        assert configure & connection == set()
        assert configure & local == set()
        assert connection & local == set()


@pytest.mark.asyncio
async def test_fatal_error_carries_the_flux_code():
    """A FatalError raises a typed error the service can classify."""
    service = _make_fake_flux_service()

    with pytest.raises(FluxFatalError) as excinfo:
        await service._handle_fatal_error(
            {"code": "UNPARSABLE_CLIENT_MESSAGE", "description": "Bad message"}
        )

    assert excinfo.value.code == "UNPARSABLE_CLIENT_MESSAGE"


def test_flux_error_codes_are_classified():
    """Flux codes a retry can't clear are classified so the service stops taking work.

    Flux reports these over the connection rather than as an HTTP status, so
    without this they'd be UNKNOWN and the service would keep looking healthy.
    """
    service = _make_fake_flux_service()

    unparsable = service._classify_error(
        FluxFatalError("bad message", code="UNPARSABLE_CLIENT_MESSAGE")
    )

    assert unparsable == ErrorCategory.INVALID_REQUEST
    # Permanent categories are what cost the processor its usability.
    assert unparsable.is_permanent


def test_unrecognized_flux_error_code_falls_back():
    """An unmapped code defers to the default classification."""
    service = _make_fake_flux_service()

    assert service._classify_error(FluxFatalError("boom", code="SOMETHING_NEW")) is None


@pytest.mark.asyncio
async def test_connection_wait_times_out_instead_of_hanging():
    """An endpoint that never confirms the connection fails instead of hanging."""
    service = _make_fake_flux_service()
    service._CONNECTION_TIMEOUT = 0.05
    service._connection_established_event = asyncio.Event()

    with pytest.raises(FluxConnectionNotConfirmedError) as excinfo:
        await service._await_connection_established()

    assert "did not confirm the connection" in str(excinfo.value)


def test_unconfirmed_connection_is_treated_as_a_rejected_request():
    """A silent endpoint means the settings were rejected, not that it was slow."""
    service = _make_fake_flux_service()

    category = service._classify_error(FluxConnectionNotConfirmedError("no confirmation"))

    assert category == ErrorCategory.INVALID_REQUEST
    assert category.is_permanent


@pytest.mark.asyncio
async def test_connection_wait_returns_once_confirmed():
    """A confirmed connection returns without waiting out the timeout."""
    service = _make_fake_flux_service()
    service._connection_established_event = asyncio.Event()
    service._connection_established_event.set()

    await service._await_connection_established()


# ----------------------------------------------------------------------
# Turn detection modes
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_manual_mode_force_ends_turn_on_vad_stop(monkeypatch):
    """In manual mode the VAD stop signal asks Flux to finalize the audio sent."""
    monkeypatch.setattr(STTService, "process_frame", _noop_process_frame)
    service = _make_fake_flux_service(FluxTurnDetection.MANUAL)
    service._user_is_speaking = True

    await service.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    assert service.sent_messages == [{"type": "ForceEndTurn"}]
    assert service._finalize_requested


@pytest.mark.asyncio
async def test_automatic_mode_ignores_vad_stop(monkeypatch):
    """In automatic mode Flux owns the turn, so VAD must not cut it short."""
    monkeypatch.setattr(STTService, "process_frame", _noop_process_frame)
    service = _make_fake_flux_service(FluxTurnDetection.AUTOMATIC)
    service._user_is_speaking = True

    await service.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    assert service.sent_messages == []


@pytest.mark.asyncio
async def test_force_end_turn_is_inert_with_no_active_turn():
    """With no turn in progress there is nothing to finalize."""
    service = _make_fake_flux_service(FluxTurnDetection.MANUAL)
    service._user_is_speaking = False

    await service.force_end_turn()

    assert service.sent_messages == []
    assert not service._finalize_requested


@pytest.mark.asyncio
async def test_force_end_turn_is_inert_on_dead_transport():
    """A closed connection can't carry a ForceEndTurn."""
    service = _make_fake_flux_service(FluxTurnDetection.MANUAL)
    service._user_is_speaking = True
    service._active = False

    await service.force_end_turn()

    assert service.sent_messages == []


@pytest.mark.asyncio
async def test_manual_mode_suppresses_turn_proposals():
    """Manual mode transcribes only: the configured strategies own the turn."""
    service = _make_fake_flux_service(FluxTurnDetection.MANUAL)

    await service._handle_message(_turn_info("StartOfTurn"))
    await service._handle_message(_turn_info("EndOfTurn", trigger="manual"))

    assert service.broadcast_frames == []
    # The transcript still reaches the pipeline; only the proposals are dropped.
    assert [type(f) for f in service.pushed_frames] == [TranscriptionFrame]
    assert service.pushed_frames[0].finalized


@pytest.mark.asyncio
async def test_automatic_mode_proposes_both_turn_edges():
    """Automatic mode hands both turn edges to the external turn strategies."""
    service = _make_fake_flux_service(FluxTurnDetection.AUTOMATIC)

    await service._handle_message(_turn_info("StartOfTurn"))
    await service._handle_message(_turn_info("EndOfTurn", trigger="model"))

    assert service.broadcast_frames == [
        ProposedUserStartedSpeakingFrame,
        ProposedUserStoppedSpeakingFrame,
    ]


@pytest.mark.asyncio
async def test_manual_trigger_confirms_a_requested_finalize():
    """An EndOfTurn we asked for settles the finalize request; one we didn't doesn't."""
    service = _make_fake_flux_service(FluxTurnDetection.MANUAL)
    service._user_is_speaking = True
    await service.force_end_turn()

    await service._handle_message(_turn_info("EndOfTurn", trigger="timeout"))
    assert service._finalize_requested  # still waiting on our own ForceEndTurn

    service._user_is_speaking = True
    await service._handle_message(_turn_info("EndOfTurn", trigger="manual"))
    assert not service._finalize_requested
    assert service._finalize_pending


@pytest.mark.asyncio
async def test_force_end_turn_losing_the_race_is_not_an_error():
    """Flux may have ended the turn already; that warning is routine, not a failure."""
    service = _make_fake_flux_service(FluxTurnDetection.MANUAL)

    await service._handle_message(
        {
            "type": "Warning",
            "code": "FORCE_END_TURN_NO_ACTIVE_TURN",
            "description": "No active turn to end",
        }
    )

    assert service.errors == []
    assert service.pushed_frames == []


@pytest.mark.asyncio
async def test_unrecognized_warning_is_not_an_error():
    """Warnings never interrupt the stream, so none of them push an error."""
    service = _make_fake_flux_service()

    await service._handle_message(
        {"type": "Warning", "code": "SOMETHING_NEW", "description": "unrecognized"}
    )

    assert service.errors == []


if __name__ == "__main__":
    unittest.main()
