#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import dataclasses
import unittest
from urllib.parse import parse_qs

import pytest

from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.deepgram.flux.stt_base import (
    DeepgramFluxSTTBase,
    DeepgramFluxSTTSettings,
    FluxConnectionNotConfirmedError,
    FluxFatalError,
)
from pipecat.utils.errors import ErrorCategory

pytest.importorskip("aws_sdk_sagemaker_runtime_http2")

from pipecat.services.deepgram.flux.sagemaker.stt import (  # noqa: E402
    DeepgramFluxSageMakerSTTService,
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


# ---------------------------------------------------------------------------
# Connection parameter limits
#
# Flux refuses the whole connection for a value outside the limits it documents,
# which costs the session its STT. Values it would refuse are normalized before
# they reach the connection URL.
# ---------------------------------------------------------------------------


def _query_params(**settings) -> dict[str, list[str]]:
    """Build the connection query string for the given settings, parsed.

    Constructing the service opens no connection and starts no task, so the
    instance needs no teardown.
    """
    service = DeepgramFluxSTTService(
        api_key="test-key",
        sample_rate=16000,
        settings=DeepgramFluxSTTService.Settings(**settings),
    )
    # sample_rate is normally set during setup, which these tests skip.
    service._sample_rate = 16000
    return parse_qs(service._build_query_string())


def test_query_string_drops_eager_threshold_above_eot_threshold():
    """Flux refuses the connection for it, so it is dropped rather than sent."""
    params = _query_params(eot_threshold=0.85, eager_eot_threshold=0.9)

    assert "eager_eot_threshold" not in params
    assert params["eot_threshold"] == ["0.85"]


def test_query_string_drops_eager_threshold_above_the_default_eot_threshold():
    """Flux compares against its own default when the connection sets no eot_threshold.

    Turning eager end-of-turn on without setting eot_threshold is the documented
    way to use it, so this is the configuration most likely to be refused.
    """
    params = _query_params(eager_eot_threshold=0.8)

    assert "eager_eot_threshold" not in params
    assert "eot_threshold" not in params


def test_query_string_drops_eager_threshold_below_the_accepted_range():
    """Flux refuses the connection for it, the same as for one above the range."""
    assert "eager_eot_threshold" not in _query_params(eager_eot_threshold=0.2)


def test_query_string_drops_eager_threshold_above_the_accepted_range():
    """The range applies even when eot_threshold leaves room above it."""
    params = _query_params(eot_threshold=1.0, eager_eot_threshold=0.95)

    assert "eager_eot_threshold" not in params
    assert params["eot_threshold"] == ["1.0"]


def test_query_string_keeps_eager_threshold_at_the_range_bounds():
    """The bounds are inclusive, so a value on either is sent as given."""
    assert _query_params(eager_eot_threshold=0.3)["eager_eot_threshold"] == ["0.3"]
    assert _query_params(eot_threshold=1.0, eager_eot_threshold=0.9)["eager_eot_threshold"] == [
        "0.9"
    ]


def test_query_string_truncates_keyterms_to_the_connection_limit():
    """An oversized list is truncated instead of failing the connection."""
    terms = [f"term{i}" for i in range(150)]

    assert _query_params(keyterm=terms)["keyterm"] == terms[:100]


def test_query_string_keeps_keyterms_at_the_connection_limit():
    """A list at the limit is sent whole."""
    terms = [f"term{i}" for i in range(100)]

    assert _query_params(keyterm=terms)["keyterm"] == terms


def test_query_string_truncates_one_keyterm_over_the_connection_limit():
    """The limit is a maximum, not a threshold one term above it."""
    terms = [f"term{i}" for i in range(101)]

    assert _query_params(keyterm=terms)["keyterm"] == terms[:100]


def test_query_string_drops_keyterms_over_the_character_limit():
    """Flux refuses the connection for an over-long keyterm, so it is dropped."""
    params = _query_params(keyterm=["a" * 100, "b" * 101, "fine"])

    assert params["keyterm"] == ["a" * 100, "fine"]


def test_query_string_drops_a_blank_keyterm():
    """Flux refuses the connection for an empty keyterm, and a blank is under the length limit.

    The character filter alone lets "" through, so the guard meant to protect the connection
    would have been what killed it. Deepgram answers HTTP 400 "Query included a keyterm that was
    the empty string".
    """
    params = _query_params(keyterm=["alpha", "", "   ", "beta"])

    assert params["keyterm"] == ["alpha", "beta"]


def test_query_string_strips_surrounding_whitespace_from_a_keyterm():
    """Stripping is what makes a whitespace-only term blank, so the sent term is the stripped one."""
    assert _query_params(keyterm=["  alpha  "])["keyterm"] == ["alpha"]


def test_query_string_applies_both_guards_on_one_connection():
    """An invalid threshold and an oversized keyterm list are handled together."""
    params = _query_params(
        eot_threshold=0.85,
        eager_eot_threshold=0.9,
        keyterm=[f"term{i}" for i in range(150)],
    )

    assert "eager_eot_threshold" not in params
    assert params["eot_threshold"] == ["0.85"]
    assert len(params["keyterm"]) == 100


if __name__ == "__main__":
    unittest.main()
