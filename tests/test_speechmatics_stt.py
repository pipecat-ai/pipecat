#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for the Speechmatics STT service.

These run fully offline — no network, no live STT session. They cover the
non-trivial, decision-carrying logic in ``stt.py``: model/operating_point
reconciliation, deprecated-param migration, settings precedence, turn-mode
gating, the segment→frame mapping, the ``send_message`` contract, and the
reconnect loop. Each test locks one behavior; changing that behavior in the
source should break the test.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from speechmatics.agent_stt import AudioEncoding, Model

from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
from pipecat.services.speechmatics.stt import (
    SpeechmaticsSTTService,
    TurnDetectionMode,
    _is_auth_rejection,
    _resolve_model,
)
from pipecat.transcriptions.language import Language

try:
    from speechmatics.agent_stt import Segment
except ImportError:  # pragma: no cover - the service import above would already fail
    Segment = None


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Keep credential resolution deterministic: no ambient Speechmatics env vars
    leak into construction, so ``api_key=`` is the only source under test."""
    monkeypatch.delenv("SPEECHMATICS_API_KEY", raising=False)
    monkeypatch.delenv("SPEECHMATICS_RT_URL", raising=False)


def _service(**kwargs) -> SpeechmaticsSTTService:
    return SpeechmaticsSTTService(api_key="test-key", sample_rate=16000, **kwargs)


# ---------------------------------------------------------------------------
# _resolve_model — model / operating_point reconciliation (deprecation logic)
# ---------------------------------------------------------------------------


def test_resolve_model_prefers_model_over_none_operating_point():
    """`model` is the canonical field; it must win when `operating_point` is unset."""
    assert _resolve_model("linden-1", None) == "linden-1"


def test_resolve_model_conflicting_values_raise():
    """Two different values is a caller error, not a silent pick — must raise so the
    ambiguity surfaces instead of one arbitrarily winning."""
    with pytest.raises(ValueError):
        _resolve_model("linden-1", "some-other-model")


def test_resolve_model_operating_point_only_warns_and_is_used():
    """The deprecated alias still functions, but using it must emit a
    DeprecationWarning (the whole point of keeping the alias observable)."""
    with pytest.warns(DeprecationWarning):
        assert _resolve_model(None, "linden-1") == "linden-1"


def test_resolve_model_defaults_when_neither_given():
    """With nothing specified, the SDK default model must be chosen — not None,
    which would later fail ``assert_given`` in _build_config."""
    assert _resolve_model(None, None) == Model.LINDEN_1.value


def test_resolve_model_returns_wire_string_for_enum_input():
    """A `Model` enum member must be reduced to its wire string, since the SDK
    config is compared/serialized by string value."""
    resolved = _resolve_model(Model.LINDEN_1, None)
    assert resolved == "linden-1"
    assert isinstance(resolved, str)


# ---------------------------------------------------------------------------
# _apply_legacy_params — deprecated InputParams -> canonical Settings migration
# ---------------------------------------------------------------------------


def test_apply_legacy_params_copies_shared_fields():
    """Every field shared by InputParams and Settings must migrate; this guards the
    intersection-copy that replaced the hand-written per-field assignments."""
    settings = SpeechmaticsSTTService.Settings()
    params = SpeechmaticsSTTService.InputParams(domain="acme", max_speakers=3)

    SpeechmaticsSTTService._apply_legacy_params(settings, params)

    assert settings.domain == "acme"
    assert settings.max_speakers == 3


def test_apply_legacy_params_returns_encoding_without_setting_it_on_settings():
    """audio_encoding has no Settings field — it must be returned for the separate
    `encoding` path, not written onto Settings."""
    settings = SpeechmaticsSTTService.Settings()
    params = SpeechmaticsSTTService.InputParams(audio_encoding=AudioEncoding.MULAW)

    encoding = SpeechmaticsSTTService._apply_legacy_params(settings, params)

    assert encoding == AudioEncoding.MULAW
    assert not hasattr(settings, "audio_encoding")


def test_apply_legacy_params_speaker_format_default_depends_on_diarization():
    """When no format is given, the default prefixes the speaker only when diarizing
    (so multi-speaker transcripts are legible), and an explicit format is preserved."""
    on = SpeechmaticsSTTService.Settings()
    SpeechmaticsSTTService._apply_legacy_params(
        on, SpeechmaticsSTTService.InputParams(enable_diarization=True)
    )
    assert on.speaker_active_format == "@{speaker_id}: {text}"

    off = SpeechmaticsSTTService.Settings()
    SpeechmaticsSTTService._apply_legacy_params(
        off, SpeechmaticsSTTService.InputParams(enable_diarization=False)
    )
    assert off.speaker_active_format == "{text}"

    explicit = SpeechmaticsSTTService.Settings()
    SpeechmaticsSTTService._apply_legacy_params(
        explicit, SpeechmaticsSTTService.InputParams(speaker_active_format="X:{text}")
    )
    assert explicit.speaker_active_format == "X:{text}"


# ---------------------------------------------------------------------------
# _check_deprecated_args — legacy kwarg handling
# ---------------------------------------------------------------------------


def test_check_deprecated_args_migrates_renamed_kwarg():
    """A renamed kwarg must land on its new field, so old call sites keep working."""
    service = _service()
    kwargs = {"enable_speaker_diarization": True}
    params = SpeechmaticsSTTService.InputParams()

    with pytest.warns(DeprecationWarning):
        found = service._check_deprecated_args(kwargs, params)

    assert found is True
    assert params.enable_diarization is True


def test_check_deprecated_args_pops_recognized_kwargs():
    """Recognized deprecated kwargs must be removed from kwargs, or they would reach
    super().__init__ as unexpected keyword arguments and blow up construction."""
    service = _service()
    kwargs = {"enable_speaker_diarization": True}

    with pytest.warns(DeprecationWarning):
        service._check_deprecated_args(kwargs, SpeechmaticsSTTService.InputParams())

    assert "enable_speaker_diarization" not in kwargs


def test_check_deprecated_args_no_replacement_kwarg_does_not_crash():
    """A deprecated kwarg with no replacement (new=None) must warn and be dropped —
    never attempt setattr(params, None, ...), which used to raise TypeError."""
    service = _service()
    kwargs = {"max_delay": 5.0}
    params = SpeechmaticsSTTService.InputParams()

    with pytest.warns(DeprecationWarning):
        found = service._check_deprecated_args(kwargs, params)

    assert found is True
    assert "max_delay" not in kwargs
    assert not hasattr(params, "max_delay")


def test_check_deprecated_args_ignores_unknown_kwargs():
    """Unknown kwargs are not ours to touch: they must stay in kwargs (to be forwarded
    to the parent) and must not count as a legacy migration."""
    service = _service()
    kwargs = {"some_future_kwarg": 1}

    found = service._check_deprecated_args(kwargs, SpeechmaticsSTTService.InputParams())

    assert found is False
    assert kwargs == {"some_future_kwarg": 1}


# ---------------------------------------------------------------------------
# Construction: validation, defaults, precedence, model resolution
# ---------------------------------------------------------------------------


def test_missing_api_key_raises():
    """No key (and none in the environment) must fail loudly at construction rather
    than defer to an opaque auth failure at connect time."""
    with pytest.raises(ValueError):
        SpeechmaticsSTTService(api_key=None, sample_rate=16000)


def test_default_turn_detection_mode_is_vad():
    """The service must default to detecting turns itself (VAD); this default drives
    turn-frame emission and endpointing behavior downstream."""
    assert _service()._settings.turn_detection_mode == TurnDetectionMode.VAD


def test_settings_take_precedence_over_deprecated_params():
    """When both the deprecated `params` and canonical `settings` set the same field,
    `settings` must win — the documented migration contract."""
    with pytest.warns(DeprecationWarning):
        service = _service(
            params=SpeechmaticsSTTService.InputParams(domain="from_params"),
            settings=SpeechmaticsSTTService.Settings(domain="from_settings"),
        )
    assert service._settings.domain == "from_settings"


def test_operating_point_resolved_into_model():
    """The deprecated `operating_point` must be reconciled into the canonical `model`
    at construction, so the SDK config is built from a single resolved value."""
    with pytest.warns(DeprecationWarning):
        service = _service(settings=SpeechmaticsSTTService.Settings(operating_point="linden-1"))
    assert service._settings.model == "linden-1"


def test_diarization_config_built_when_enabled():
    """Enabling diarization with a knob must produce a wire diarization config carrying
    that knob — the path that actually turns on speaker attribution."""
    service = _service(
        settings=SpeechmaticsSTTService.Settings(enable_diarization=True, max_speakers=2)
    )
    assert service._config.diarization == "speaker"
    assert service._config.speaker_diarization_config.max_speakers == 2


def test_no_diarization_leaves_config_empty():
    """With diarization off, neither the diarization flag nor a speaker config may be
    sent — otherwise the engine would attempt attribution it was not asked for."""
    service = _service()
    assert service._config.diarization is None
    assert service._config.speaker_diarization_config is None


# ---------------------------------------------------------------------------
# Turn-mode gating — the single source of "does the service own turns?"
# ---------------------------------------------------------------------------


def test_service_closes_turns_true_for_vad():
    """VAD mode means the service endpoints and emits turn frames; the gate must say so."""
    service = _service(
        settings=SpeechmaticsSTTService.Settings(turn_detection_mode=TurnDetectionMode.VAD)
    )
    assert service._service_closes_turns is True


def test_service_closes_turns_false_for_external():
    """EXTERNAL mode hands endpointing to the caller; the gate must be False so turn
    frames, turn-event subscriptions, and processing metrics stay off."""
    service = _service(
        settings=SpeechmaticsSTTService.Settings(turn_detection_mode=TurnDetectionMode.EXTERNAL)
    )
    assert service._service_closes_turns is False


# ---------------------------------------------------------------------------
# send_message contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_message_without_connection_raises():
    """The documented contract: sending with no live session raises. Without this the
    call silently no-ops (or defers a None-deref into a background task)."""
    service = _service()  # never connected, so _client is None
    with pytest.raises(RuntimeError):
        await service.send_message("SomeMessage")


@pytest.mark.asyncio
async def test_send_message_propagates_send_failure():
    """A failure from a live client must reach the caller. This locks the fix that
    awaits the send instead of firing it off in an untracked task (where the error
    would be swallowed) — a passing no-connection test alone cannot catch that."""

    class _FailingClient:
        async def send_message(self, payload):
            raise ValueError("bad payload")

    service = _service()
    service._client = _FailingClient()

    with pytest.raises(RuntimeError):
        await service.send_message("SomeMessage")


# ---------------------------------------------------------------------------
# _segment_to_frame — pure Segment -> Pipecat frame mapping
# ---------------------------------------------------------------------------


def test_segment_to_frame_final_vs_interim_type():
    """`finalized` selects the frame type; downstream aggregators treat final and
    interim transcripts differently, so the mapping must honor it."""
    service = _service()
    segment = Segment(transcript="hello", speaker="S1")

    final = service._segment_to_frame(segment, finalized=True)
    interim = service._segment_to_frame(segment, finalized=False)
    assert isinstance(final, TranscriptionFrame)
    assert isinstance(interim, InterimTranscriptionFrame)


def test_segment_to_frame_applies_speaker_format():
    """The configured speaker_active_format must shape the emitted text and the speaker
    must become the frame's user_id — that is how per-speaker context reaches the LLM."""
    service = _service(
        settings=SpeechmaticsSTTService.Settings(speaker_active_format="@{speaker_id}: {text}")
    )
    frame = service._segment_to_frame(Segment(transcript="hi", speaker="S1"), finalized=True)

    assert frame.text == "@S1: hi"
    assert frame.user_id == "S1"


# ---------------------------------------------------------------------------
# _locale_to_speechmatics_locale — regional output locale mapping
# ---------------------------------------------------------------------------


def test_locale_maps_regional_variant():
    """A regional English variant must map to its Speechmatics output locale code."""
    assert _service()._locale_to_speechmatics_locale("en", Language.EN_GB) == "en-GB"


def test_locale_none_without_regional_variant():
    """A base language with no regional variant must yield no output locale, so the
    engine is not handed a spurious locale."""
    assert _service()._locale_to_speechmatics_locale("en", Language.EN) is None


def test_unsupported_regional_variant_constructs_and_falls_back():
    """An English variant with no Speechmatics output locale is built from __init__,
    before the processor has a name, so the fallback path must not touch `self`."""
    service = _service(settings=SpeechmaticsSTTService.Settings(language=Language.EN_IN))
    assert service._config.language == "en"
    assert service._config.output_locale is None


# ---------------------------------------------------------------------------
# Reconnect — the self-healing for connect/send failures. It runs inside
# STTService._reconnect(), which buffers and replays audio for the whole call.
# ---------------------------------------------------------------------------


def _stub_reconnect_attempts(service, monkeypatch, outcomes: list[bool]) -> dict:
    """Drive `_do_reconnect` through `outcomes`, one per attempt, without sleeping."""
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    service._disconnect = AsyncMock()
    attempts = {"n": 0}

    async def fake_open(report_error=True):
        attempts["n"] += 1
        return outcomes[attempts["n"] - 1]

    service._open_connection = fake_open
    return attempts


@pytest.mark.asyncio
async def test_do_reconnect_retries_until_success(monkeypatch):
    """A transient drop must be retried — the core guarantee that one failure does not
    permanently deafen the session."""
    service = _service()
    attempts = _stub_reconnect_attempts(service, monkeypatch, [False, False, True])

    await service._do_reconnect()

    assert attempts["n"] == 3


@pytest.mark.asyncio
async def test_do_reconnect_raises_once_attempts_are_exhausted(monkeypatch):
    """Retries are bounded, and exhausting them must raise so STTService._reconnect
    reports the failure instead of leaving the session silently dead."""
    service = _service()
    attempts = _stub_reconnect_attempts(
        service, monkeypatch, [False] * service.RECONNECT_MAX_ATTEMPTS
    )

    with pytest.raises(ConnectionError):
        await service._do_reconnect()

    assert attempts["n"] == service.RECONNECT_MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_do_reconnect_stops_when_session_is_rejected(monkeypatch):
    """A rejected session (`_closed`) will not clear on retry, so the loop must stop at
    once rather than spinning against a permanent error."""
    service = _service()
    attempts = _stub_reconnect_attempts(service, monkeypatch, [False, True])
    open_connection = service._open_connection

    async def reject(report_error=True):
        service._closed = True  # what _fail_fatally does
        return await open_connection(report_error=report_error)

    service._open_connection = reject

    await service._do_reconnect()

    assert attempts["n"] == 1


@pytest.mark.asyncio
async def test_disconnect_drains_message_queue():
    """Messages buffered from one session must not survive into the next. The consumer
    task is cancelled on disconnect, so anything left queued would be replayed by the
    fresh consumer started on reconnect (the queue is reused). Disconnect must clear it."""
    service = _service()
    # Simulate messages the client buffered but the (now-cancelled) consumer never drained.
    service._stt_msg_queue.put_nowait({"message": "AddSegment", "stale": True})
    service._stt_msg_queue.put_nowait({"message": "EndOfTurn", "stale": True})
    assert service._stt_msg_queue.qsize() == 2

    await service._disconnect()  # no client/tasks set — exercises the drain path only

    assert service._stt_msg_queue.empty()


# ---------------------------------------------------------------------------
# _is_auth_rejection — classifying a rejected credential out of the generic
# ConnectionError so a bad key is fatal, not retried forever.
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


class _NewStyleInvalidStatus(Exception):
    """Shape of websockets>=13 InvalidStatus: status on `.response.status_code`."""

    def __init__(self, status_code: int):
        self.response = _FakeResponse(status_code)
        super().__init__(f"server rejected WebSocket connection: HTTP {status_code}")


class _LegacyInvalidStatusCode(Exception):
    """Shape of legacy websockets InvalidStatusCode: status on `.status_code`."""

    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f"server rejected WebSocket connection: HTTP {status_code}")


@pytest.mark.parametrize("status", [401, 403])
def test_is_auth_rejection_new_style_status(status):
    """A websockets>=13 handshake rejection exposes the status on `.response.status_code`."""
    assert _is_auth_rejection(_NewStyleInvalidStatus(status)) is True


@pytest.mark.parametrize("status", [401, 403])
def test_is_auth_rejection_legacy_status(status):
    """The legacy websockets handshake rejection exposes it on `.status_code`."""
    assert _is_auth_rejection(_LegacyInvalidStatusCode(status)) is True


def test_is_auth_rejection_reads_chained_cause():
    """The SDK re-wraps the handshake error in a ConnectionError; the status must still be
    found through the exception chain (__cause__/__context__), not just the top exception."""
    try:
        try:
            raise _NewStyleInvalidStatus(401)
        except Exception as inner:
            raise ConnectionError("WebSocket connection error") from inner
    except ConnectionError as wrapped:
        assert _is_auth_rejection(wrapped) is True


def test_is_auth_rejection_message_fallback():
    """When only the status text survives (no structured attribute), the message is used."""
    assert (
        _is_auth_rejection(
            ConnectionError("WebSocket connection error: server rejected connection: HTTP 403")
        )
        is True
    )


def test_is_auth_rejection_false_for_transient_drop():
    """A plain network drop carries no auth status and must stay retryable (not fatal)."""
    assert _is_auth_rejection(ConnectionError("WebSocket connection error: timed out")) is False


def test_is_auth_rejection_false_for_other_http_status():
    """A non-auth handshake status (e.g. 500) is not an auth rejection."""
    assert _is_auth_rejection(_NewStyleInvalidStatus(500)) is False


class _StubClient:
    """Minimal AgentSttAsyncClient stand-in whose connect() raises a chosen error."""

    def __init__(self, error: Exception):
        self._error = error

    def __call__(self, *args, **kwargs):  # constructed as AgentSttAsyncClient(...)
        return self

    def on(self, *args, **kwargs):
        pass

    async def connect(self):
        raise self._error


def _connection_error_with_status(status: int) -> ConnectionError:
    try:
        raise _NewStyleInvalidStatus(status)
    except Exception as inner:
        try:
            raise ConnectionError("WebSocket connection error") from inner
        except ConnectionError as wrapped:
            return wrapped


@pytest.mark.asyncio
async def test_open_connection_auth_rejection_is_fatal(monkeypatch):
    """A 401 handshake rejection must stop the session (fatal error, no reconnect), not
    fall into the retryable branch that reconnects forever."""
    service = _service()
    service.push_error = AsyncMock()
    monkeypatch.setattr(
        "pipecat.services.speechmatics.stt.AgentSttAsyncClient",
        _StubClient(_connection_error_with_status(401)),
    )

    ok = await service._open_connection(report_error=True)

    assert ok is False
    assert service._closed is True  # _fail_fatally ran → no reconnect
    service.push_error.assert_awaited_once()
    assert service.push_error.call_args.kwargs.get("fatal") is True


@pytest.mark.asyncio
async def test_open_connection_transient_drop_stays_retryable(monkeypatch):
    """A plain connection drop must remain retryable — surfaced, but not fatal — so the
    reconnect loop can heal it."""
    service = _service()
    service.push_error = AsyncMock()
    monkeypatch.setattr(
        "pipecat.services.speechmatics.stt.AgentSttAsyncClient",
        _StubClient(ConnectionError("WebSocket connection error: timed out")),
    )

    ok = await service._open_connection(report_error=True)

    assert ok is False
    assert service._closed is False  # still retryable
    service.push_error.assert_awaited_once()
    assert service.push_error.call_args.kwargs.get("fatal") is not True


# ---------------------------------------------------------------------------
# _update_settings — runtime model / operating_point re-resolution
#
# `_build_config` reads only `s.model`, and `_resolve_model` runs once at
# construction. A runtime settings update must re-fold `operating_point` (and
# `model`) into `model`, or the reconnect it triggers rebuilds the *same* config
# — an audio gap that changes nothing, silently.
# ---------------------------------------------------------------------------


def _stub_reconnect(service) -> None:
    """Neutralize the connection side effects so _update_settings exercises only
    the settings/config logic (no socket, no pipeline)."""
    service._disconnect = AsyncMock()
    service._request_reconnect = AsyncMock()
    service.set_usable = AsyncMock()


@pytest.mark.asyncio
async def test_update_settings_operating_point_reresolves_into_model():
    """Changing the deprecated `operating_point` at runtime must update `model` (the
    only field the wire config reads) and rebuild the config to match — not silently
    reconnect onto the model it started with."""
    service = _service()  # model defaults to linden-1, operating_point unset
    _stub_reconnect(service)

    with pytest.warns(DeprecationWarning):
        await service._update_settings(SpeechmaticsSTTService.Settings(operating_point="linden-2"))

    assert service._settings.model == "linden-2"
    assert service._config.model == "linden-2"  # config was rebuilt with the new model
    service._request_reconnect.assert_awaited_once()  # a reconnect actually happened


@pytest.mark.asyncio
async def test_update_settings_operating_point_does_not_clash_with_resolved_model():
    """The landmine: after construction `model` is the *resolved* string, so a naive
    re-resolve of both fields would raise (model != operating_point). A lone
    `operating_point` update must win on its own instead of raising."""
    service = _service(settings=SpeechmaticsSTTService.Settings(model="linden-1"))
    _stub_reconnect(service)

    # would raise ValueError if resolved against the stale model="linden-1"
    with pytest.warns(DeprecationWarning):
        await service._update_settings(SpeechmaticsSTTService.Settings(operating_point="linden-2"))

    assert service._settings.model == "linden-2"


@pytest.mark.asyncio
async def test_update_settings_model_reresolves_into_model():
    """Changing `model` directly at runtime must take effect in the rebuilt config."""
    service = _service()
    _stub_reconnect(service)

    await service._update_settings(SpeechmaticsSTTService.Settings(model="linden-2"))

    assert service._settings.model == "linden-2"
    assert service._config.model == "linden-2"


@pytest.mark.asyncio
async def test_update_settings_unrelated_field_leaves_model_untouched():
    """An update that touches neither `model` nor `operating_point` must not re-resolve
    (which would otherwise re-run the deprecation/validation path spuriously)."""
    service = _service()  # model resolved to linden-1
    _stub_reconnect(service)

    await service._update_settings(SpeechmaticsSTTService.Settings(domain="finance"))

    assert service._settings.model == "linden-1"
    assert service._settings.domain == "finance"
