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

import pytest
from speechmatics.agent_stt import AudioEncoding, Model

from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
from pipecat.services.speechmatics.stt import (
    SpeechmaticsSTTService,
    TurnDetectionMode,
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
    service = _service(settings=SpeechmaticsSTTService.Settings(enable_diarization=True, max_speakers=2))
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
    assert _service(settings=SpeechmaticsSTTService.Settings(turn_detection_mode=TurnDetectionMode.VAD))._service_closes_turns is True


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

    assert isinstance(service._segment_to_frame(segment, finalized=True), TranscriptionFrame)
    assert isinstance(service._segment_to_frame(segment, finalized=False), InterimTranscriptionFrame)


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


# ---------------------------------------------------------------------------
# Reconnect loop — the self-healing added for connect/send failures
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconnect_loop_retries_until_success():
    """A transient drop must be retried until the connection returns — the core
    guarantee that one failure does not permanently deafen the session."""
    service = _service()
    service.RECONNECT_INITIAL_DELAY = 0.0
    service.RECONNECT_MAX_DELAY = 0.0
    service._client = None

    attempts = {"n": 0}

    async def fake_open(report_error=True):
        attempts["n"] += 1
        if attempts["n"] < 3:
            return False
        service._client = object()  # simulate a live connection
        return True

    service._open_connection = fake_open

    await service._reconnect_loop()

    assert attempts["n"] == 3
    assert service._client is not None
    assert service._reconnect_task is None  # loop clears its own handle on exit


def test_schedule_reconnect_noop_when_closed():
    """After teardown (`_closed`), no reconnect may be scheduled — otherwise a retry
    could fire against a session that is shutting down."""
    service = _service()
    service._closed = True

    service._schedule_reconnect()

    assert service._reconnect_task is None


def test_schedule_reconnect_noop_when_already_running():
    """A reconnect already in flight must not be duplicated; a second scheduling call
    has to leave the existing attempt untouched."""
    service = _service()
    sentinel = object()
    service._reconnect_task = sentinel

    service._schedule_reconnect()

    assert service._reconnect_task is sentinel
