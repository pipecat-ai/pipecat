#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the xAI (Grok) Realtime session parameters.

Covers the session fields xAI documents for the Voice Agent API that the
service exposes:

- ``turn_detection``: threshold, prefix_padding_ms, silence_duration_ms,
  idle_timeout_ms
- ``audio.input.transcription``: language_hint, keyterms
- ``audio.output.speed``
- ``reasoning.effort``

Every one of these defaults to ``None`` and client events are serialized with
``exclude_none=True``, so the wire payload stays byte-identical unless a field
is explicitly set. The "defaults" tests below pin that guarantee.
"""

from pipecat.services.xai.realtime import events
from pipecat.services.xai.realtime.llm import GrokRealtimeLLMService

# ---------------------------------------------------------------------------
# Turn detection
# ---------------------------------------------------------------------------


def test_turn_detection_defaults_stay_off_the_wire():
    """An unconfigured TurnDetection serializes exactly as it did before."""
    td = events.TurnDetection()

    assert td.threshold is None
    assert td.prefix_padding_ms is None
    assert td.silence_duration_ms is None
    assert td.idle_timeout_ms is None
    assert td.model_dump(exclude_none=True) == {"type": "server_vad"}


def test_turn_detection_accepts_vad_tuning():
    """Confirm the wire shape matches the documented xAI schema."""
    td = events.TurnDetection(
        threshold=0.6,
        prefix_padding_ms=300,
        silence_duration_ms=800,
        idle_timeout_ms=12000,
    )

    assert td.model_dump(exclude_none=True) == {
        "type": "server_vad",
        "threshold": 0.6,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 800,
        "idle_timeout_ms": 12000,
    }


def test_turn_detection_coerces_dict():
    """Pydantic coerces a nested dict into TurnDetection automatically."""
    sp = events.SessionProperties.model_validate(
        {"turn_detection": {"type": "server_vad", "threshold": 0.4}}
    )

    assert isinstance(sp.turn_detection, events.TurnDetection)
    assert sp.turn_detection.threshold == 0.4


# ---------------------------------------------------------------------------
# Input audio transcription
# ---------------------------------------------------------------------------


def test_input_audio_transcription_defaults_stay_off_the_wire():
    assert events.InputAudioTranscription().model_dump(exclude_none=True) == {}


def test_input_audio_transcription_round_trip():
    transcription = events.InputAudioTranscription(
        language_hint="en",
        keyterms=["Pipecat", "Daily"],
    )

    assert transcription.model_dump(exclude_none=True) == {
        "language_hint": "en",
        "keyterms": ["Pipecat", "Daily"],
    }


def test_audio_input_carries_transcription():
    audio_input = events.AudioInput(
        format=events.PCMAudioFormat(rate=8000),
        transcription=events.InputAudioTranscription(language_hint="es"),
    )

    assert audio_input.model_dump(exclude_none=True) == {
        "format": {"type": "audio/pcm", "rate": 8000},
        "transcription": {"language_hint": "es"},
    }


def test_audio_input_without_transcription_is_unchanged():
    audio_input = events.AudioInput(format=events.PCMAudioFormat(rate=24000))

    assert audio_input.model_dump(exclude_none=True) == {
        "format": {"type": "audio/pcm", "rate": 24000}
    }


# ---------------------------------------------------------------------------
# Output speed
# ---------------------------------------------------------------------------


def test_audio_output_carries_speed():
    audio_output = events.AudioOutput(format=events.PCMAudioFormat(rate=24000), speed=1.2)

    assert audio_output.model_dump(exclude_none=True) == {
        "format": {"type": "audio/pcm", "rate": 24000},
        "speed": 1.2,
    }


def test_audio_output_without_speed_is_unchanged():
    audio_output = events.AudioOutput(format=events.PCMAudioFormat(rate=24000))

    assert audio_output.model_dump(exclude_none=True) == {
        "format": {"type": "audio/pcm", "rate": 24000}
    }


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------


def test_session_properties_accepts_reasoning_object():
    sp = events.SessionProperties(reasoning=events.Reasoning(effort="none"))

    assert sp.reasoning is not None
    assert sp.reasoning.effort == "none"


def test_session_properties_coerces_reasoning_dict():
    sp = events.SessionProperties.model_validate({"reasoning": {"effort": "high"}})

    assert isinstance(sp.reasoning, events.Reasoning)
    assert sp.reasoning.effort == "high"


def test_reasoning_absent_by_default():
    sp = events.SessionProperties()

    assert sp.reasoning is None
    assert "reasoning" not in sp.model_dump(exclude_none=True)


# ---------------------------------------------------------------------------
# Full session.update wire shape
# ---------------------------------------------------------------------------


def test_session_update_serializes_every_new_field():
    sp = events.SessionProperties(
        instructions="You are a helpful assistant.",
        voice="Ara",
        turn_detection=events.TurnDetection(
            threshold=0.6,
            prefix_padding_ms=300,
            silence_duration_ms=800,
            idle_timeout_ms=12000,
        ),
        audio=events.AudioConfiguration(
            input=events.AudioInput(
                format=events.PCMAudioFormat(rate=8000),
                transcription=events.InputAudioTranscription(
                    language_hint="en",
                    keyterms=["Pipecat"],
                ),
            ),
            output=events.AudioOutput(format=events.PCMAudioFormat(rate=8000), speed=1.1),
        ),
        reasoning=events.Reasoning(effort="none"),
    )

    session = events.SessionUpdateEvent(session=sp).model_dump(exclude_none=True)["session"]

    assert session["turn_detection"]["threshold"] == 0.6
    assert session["turn_detection"]["idle_timeout_ms"] == 12000
    assert session["audio"]["input"]["transcription"]["language_hint"] == "en"
    assert session["audio"]["input"]["transcription"]["keyterms"] == ["Pipecat"]
    assert session["audio"]["output"]["speed"] == 1.1
    assert session["reasoning"] == {"effort": "none"}


def test_session_update_default_shape_is_unchanged():
    """A default session serializes to the same payload as before this change.

    None of the new fields appear unless they are set — that is the whole
    backwards-compatibility claim, so pin the exact wire shape.
    """
    session = events.SessionUpdateEvent(session=events.SessionProperties()).model_dump(
        exclude_none=True
    )["session"]

    assert session == {"voice": "eve", "turn_detection": {"type": "server_vad"}}


# ---------------------------------------------------------------------------
# Audio config backfill
# ---------------------------------------------------------------------------


def _configured_props(session_properties: events.SessionProperties) -> events.SessionProperties:
    """Run ``_ensure_audio_config`` and return the service's own session properties.

    The service stores its settings as a Pydantic model, so reading back the
    caller's instance would not show the backfill.
    """
    service = GrokRealtimeLLMService(
        api_key="test-key",
        settings=GrokRealtimeLLMService.Settings(session_properties=session_properties),
    )
    service._ensure_audio_config(8000, 24000)
    return service._settings.session_properties


def test_ensure_audio_config_fills_missing_formats():
    """A user-supplied AudioInput without a format still gets the pipeline rate.

    Without the backfill, configuring ``transcription`` (which requires building
    an ``AudioInput``) would drop the sample rate from the session.
    """
    props = _configured_props(
        events.SessionProperties(
            audio=events.AudioConfiguration(
                input=events.AudioInput(
                    transcription=events.InputAudioTranscription(language_hint="en")
                ),
                output=events.AudioOutput(speed=1.3),
            )
        )
    )

    assert props.audio.input.format == events.PCMAudioFormat(rate=8000)
    assert props.audio.output.format == events.PCMAudioFormat(rate=24000)
    # The user's own values survive the backfill.
    assert props.audio.input.transcription.language_hint == "en"
    assert props.audio.output.speed == 1.3


def test_ensure_audio_config_keeps_user_formats():
    props = _configured_props(
        events.SessionProperties(
            audio=events.AudioConfiguration(
                input=events.AudioInput(format=events.PCMUAudioFormat()),
                output=events.AudioOutput(format=events.PCMAudioFormat(rate=16000)),
            )
        )
    )

    assert props.audio.input.format == events.PCMUAudioFormat()
    assert props.audio.output.format == events.PCMAudioFormat(rate=16000)


def test_ensure_audio_config_creates_missing_audio():
    props = _configured_props(events.SessionProperties())

    assert props.audio.input.format == events.PCMAudioFormat(rate=8000)
    assert props.audio.output.format == events.PCMAudioFormat(rate=24000)
