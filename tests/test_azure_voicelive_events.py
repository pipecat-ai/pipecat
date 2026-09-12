#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Azure Voice Live server event parsing and session serialization.

The payloads below are captured from a live Voice Live session
(api-version 2026-07-15, model gpt-4o-mini), with audio deltas truncated.
"""

import json

from pipecat.services.azure.voicelive import events


def _event(payload: dict) -> str:
    return json.dumps(payload)


def test_parse_session_created():
    """session.created is the first event after connecting."""
    raw = _event(
        {
            "event_id": "event_1",
            "type": "session.created",
            "session": {"id": "sess_1", "object": "realtime.session", "model": "gpt-4o-mini"},
        }
    )

    evt = events.parse_server_event(raw)

    assert isinstance(evt, events.SessionCreatedEvent)
    assert evt.session["model"] == "gpt-4o-mini"


def test_parse_conversation_item_created_with_empty_previous_item_id():
    """The service sends an empty string, not null, for the first item."""
    raw = _event(
        {
            "event_id": "event_2",
            "type": "conversation.item.created",
            "previous_item_id": "",
            "item": {
                "id": "item_1",
                "object": "realtime.item",
                "type": "message",
                "status": "completed",
                "role": "user",
                "content": [{"type": "input_text", "text": "Say hello in five words."}],
            },
        }
    )

    evt = events.parse_server_event(raw)

    assert isinstance(evt, events.ConversationItemCreated)
    assert evt.previous_item_id == ""
    assert evt.item.role == "user"
    assert evt.item.content is not None
    assert evt.item.content[0].text == "Say hello in five words."


def test_parse_output_item_added_is_incomplete():
    """An assistant item is announced before its content exists."""
    raw = _event(
        {
            "event_id": "event_3",
            "type": "response.output_item.added",
            "response_id": "resp_1",
            "output_index": 0,
            "item": {
                "id": "msg_1",
                "object": "realtime.item",
                "type": "message",
                "status": "incomplete",
                "role": "assistant",
                "content": [],
            },
        }
    )

    evt = events.parse_server_event(raw)

    assert isinstance(evt, events.ResponseOutputItemAdded)
    assert evt.item.status == "incomplete"
    assert evt.item.role == "assistant"


def test_parse_audio_delta():
    """Voice Live uses the original event name, not response.output_audio.delta."""
    raw = _event(
        {
            "event_id": "event_4",
            "type": "response.audio.delta",
            "response_id": "resp_1",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "R/8N/5/+Y/6T/gn/df/E/ycAmACqAEcAIQBRAAMA",
        }
    )

    evt = events.parse_server_event(raw)

    assert isinstance(evt, events.ResponseAudioDelta)
    assert evt.item_id == "msg_1"
    assert evt.delta


def test_parse_audio_transcript_delta():
    raw = _event(
        {
            "event_id": "event_5",
            "type": "response.audio_transcript.delta",
            "response_id": "resp_1",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hello",
        }
    )

    evt = events.parse_server_event(raw)

    assert isinstance(evt, events.ResponseAudioTranscriptDelta)
    assert evt.delta == "Hello"


def test_parse_response_done_exposes_usage_and_status():
    """Token usage drives LLM usage metrics, and carries unmodeled extras."""
    raw = _event(
        {
            "event_id": "event_6",
            "type": "response.done",
            "response": {
                "object": "realtime.response",
                "id": "resp_1",
                "status": "completed",
                "status_details": None,
                "output": [
                    {
                        "id": "msg_1",
                        "object": "realtime.item",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "audio", "transcript": "Hello! How are you today?"}],
                    }
                ],
                "usage": {
                    "total_tokens": 65,
                    "input_tokens": 23,
                    "output_tokens": 42,
                    "input_token_details": {
                        "cached_tokens": 0,
                        "text_tokens": 23,
                        "audio_tokens": 0,
                        "image_tokens": 0,
                    },
                    "output_token_details": {"text_tokens": 10, "audio_tokens": 32},
                },
            },
        }
    )

    evt = events.parse_server_event(raw)

    assert isinstance(evt, events.ResponseDone)
    assert evt.status == "completed"
    usage = evt.usage
    assert usage is not None
    assert usage.total_tokens == 65
    assert usage.input_tokens == 23
    assert usage.output_tokens == 42


def test_parse_function_call_arguments_done():
    raw = _event(
        {
            "event_id": "event_7",
            "type": "response.function_call_arguments.done",
            "response_id": "resp_1",
            "item_id": "fc_1",
            "output_index": 0,
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": '{"city":"Pune"}',
        }
    )

    evt = events.parse_server_event(raw)

    assert isinstance(evt, events.ResponseFunctionCallArgumentsDone)
    assert evt.call_id == "call_1"
    assert json.loads(evt.arguments) == {"city": "Pune"}


def test_parse_error_event():
    raw = _event(
        {
            "event_id": "event_8",
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "code": "response_cancel_not_active",
                "message": "No active response to cancel.",
            },
        }
    )

    evt = events.parse_server_event(raw)

    assert isinstance(evt, events.ErrorEvent)
    assert evt.error.code == "response_cancel_not_active"


def test_parse_unmodeled_event_returns_none():
    """Avatar and animation events are ignored rather than raising."""
    raw = _event({"type": "session.avatar.switch_to_speaking", "event_id": "event_9"})

    assert events.parse_server_event(raw) is None


def test_parse_word_timestamp_delta():
    raw = _event(
        {
            "event_id": "event_10",
            "type": "response.audio_timestamp.delta",
            "response_id": "resp_1",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "audio_offset_ms": 120,
            "audio_duration_ms": 220,
            "text": "Hello",
            "timestamp_type": "word",
        }
    )

    evt = events.parse_server_event(raw)

    assert isinstance(evt, events.ResponseAudioTimestampDelta)
    assert evt.text == "Hello"
    assert evt.audio_offset_ms == 120


def test_session_update_serializes_flat():
    """Voice Live carries audio settings on the session, not under `audio`."""
    session = events.SessionProperties(
        instructions="Be brief.",
        modalities=["text", "audio"],
        voice=events.AzureStandardVoice(name="en-US-Ava:DragonHDLatestNeural"),
        input_audio_format="pcm16",
        output_audio_format="pcm16",
        input_audio_sampling_rate=16000,
        turn_detection=events.TurnDetection(type="azure_semantic_vad", silence_duration_ms=500),
        input_audio_transcription=events.InputAudioTranscription(model="azure-speech"),
        input_audio_noise_reduction=events.InputAudioNoiseReduction(),
    )

    payload = events.SessionUpdateEvent(session=session).model_dump(exclude_none=True)

    assert payload["type"] == "session.update"
    assert "audio" not in payload["session"]
    assert payload["session"]["turn_detection"]["type"] == "azure_semantic_vad"
    assert payload["session"]["voice"] == {
        "type": "azure-standard",
        "name": "en-US-Ava:DragonHDLatestNeural",
    }
    assert payload["session"]["input_audio_sampling_rate"] == 16000
    assert payload["session"]["input_audio_noise_reduction"]["type"] == (
        "azure_deep_noise_suppression"
    )


def test_unset_session_fields_are_omitted():
    """Only what the caller configured is sent, so server defaults stand."""
    payload = events.SessionUpdateEvent(
        session=events.SessionProperties(instructions="Be brief.")
    ).model_dump(exclude_none=True)

    assert payload["session"] == {"instructions": "Be brief."}


def test_disabled_turn_detection_is_sent_as_an_explicit_null():
    """Omitting the field instead leaves the service's own server VAD running."""
    payload = events.SessionUpdateEvent(
        session=events.SessionProperties(instructions="Be brief.", turn_detection=None)
    ).model_dump(exclude_none=True)

    assert payload["session"]["turn_detection"] is None


def test_turn_detection_false_also_disables():
    payload = events.SessionUpdateEvent(
        session=events.SessionProperties(instructions="Be brief.", turn_detection=False)
    ).model_dump(exclude_none=True)

    assert payload["session"]["turn_detection"] is None


def test_unset_turn_detection_is_omitted():
    """Leaving it unset keeps the service's default turn detection."""
    payload = events.SessionUpdateEvent(
        session=events.SessionProperties(instructions="Be brief.")
    ).model_dump(exclude_none=True)

    assert "turn_detection" not in payload["session"]
