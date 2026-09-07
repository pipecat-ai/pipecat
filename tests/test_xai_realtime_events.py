#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for xAI (Grok) Realtime server event parsing and handling."""

import json

import pytest

from pipecat.services.xai.realtime import events


def _event(payload: dict) -> str:
    return json.dumps(payload)


def test_parse_session_created_from_live_shape():
    """session.created is emitted on every connect and must parse cleanly."""
    raw = _event(
        {
            "type": "session.created",
            "event_id": "evt-1",
            "session": {
                "id": "sess-1",
                "object": "realtime.session",
                "instructions": "",
                "voice": "xai_ara",
                "modalities": ["audio"],
                "turn_detection": {"type": None},
                "tools": [],
                "model": "grok-voice-think-fast-2.0",
            },
        }
    )

    evt = events.parse_server_event(raw)

    assert isinstance(evt, events.SessionCreatedEvent)
    assert evt.session.voice == "xai_ara"
    assert evt.session.turn_detection is not None
    assert evt.session.turn_detection.type is None


def test_parse_transcription_updated_is_cumulative():
    raw = _event(
        {
            "type": "conversation.item.input_audio_transcription.updated",
            "event_id": "evt-2",
            "item_id": "item-1",
            "content_index": 0,
            "transcript": "hello there",
        }
    )

    evt = events.parse_server_event(raw)

    assert isinstance(evt, events.ConversationItemInputAudioTranscriptionUpdated)
    assert evt.transcript == "hello there"


@pytest.mark.parametrize(
    "event_type",
    ["response.output_text.delta", "response.text.delta"],
)
def test_parse_text_delta_aliases(event_type):
    raw = _event(
        {
            "type": event_type,
            "event_id": "evt-3",
            "response_id": "resp-1",
            "item_id": "item-1",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hi",
        }
    )

    evt = events.parse_server_event(raw)

    assert isinstance(evt, events.ResponseTextDelta)
    assert evt.delta == "Hi"


@pytest.mark.parametrize(
    "event_type",
    [
        "conversation.item.deleted",
        "conversation.item.truncated",
        "input_audio_buffer.timeout_triggered",
        "input_audio_buffer.dtmf_event_received",
        "mcp_list_tools.in_progress",
        "mcp_list_tools.completed",
        "mcp_list_tools.failed",
        "response.mcp_call_arguments.delta",
        "response.mcp_call_arguments.done",
        "response.mcp_call.in_progress",
        "response.mcp_call.completed",
        "response.mcp_call.failed",
    ],
)
def test_documented_server_events_are_registered(event_type):
    """Every documented server event type should parse without raising."""
    payload = {"type": event_type, "event_id": "evt-x"}
    if event_type == "conversation.item.deleted":
        payload["item_id"] = "item-1"
    elif event_type == "conversation.item.truncated":
        payload.update({"item_id": "item-1", "content_index": 0, "audio_end_ms": 250})
    elif event_type == "input_audio_buffer.timeout_triggered":
        payload["item_id"] = "item-1"
    elif event_type == "input_audio_buffer.dtmf_event_received":
        payload["digit"] = "5"
    elif "mcp_call_arguments" in event_type:
        payload.update({"delta": "{}", "arguments": "{}", "name": "tool", "call_id": "c1"})

    evt = events.parse_server_event(_event(payload))
    assert evt.type == event_type


def test_unknown_event_still_raises():
    with pytest.raises(Exception, match="Unimplemented server event type"):
        events.parse_server_event(_event({"type": "not.a.real.event", "event_id": "e"}))


def test_client_truncate_and_delete_events_serialize():
    truncate = events.ConversationItemTruncateEvent(
        item_id="item-1", content_index=0, audio_end_ms=250
    )
    delete = events.ConversationItemDeleteEvent(item_id="item-1")
    assert truncate.model_dump(exclude_none=True)["type"] == "conversation.item.truncate"
    assert delete.model_dump(exclude_none=True)["type"] == "conversation.item.delete"


def test_session_properties_accept_extended_fields():
    props = events.SessionProperties(
        reasoning=events.Reasoning(effort="none"),
        resumption=events.SessionResumption(enabled=True),
        replace={"Acme Mobile": "Acme Mobull"},
        turn_detection=events.TurnDetection(
            type="server_vad",
            idle_timeout_ms=5000,
            threshold=0.85,
        ),
        audio=events.AudioConfiguration(
            input=events.AudioInput(
                transcription=events.InputAudioTranscription(model="grok-transcribe")
            )
        ),
    )
    dumped = props.model_dump(exclude_none=True)
    assert dumped["reasoning"]["effort"] == "none"
    assert dumped["resumption"]["enabled"] is True
    assert dumped["turn_detection"]["idle_timeout_ms"] == 5000
    assert dumped["audio"]["input"]["transcription"]["model"] == "grok-transcribe"
