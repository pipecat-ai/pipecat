#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json

from pipecat.services.openai.realtime import events


def test_parse_server_event_returns_none_for_unmodeled_type():
    raw = json.dumps({"event_id": "ev_1", "type": "output_audio_buffer.cleared"})

    assert events.parse_server_event(raw) is None


def test_parse_response_done_with_only_status():
    raw = json.dumps(
        {"event_id": "ev_1", "type": "response.done", "response": {"status": "completed"}}
    )

    evt = events.parse_server_event(raw)

    assert isinstance(evt, events.ResponseDone)
    assert evt.response.usage is None
    assert evt.response.status_details is None


def test_parse_session_updated_with_unmodeled_transcription_fields():
    """The server echoes transcription fields Pipecat may not model; parsing ignores them."""
    transcription = {
        "model": "gpt-live-transcribe",
        "language": None,
        "languages": ["en", "es"],
        "prompt": None,
        "a_field_openai_adds_later": True,
    }
    raw = json.dumps(
        {
            "event_id": "ev_1",
            "type": "session.updated",
            "session": {"type": "realtime", "audio": {"input": {"transcription": transcription}}},
        }
    )

    evt = events.parse_server_event(raw)

    assert isinstance(evt, events.SessionUpdatedEvent)
    parsed = evt.session.audio.input.transcription
    assert parsed.model == "gpt-live-transcribe"
    assert parsed.languages == ["en", "es"]


def test_session_update_serializes_retention_ratio_truncation():
    session = events.SessionProperties(
        truncation=events.TruncationRetentionRatio(
            retention_ratio=0.8,
            token_limits=events.TruncationTokenLimits(post_instructions=8000),
        )
    )

    dump = events.SessionUpdateEvent(session=session).model_dump(exclude_none=True)

    assert dump["session"]["truncation"] == {
        "type": "retention_ratio",
        "retention_ratio": 0.8,
        "token_limits": {"post_instructions": 8000},
    }
