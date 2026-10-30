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
