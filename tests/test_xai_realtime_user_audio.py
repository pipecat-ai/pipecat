#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for user-audio and interruption handling in GrokRealtimeLLMService."""

import base64
from typing import Any

import pytest

from pipecat.frames.frames import InputAudioRawFrame
from pipecat.services.xai.realtime import events
from pipecat.services.xai.realtime.events import SessionProperties, TurnDetection
from pipecat.services.xai.realtime.llm import GrokRealtimeLLMService


class _EventRecorder:
    def __init__(self):
        self.events: list[Any] = []

    async def __call__(self, event):
        self.events.append(event)

    def kinds(self) -> list[str]:
        return [type(e).__name__ for e in self.events]


def _make_service(*, server_vad: bool) -> tuple[GrokRealtimeLLMService, _EventRecorder]:
    turn_detection = TurnDetection(type="server_vad") if server_vad else None
    service = GrokRealtimeLLMService(
        api_key="test-key",
        settings=GrokRealtimeLLMService.Settings(
            session_properties=SessionProperties(turn_detection=turn_detection),
        ),
    )
    recorder = _EventRecorder()
    service.send_client_event = recorder  # type: ignore[method-assign]

    async def _noop(*args, **kwargs):
        pass

    service.stop_all_metrics = _noop  # type: ignore[method-assign]
    return service, recorder


def _audio_frame(data: bytes = b"\xaa\xbb") -> InputAudioRawFrame:
    return InputAudioRawFrame(audio=data, sample_rate=24000, num_channels=1)


def test_default_model_is_grok_voice_latest():
    service = GrokRealtimeLLMService(api_key="test-key")
    assert service._settings.model == "grok-voice-latest"


@pytest.mark.asyncio
async def test_user_audio_dropped_until_session_ready():
    from unittest.mock import patch

    service, recorder = _make_service(server_vad=True)
    assert service._api_session_ready is False

    with patch("pipecat.services.xai.realtime.llm.logger.debug") as mock_debug:
        await service._send_user_audio(_audio_frame())
        await service._send_user_audio(_audio_frame())

    assert recorder.kinds() == []
    assert service._logged_audio_drop_before_session_ready is True
    drop_calls = [c for c in mock_debug.call_args_list if "Dropping user audio" in str(c)]
    assert len(drop_calls) == 1


@pytest.mark.asyncio
async def test_user_audio_flows_after_session_ready_without_conversation_setup():
    """Audio-only pipelines never call _create_response; audio must still flow."""
    service, recorder = _make_service(server_vad=True)
    service._api_session_ready = True
    assert service._llm_needs_conversation_setup is True

    await service._send_user_audio(_audio_frame(b"\x11\x22"))

    assert recorder.kinds() == ["InputAudioBufferAppendEvent"]
    assert recorder.events[0].audio == base64.b64encode(b"\x11\x22").decode()


@pytest.mark.asyncio
async def test_server_vad_interruption_cancels_without_clearing_input():
    service, recorder = _make_service(server_vad=True)
    service._api_session_ready = True

    await service._send_user_audio(_audio_frame())
    await service._handle_interruption()

    assert recorder.kinds() == [
        "InputAudioBufferAppendEvent",
        "ResponseCancelEvent",
    ]
    assert "InputAudioBufferClearEvent" not in recorder.kinds()


@pytest.mark.asyncio
async def test_manual_turn_interruption_clears_and_cancels():
    service, recorder = _make_service(server_vad=False)

    await service._handle_interruption()

    assert recorder.kinds() == [
        "InputAudioBufferClearEvent",
        "ResponseCancelEvent",
    ]


@pytest.mark.asyncio
async def test_interruption_truncates_in_flight_audio_on_the_wire():
    import time

    from pipecat.services.xai.realtime.events import (
        AudioConfiguration,
        AudioOutput,
        PCMAudioFormat,
    )
    from pipecat.services.xai.realtime.llm import CurrentAudioResponse

    service, recorder = _make_service(server_vad=True)
    service._settings.session_properties.audio = AudioConfiguration(
        output=AudioOutput(format=PCMAudioFormat(rate=24000))
    )
    service._current_audio_response = CurrentAudioResponse(
        item_id="item-audio",
        content_index=0,
        start_time_ms=int(time.time() * 1000) - 500,
        total_size=48000,  # 1s at 24kHz mono 16-bit
    )

    await service._handle_interruption()

    assert "ResponseCancelEvent" in recorder.kinds()
    assert "ConversationItemTruncateEvent" in recorder.kinds()
    truncate = next(
        e for e in recorder.events if isinstance(e, events.ConversationItemTruncateEvent)
    )
    assert truncate.item_id == "item-audio"
    assert truncate.content_index == 0
    assert 0 < truncate.audio_end_ms <= 1000
    assert service._current_audio_response is None


@pytest.mark.asyncio
async def test_delete_conversation_item_sends_client_event():
    service, recorder = _make_service(server_vad=True)
    await service.delete_conversation_item("item-1")
    assert recorder.kinds() == ["ConversationItemDeleteEvent"]
    assert recorder.events[0].item_id == "item-1"


@pytest.mark.asyncio
async def test_force_message_sends_force_message_item():
    service, recorder = _make_service(server_vad=True)
    await service.force_message("This call is being recorded.")
    assert recorder.kinds() == ["ConversationItemCreateEvent"]
    item = recorder.events[0].item
    assert item.type == "force_message"
    assert item.content[0].text == "This call is being recorded."
