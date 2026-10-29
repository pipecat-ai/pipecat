#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import time
from unittest.mock import AsyncMock

import pytest

from pipecat.frames.frames import (
    FunctionCallCancelFrame,
    FunctionCallResultFrame,
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    TTSTextFrame,
)
from pipecat.services.elevenlabs.realtime.llm import ElevenLabsRealtimeLLMService
from pipecat.services.openai.realtime import events


class _ClosedWebSocket:
    def __init__(self, close_code: int, close_reason: str):
        self.close_code = close_code
        self.close_reason = close_reason

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


def _service(**kwargs) -> ElevenLabsRealtimeLLMService:
    service = ElevenLabsRealtimeLLMService(api_key="test-key", **kwargs)
    service._ws_send = AsyncMock()
    service._websocket = object()
    return service


def _sent(service: ElevenLabsRealtimeLLMService) -> list[dict]:
    return [call.args[0] for call in service._ws_send.call_args_list]


def _session_update(**session) -> events.SessionUpdateEvent:
    return events.SessionUpdateEvent(session=events.SessionProperties(**session))


def _transcript_delta(delta: str) -> events.ResponseAudioTranscriptDelta:
    return events.ResponseAudioTranscriptDelta(
        event_id="ev",
        type="response.output_audio_transcript.delta",
        response_id="resp_1",
        item_id="item_1",
        output_index=0,
        content_index=0,
        delta=delta,
    )


@pytest.mark.asyncio
async def test_connects_with_xi_api_key_to_url_without_model_query():
    service = ElevenLabsRealtimeLLMService(api_key="test-key")

    assert await service._auth_headers() == {"xi-api-key": "test-key"}
    assert service.base_url == "wss://api.elevenlabs.io/v1/convai/realtime"
    assert service._settings.model == "eleven_realtime_v1_mini"


@pytest.mark.asyncio
async def test_first_session_update_carries_model_and_session_extensions():
    service = _service(max_duration_seconds=1200, input_pacing="bursty", turn_timeout=30)

    await service.send_client_event(_session_update(model="eleven_realtime_v1_mini"))

    session = _sent(service)[0]["session"]
    assert session["model"] == "eleven_realtime_v1_mini"
    assert session["max_duration_seconds"] == 1200
    assert session["audio"]["input"]["pacing"] == "bursty"
    assert session["audio"]["input"]["turn_detection"] == {
        "type": "semantic_vad",
        "turn_timeout": 30,
    }


@pytest.mark.asyncio
async def test_session_update_after_configuration_keeps_only_live_fields():
    service = _service()
    voice = events.AudioConfiguration(output=events.AudioOutput(voice="brian"))
    await service.send_client_event(_session_update(instructions="Be brief.", audio=voice))
    service._api_session_ready = True

    other_voice = events.AudioConfiguration(output=events.AudioOutput(voice="lily"))
    await service.send_client_event(
        _session_update(instructions="Be chatty.", tool_choice="auto", audio=other_voice)
    )

    assert _sent(service)[1]["session"] == {
        "type": "realtime",
        "instructions": "Be chatty.",
        "tool_choice": "auto",
    }


@pytest.mark.asyncio
async def test_user_audio_waits_for_the_configured_session():
    service = _service()
    frame = InputAudioRawFrame(audio=bytes(480), sample_rate=24000, num_channels=1)

    await service._send_user_audio(frame)
    assert _sent(service) == []

    service._api_session_ready = True
    await service._send_user_audio(frame)
    assert _sent(service)[0]["type"] == "input_audio_buffer.append"


@pytest.mark.asyncio
async def test_system_message_items_are_sent_as_user_items():
    service = _service()

    await service._send_message_item("Context for the model.")
    await service._send_message_item("Earlier reply.", role="assistant")

    roles = [message["item"]["role"] for message in _sent(service)]
    assert roles == ["user", "assistant"]


@pytest.mark.asyncio
async def test_audio_tags_are_stripped_across_transcript_deltas():
    service = _service()
    service.push_frame = AsyncMock()
    service.stop_processing_metrics = AsyncMock()
    service._call_event_handler = AsyncMock()

    for delta in ["Hi [chee", "rful] there, item [1"]:
        await service._handle_evt_audio_transcript_delta(_transcript_delta(delta))
    await service._handle_evt_response_done(
        events.ResponseDone.model_validate(
            {"event_id": "ev", "type": "response.done", "response": {"status": "completed"}}
        )
    )

    frames = [call.args[0] for call in service.push_frame.call_args_list]
    texts = [frame.text for frame in frames if isinstance(frame, TTSTextFrame)]
    assert texts == ["Hi ", " there, item ", "[1"]
    assert isinstance(frames[-1], LLMFullResponseEndFrame)


@pytest.mark.asyncio
async def test_cancelled_function_call_is_not_answered():
    service = _service()
    service._open_function_calls.add("call_1")

    await service._handle_function_call_cancel(
        FunctionCallCancelFrame(function_name="get_weather", tool_call_id="call_1")
    )

    assert _sent(service) == []


@pytest.mark.asyncio
async def test_result_after_the_server_deadline_is_sent_as_a_message():
    service = _service()
    service._open_function_calls.add("call_1")
    service._function_call_opened_at["call_1"] = time.monotonic() - 21

    await service._handle_function_call_result(
        FunctionCallResultFrame(
            function_name="get_weather",
            tool_call_id="call_1",
            arguments={},
            result={"temperature": 24},
        )
    )

    item = _sent(service)[0]["item"]
    assert item["type"] == "message"
    assert item["role"] == "user"


@pytest.mark.asyncio
async def test_unexpected_close_pushes_an_error_with_the_close_code():
    service = _service()
    service._websocket = _ClosedWebSocket(4001, "Authentication required")
    service.push_error = AsyncMock()

    await service._receive_task_handler()

    error_msg = service.push_error.call_args.kwargs["error_msg"]
    assert "4001" in error_msg
    assert "Authentication required" in error_msg
