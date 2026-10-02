#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for text-injected user turns in AzureVoiceLiveLLMService.

These tests drive the context handler directly with a fake
``send_client_event`` and assert on the client events emitted to the service.
An audio turn reaches Voice Live as audio and gets a response for it; a turn
that only appears in the context has to be sent.
"""

from typing import Any

import pytest

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.azure.voicelive import events
from pipecat.services.azure.voicelive.llm import AzureVoiceLiveLLMService


class _EventRecorder:
    """Records the client events sent via ``send_client_event``."""

    def __init__(self):
        self.events: list[Any] = []

    async def __call__(self, event):
        self.events.append(event)

    def kinds(self) -> list[str]:
        return [type(e).__name__ for e in self.events]

    def user_texts(self) -> list[str]:
        texts = []
        for e in self.events:
            if isinstance(e, events.ConversationItemCreateEvent) and e.item.role == "user":
                texts.extend(c.text for c in (e.item.content or []) if c.text)
        return texts


async def _ignore_frame(*args, **kwargs) -> None:
    """Stand-in for ``push_frame``, which needs a linked processor."""


def _make_service(
    session_properties: events.SessionProperties | None = None,
) -> tuple[AzureVoiceLiveLLMService, _EventRecorder]:
    """Construct a service wired to a fake send_client_event."""
    settings = (
        AzureVoiceLiveLLMService.Settings(session_properties=session_properties)
        if session_properties
        else None
    )
    service = AzureVoiceLiveLLMService(
        api_key="test-key",
        endpoint="https://my-resource.services.ai.azure.com",
        settings=settings,
    )
    recorder = _EventRecorder()
    service.send_client_event = recorder
    service.push_frame = _ignore_frame
    # The session is configured by the time context frames arrive.
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False
    return service, recorder


async def _start(service: AzureVoiceLiveLLMService, context: LLMContext) -> None:
    """Hand the service its first context and let the response it asks for finish."""
    await service._handle_context(context)
    await service._handle_evt_response_done(
        events.ResponseDone(type="response.done", response={"id": "resp_0", "status": "completed"})
    )


@pytest.mark.asyncio
async def test_a_text_user_turn_is_sent_and_answered():
    """Nothing else carries the turn to the service when there is no audio."""
    service, recorder = _make_service()
    context = LLMContext([{"role": "developer", "content": "Be brief."}])

    await _start(service, context)
    recorder.events.clear()

    context.add_message({"role": "user", "content": "What is the capital of France?"})
    await service._handle_context(context)

    assert recorder.user_texts() == ["What is the capital of France?"]
    assert "ResponseCreateEvent" in recorder.kinds()


@pytest.mark.asyncio
async def test_a_text_turn_during_a_response_is_answered_when_it_finishes():
    """Voice Live rejects a second response while one is running."""
    service, recorder = _make_service()
    context = LLMContext([{"role": "developer", "content": "Be brief."}])

    await _start(service, context)
    service._response_in_flight = True
    recorder.events.clear()

    context.add_message({"role": "user", "content": "What is the capital of Japan?"})
    await service._handle_context(context)

    assert recorder.user_texts() == ["What is the capital of Japan?"]
    assert "ResponseCreateEvent" not in recorder.kinds()
    assert service._run_llm_when_response_done is True


@pytest.mark.asyncio
async def test_a_server_vad_turn_is_not_sent_again():
    """Server VAD already has the audio and created its own response."""
    service, recorder = _make_service()
    context = LLMContext([{"role": "developer", "content": "Be brief."}])

    await _start(service, context)
    recorder.events.clear()

    # The service pushes the transcript; it lands in the context after.
    await _transcription_completed(service, "What is the capital of France?")
    context.add_message({"role": "user", "content": "What is the capital of France?"})
    await service._handle_context(context)

    assert recorder.user_texts() == []
    assert "ResponseCreateEvent" not in recorder.kinds()


@pytest.mark.asyncio
async def test_a_tool_result_landing_first_does_not_release_the_server_vad_turn():
    """The transcript is written after the response starts, so a tool result can land first."""
    service, recorder = _make_service()
    context = LLMContext([{"role": "developer", "content": "Be brief."}])

    await _start(service, context)
    await _transcription_completed(service, "What's the weather?")

    # The spoken turn's tool call completes before its transcript is written.
    context.add_message(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_current_weather", "arguments": "{}"},
                }
            ],
        }
    )
    context.add_message({"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 75}'})
    await service._handle_context(context)

    recorder.events.clear()
    context.add_message({"role": "user", "content": "What's the weather?"})
    await service._handle_context(context)

    assert recorder.user_texts() == []
    assert "ResponseCreateEvent" not in recorder.kinds()

    context.add_message({"role": "user", "content": "typed turn"})
    await service._handle_context(context)

    assert recorder.user_texts() == ["typed turn"]


@pytest.mark.asyncio
async def test_a_manual_turn_is_not_sent_again_when_its_transcript_lands():
    """A manual turn commits the caller's audio, so Voice Live already has the turn."""
    service, recorder = _make_service(
        events.SessionProperties(
            turn_detection=None,
            input_audio_transcription=events.InputAudioTranscription(model="azure-speech"),
        )
    )
    context = LLMContext([{"role": "developer", "content": "Be brief."}])
    await _start(service, context)

    await service._handle_user_stopped_speaking(None)
    await _transcription_completed(service, "What is the capital of France?")
    recorder.events.clear()

    context.add_message({"role": "user", "content": "What is the capital of France?"})
    await service._handle_context(context)

    assert recorder.user_texts() == []
    assert "ResponseCreateEvent" not in recorder.kinds()


@pytest.mark.asyncio
async def test_an_assistant_message_does_not_create_a_turn():
    service, recorder = _make_service()
    context = LLMContext([{"role": "developer", "content": "Be brief."}])

    await _start(service, context)
    recorder.events.clear()

    context.add_message({"role": "assistant", "content": "Paris."})
    await service._handle_context(context)

    assert recorder.user_texts() == []
    assert "ResponseCreateEvent" not in recorder.kinds()


async def _transcription_completed(service, transcript):
    await service._handle_evt_input_audio_transcription_completed(
        type("Evt", (), {"item_id": "item_1", "transcript": transcript})()
    )


@pytest.mark.asyncio
async def test_an_empty_repeat_transcription_does_not_release_the_claim():
    """Voice Live reports an empty transcript for an item again once the next audio commits."""
    service, recorder = _make_service()
    context = LLMContext([{"role": "developer", "content": "Be brief."}])
    await _start(service, context)

    await _transcription_completed(service, "What is the capital of France?")
    await _transcription_completed(service, "")
    recorder.events.clear()

    context.add_message({"role": "user", "content": "What is the capital of France?"})
    await service._handle_context(context)

    assert recorder.user_texts() == []
    assert "ResponseCreateEvent" not in recorder.kinds()


@pytest.mark.asyncio
async def test_list_content_is_flattened_to_text():
    service, recorder = _make_service()
    context = LLMContext([{"role": "developer", "content": "Be brief."}])

    await _start(service, context)
    recorder.events.clear()

    context.add_message(
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is"}, {"type": "text", "text": "the time?"}],
        }
    )
    await service._handle_context(context)

    assert recorder.user_texts() == ["What is the time?"]
