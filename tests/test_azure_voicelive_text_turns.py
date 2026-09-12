#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for text-injected user turns in AzureVoiceLiveLLMService.

These tests drive the context handler directly with a fake
``send_client_event`` and assert on the client events emitted to the service.
An audio turn reaches the service as audio and server VAD creates its own
response for it; a turn that only appears in the context has to be sent.
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


def _make_service() -> tuple[AzureVoiceLiveLLMService, _EventRecorder]:
    """Construct a service wired to a fake send_client_event."""
    service = AzureVoiceLiveLLMService(
        api_key="test-key",
        endpoint="https://my-resource.services.ai.azure.com",
    )
    recorder = _EventRecorder()
    service.send_client_event = recorder
    # The session is configured by the time context frames arrive.
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False
    return service, recorder


@pytest.mark.asyncio
async def test_a_text_user_turn_is_sent_and_answered():
    """Nothing else carries the turn to the service when there is no audio."""
    service, recorder = _make_service()
    context = LLMContext([{"role": "developer", "content": "Be brief."}])

    await service._handle_context(context)
    recorder.events.clear()

    context.add_message({"role": "user", "content": "What is the capital of France?"})
    await service._handle_context(context)

    assert recorder.user_texts() == ["What is the capital of France?"]
    assert "ResponseCreateEvent" in recorder.kinds()


@pytest.mark.asyncio
async def test_a_server_vad_turn_is_not_sent_again():
    """Server VAD already has the audio and created its own response."""
    service, recorder = _make_service()
    context = LLMContext([{"role": "developer", "content": "Be brief."}])

    await service._handle_context(context)
    recorder.events.clear()

    # Server VAD closed the turn; the transcript lands in the context after.
    await service._handle_evt_speech_stopped(None)
    context.add_message({"role": "user", "content": "What is the capital of France?"})
    await service._handle_context(context)

    assert recorder.user_texts() == []
    assert "ResponseCreateEvent" not in recorder.kinds()


@pytest.mark.asyncio
async def test_an_assistant_message_does_not_create_a_turn():
    service, recorder = _make_service()
    context = LLMContext([{"role": "developer", "content": "Be brief."}])

    await service._handle_context(context)
    recorder.events.clear()

    context.add_message({"role": "assistant", "content": "Paris."})
    await service._handle_context(context)

    assert recorder.user_texts() == []
    assert "ResponseCreateEvent" not in recorder.kinds()


@pytest.mark.asyncio
async def test_a_text_turn_after_a_server_vad_turn_is_still_sent():
    """The skip applies to the turn server VAD handled, not to later ones."""
    service, recorder = _make_service()
    context = LLMContext([{"role": "developer", "content": "Be brief."}])

    await service._handle_context(context)

    await service._handle_evt_speech_stopped(None)
    context.add_message({"role": "user", "content": "spoken turn"})
    await service._handle_context(context)

    recorder.events.clear()
    context.add_message({"role": "user", "content": "typed turn"})
    await service._handle_context(context)

    assert recorder.user_texts() == ["typed turn"]


@pytest.mark.asyncio
async def test_list_content_is_flattened_to_text():
    service, recorder = _make_service()
    context = LLMContext([{"role": "developer", "content": "Be brief."}])

    await service._handle_context(context)
    recorder.events.clear()

    context.add_message(
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is"}, {"type": "text", "text": "the time?"}],
        }
    )
    await service._handle_context(context)

    assert recorder.user_texts() == ["What is the time?"]
