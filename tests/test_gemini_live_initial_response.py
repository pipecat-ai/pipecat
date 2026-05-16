from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService


class _TestGeminiLiveLLMService(GeminiLiveLLMService):
    def create_client(self):
        self._client = SimpleNamespace(aio=SimpleNamespace(live=SimpleNamespace(connect=None)))


class _FakeSession:
    def __init__(self):
        self.send_client_content = AsyncMock()
        self.send_realtime_input = AsyncMock()
        self.close = AsyncMock()


def _make_service(
    *, system_instruction: str | None = None, inference_on_context_initialization: bool = True
) -> _TestGeminiLiveLLMService:
    service = _TestGeminiLiveLLMService(
        api_key="test-key",
        system_instruction=system_instruction,
        inference_on_context_initialization=inference_on_context_initialization,
    )
    service.start_ttfb_metrics = AsyncMock()
    service.stop_all_metrics = AsyncMock()
    service.cancel_task = AsyncMock()
    service.push_error = AsyncMock()
    return service


@pytest.mark.asyncio
async def test_non_gemini_3_initial_response_uses_blank_user_turn_when_only_system_instruction():
    service = _make_service(system_instruction="Start by greeting the caller.")
    service._session = _FakeSession()
    service._context = LLMContext(messages=[])

    await service._create_initial_response()

    service._session.send_client_content.assert_awaited_once()
    kwargs = service._session.send_client_content.await_args.kwargs
    assert kwargs["turn_complete"] is True
    assert len(kwargs["turns"]) == 1
    assert kwargs["turns"][0].role == "user"
    assert kwargs["turns"][0].parts[0].text == " "
    service._session.send_realtime_input.assert_not_awaited()
    assert service._ready_for_realtime_input is True


@pytest.mark.asyncio
async def test_non_gemini_3_empty_seed_creates_blank_user_turn_and_arms_workaround():
    service = _make_service(
        system_instruction="Wait for the caller before responding.",
        inference_on_context_initialization=False,
    )
    service._session = _FakeSession()
    service._context = LLMContext(messages=[])

    await service._create_initial_response()

    service._session.send_client_content.assert_awaited_once()
    kwargs = service._session.send_client_content.await_args.kwargs
    assert kwargs["turn_complete"] is False
    assert len(kwargs["turns"]) == 1
    assert kwargs["turns"][0].role == "user"
    assert kwargs["turns"][0].parts[0].text == " "
    assert service._needs_initial_turn_complete_message is True
    assert service._ready_for_realtime_input is True
