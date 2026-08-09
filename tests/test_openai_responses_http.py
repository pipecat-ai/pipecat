#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the HTTP variant of OpenAIResponsesHttpLLMService."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionToolCall,
    ResponseIncompleteEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseReasoningItem,
    ResponseReasoningSummaryTextDeltaEvent,
)
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)

from pipecat.frames.frames import (
    ErrorFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMServiceMetadataFrame,
    LLMThoughtEndFrame,
    LLMThoughtStartFrame,
    LLMThoughtTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.responses.llm import OpenAIResponsesHttpLLMService
from pipecat.tests.utils import run_test


def _make_service(**kwargs):
    """Create an HTTP service with the client and metrics hooks mocked out."""
    with patch.object(OpenAIResponsesHttpLLMService, "_create_client"):
        service = OpenAIResponsesHttpLLMService(api_key="test-key", **kwargs)

    service._client = AsyncMock()
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    service.start_llm_usage_metrics = AsyncMock()
    service._push_llm_text = AsyncMock()

    # Skip the adapter / param-building plumbing; we only exercise the
    # streaming completion handler here.
    adapter = MagicMock()
    adapter.get_messages_for_logging.return_value = []
    adapter.get_llm_invocation_params.return_value = {}
    service.get_llm_adapter = MagicMock(return_value=adapter)
    service._build_response_params = MagicMock(return_value={})

    return service


class _FakeAsyncStream:
    """Minimal stand-in for openai.AsyncStream that yields preset events."""

    def __init__(self, events):
        self._events = list(events)

    async def _iterator(self):
        for event in self._events:
            yield event

    def __aiter__(self):
        return self._iterator()

    async def close(self):
        pass


def _completed_event(usage):
    """Build a ResponseCompletedEvent carrying the given usage object."""
    response = MagicMock()
    response.usage = usage
    response.model = "gpt-4.1"
    event = MagicMock(spec=ResponseCompletedEvent)
    event.response = response
    return event


async def _run(service, *events):
    """Drive _process_context over a fake stream of the given events."""
    service._client.responses.create = AsyncMock(return_value=_FakeAsyncStream(events))
    await service._process_context(MagicMock(spec=LLMContext))


# ---------------------------------------------------------------------------
# _process_context — token usage parsing
# ---------------------------------------------------------------------------


class TestHttpTokenUsageMetrics:
    @pytest.mark.asyncio
    async def test_token_usage_with_details(self):
        """Native OpenAI responses (detail objects present) pass through unchanged."""
        service = _make_service()

        usage = ResponseUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            input_tokens_details=InputTokensDetails(cached_tokens=20),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=10),
        )
        await _run(service, _completed_event(usage))

        tokens = service.start_llm_usage_metrics.call_args[0][0]
        assert tokens.prompt_tokens == 100
        assert tokens.completion_tokens == 50
        assert tokens.total_tokens == 150
        assert tokens.cache_read_input_tokens == 20
        assert tokens.reasoning_tokens == 10

    @pytest.mark.asyncio
    async def test_token_usage_with_missing_details(self):
        """A third-party server may omit input/output token detail sub-objects.

        The OpenAI SDK leaves them as None. The handler must not raise and must
        fall back to 0 for cached/reasoning tokens. Regression test for the
        'NoneType' object has no attribute 'cached_tokens' crash.
        """
        service = _make_service()

        # construct() mirrors the SDK's lenient parse of a usage object that
        # lacks the detail sub-objects: the fields end up as None.
        usage = ResponseUsage.construct(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            input_tokens_details=None,
            output_tokens_details=None,
        )
        await _run(service, _completed_event(usage))

        assert service.start_llm_usage_metrics.called
        tokens = service.start_llm_usage_metrics.call_args[0][0]
        assert tokens.prompt_tokens == 10
        assert tokens.completion_tokens == 5
        assert tokens.total_tokens == 15
        assert tokens.cache_read_input_tokens == 0
        assert tokens.reasoning_tokens == 0

    @pytest.mark.asyncio
    async def test_token_usage_with_empty_details(self):
        """A third-party server may send empty detail sub-objects.

        The detail object is present but its fields (cached_tokens/
        reasoning_tokens) come back as None under the SDK's lenient parse. The
        handler must coalesce those to 0 rather than leak None into metrics.
        """
        service = _make_service()

        usage = ResponseUsage.construct(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            input_tokens_details=InputTokensDetails.construct(),
            output_tokens_details=OutputTokensDetails.construct(),
        )
        await _run(service, _completed_event(usage))

        assert service.start_llm_usage_metrics.called
        tokens = service.start_llm_usage_metrics.call_args[0][0]
        assert tokens.cache_read_input_tokens == 0
        assert tokens.reasoning_tokens == 0

    @pytest.mark.asyncio
    async def test_token_usage_with_missing_top_level_counts(self):
        """A third-party server may omit the top-level token counts.

        The SDK's lenient parse leaves omitted required counts as None. The
        handler must coalesce them to 0 so metrics never receive None.
        """
        service = _make_service()

        usage = ResponseUsage.construct(input_tokens=10)
        await _run(service, _completed_event(usage))

        assert service.start_llm_usage_metrics.called
        tokens = service.start_llm_usage_metrics.call_args[0][0]
        assert tokens.prompt_tokens == 10
        assert tokens.completion_tokens == 0
        assert tokens.total_tokens == 0
        assert tokens.cache_read_input_tokens == 0
        assert tokens.reasoning_tokens == 0


# ---------------------------------------------------------------------------
# _process_context — reasoning capture
# ---------------------------------------------------------------------------


class TestHttpReasoningCapture:
    @pytest.mark.asyncio
    async def test_summary_streamed_and_reasoning_item_persisted(self):
        service = _make_service()
        service.push_frame = AsyncMock()
        adapter = service.get_llm_adapter()
        adapter.create_llm_specific_message.side_effect = lambda m: m

        delta1 = MagicMock(spec=ResponseReasoningSummaryTextDeltaEvent)
        delta1.delta = "Think"
        delta2 = MagicMock(spec=ResponseReasoningSummaryTextDeltaEvent)
        delta2.delta = "ing..."

        summary_part = MagicMock()
        summary_part.text = "Thinking..."
        item = MagicMock(spec=ResponseReasoningItem)
        item.id = "rs_1"
        item.summary = [summary_part]
        item.encrypted_content = "ENCRYPTED"
        done = MagicMock(spec=ResponseOutputItemDoneEvent)
        done.item = item

        await _run(service, delta1, delta2, done)

        pushed = [c.args[0] for c in service.push_frame.call_args_list]
        assert sum(isinstance(f, LLMThoughtStartFrame) for f in pushed) == 1
        assert [f.text for f in pushed if isinstance(f, LLMThoughtTextFrame)] == ["Think", "ing..."]
        assert sum(isinstance(f, LLMThoughtEndFrame) for f in pushed) == 1

        append_frames = [f for f in pushed if isinstance(f, LLMMessagesAppendFrame)]
        assert len(append_frames) == 1
        stored = adapter.create_llm_specific_message.call_args[0][0]
        assert stored == {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "Thinking..."}],
            "encrypted_content": "ENCRYPTED",
        }


# ---------------------------------------------------------------------------
# _process_context — terminal error stream events (regression for #5138)
# ---------------------------------------------------------------------------


def _failed_event(event_cls=ResponseFailedEvent, event_type="response.failed", **response_kwargs):
    """Build a Response{Failed,Incomplete}Event carrying the given response fields."""
    response = MagicMock(**response_kwargs)
    event = MagicMock(spec=event_cls)
    event.type = event_type
    event.response = response
    return event


def _error_event(message="rate limited", code="rate_limit_exceeded"):
    event = MagicMock(spec=ResponseErrorEvent)
    event.type = "error"
    event.message = message
    event.code = code
    return event


class TestHttpStreamErrorEvents:
    @pytest.mark.asyncio
    async def test_response_failed_with_top_level_error_pushes_error(self):
        """A ResponseFailedEvent with `response.error` populated must push an error."""
        service = _make_service()
        service.push_error = AsyncMock()

        error = MagicMock()
        error.message = "upstream provider error"
        event = _failed_event(error=error)

        await _run(service, event)

        service.push_error.assert_called_once()
        assert "upstream provider error" in service.push_error.call_args.kwargs["error_msg"]

    @pytest.mark.asyncio
    async def test_response_failed_with_status_details_pushes_error(self):
        """Some third-party servers populate the older `status_details` shape
        instead of the typed `error` field. That must still surface an error."""
        service = _make_service()
        service.push_error = AsyncMock()

        event = _failed_event(
            error=None,
            status_details={"error": {"message": "provider outage"}},
        )

        await _run(service, event)

        service.push_error.assert_called_once()
        assert "provider outage" in service.push_error.call_args.kwargs["error_msg"]

    @pytest.mark.asyncio
    async def test_response_failed_with_no_error_details_uses_fallback_message(self):
        """When neither `error` nor `status_details` carry a message, fall back
        to a generic message derived from the event type rather than raising."""
        service = _make_service()
        service.push_error = AsyncMock()

        event = _failed_event(error=None, status_details=None)

        await _run(service, event)

        service.push_error.assert_called_once()
        assert "failed" in service.push_error.call_args.kwargs["error_msg"]

    @pytest.mark.asyncio
    async def test_response_incomplete_pushes_error(self):
        """A ResponseIncompleteEvent mid-stream must also push an error."""
        service = _make_service()
        service.push_error = AsyncMock()

        error = MagicMock()
        error.message = "max output tokens reached"
        event = _failed_event(
            event_cls=ResponseIncompleteEvent,
            event_type="response.incomplete",
            error=error,
        )

        await _run(service, event)

        service.push_error.assert_called_once()
        assert "max output tokens reached" in service.push_error.call_args.kwargs["error_msg"]

    @pytest.mark.asyncio
    async def test_response_error_event_pushes_error(self):
        """The generic in-stream `error` event must push an error with its message."""
        service = _make_service()
        service.push_error = AsyncMock()

        await _run(service, _error_event(message="rate limited"))

        service.push_error.assert_called_once()
        assert "rate limited" in service.push_error.call_args.kwargs["error_msg"]

    @pytest.mark.asyncio
    async def test_terminal_error_does_not_run_announced_function_call(self):
        """A function call whose arguments never finished streaming before a
        terminal error must not be executed with fabricated empty arguments."""
        service = _make_service()
        service.push_error = AsyncMock()
        service.run_function_calls = AsyncMock()

        item = MagicMock(spec=ResponseFunctionToolCall)
        item.id = "item_1"
        item.name = "get_weather"
        item.call_id = "call_1"
        added = MagicMock(spec=ResponseOutputItemAddedEvent)
        added.item = item

        delta = MagicMock(spec=ResponseFunctionCallArgumentsDeltaEvent)
        delta.item_id = "item_1"
        delta.delta = '{"city": "SF"'

        error = MagicMock()
        error.message = "upstream provider error"

        await _run(service, added, delta, _failed_event(error=error))

        service.push_error.assert_called_once()
        service.run_function_calls.assert_not_called()

    @pytest.mark.asyncio
    async def test_response_failed_reaches_pipeline_as_error_frame(self):
        """Full pipeline-level check: an in-stream response.failed event must
        surface as an ErrorFrame, not a silent, empty turn."""
        service = _make_service()

        error = MagicMock()
        error.message = "upstream provider error"
        service._client.responses.create = AsyncMock(
            return_value=_FakeAsyncStream([_failed_event(error=error)])
        )

        context = LLMContext()

        down_frames, up_frames = await run_test(
            service,
            frames_to_send=[LLMContextFrame(context=context)],
            expected_down_frames=[
                LLMServiceMetadataFrame,
                LLMFullResponseStartFrame,
                LLMFullResponseEndFrame,
            ],
        )

        error_frames = [f for f in list(down_frames) + list(up_frames) if isinstance(f, ErrorFrame)]
        assert error_frames, "Expected an ErrorFrame after an in-stream response.failed event"
