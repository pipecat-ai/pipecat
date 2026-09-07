#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the HTTP variant of OpenAIResponsesHttpLLMService."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import APITimeoutError
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseIncompleteEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseReasoningItem,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseTextDeltaEvent,
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


def _failed_event(error):
    """Build a ResponseFailedEvent whose response carries the given error object."""
    response = MagicMock()
    response.error = error
    event = MagicMock(spec=ResponseFailedEvent)
    event.response = response
    return event


def _incomplete_event(incomplete_details):
    """Build a ResponseIncompleteEvent carrying the given incomplete_details."""
    response = MagicMock()
    response.incomplete_details = incomplete_details
    event = MagicMock(spec=ResponseIncompleteEvent)
    event.response = response
    return event


def _error_event(message):
    """Build a top-level ResponseErrorEvent with the given message."""
    event = MagicMock(spec=ResponseErrorEvent)
    event.message = message
    return event


def _text_delta_event(text):
    """Build a ResponseTextDeltaEvent carrying the given delta."""
    event = MagicMock(spec=ResponseTextDeltaEvent)
    event.delta = text
    return event


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
            input_tokens_details=InputTokensDetails(cached_tokens=20, cache_write_tokens=0),
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
# _process_context — in-stream terminal error events
# ---------------------------------------------------------------------------


class TestHttpStreamErrorEvents:
    """In-stream terminal events must surface an error, not an empty turn.

    Errors raised before streaming starts are handled by the caller's except
    blocks. These events arrive on an otherwise healthy 200 stream, so without
    an explicit branch the loop ends and the turn looks successful.
    """

    @pytest.mark.asyncio
    async def test_response_failed_pushes_error(self):
        service = _make_service()
        service.push_error = AsyncMock()

        error = MagicMock()
        error.message = "Content filter triggered"
        await _run(service, _failed_event(error))

        service.push_error.assert_called_once()
        assert "Content filter triggered" in service.push_error.call_args.kwargs["error_msg"]

    @pytest.mark.asyncio
    async def test_response_failed_without_error_details(self):
        """A third-party server may omit the error object on a failed response.

        The SDK's lenient parse leaves it as None; the handler must still push
        an error rather than raise on attribute access.
        """
        service = _make_service()
        service.push_error = AsyncMock()

        await _run(service, _failed_event(None))

        service.push_error.assert_called_once()
        assert "Response failed" in service.push_error.call_args.kwargs["error_msg"]

    @pytest.mark.asyncio
    async def test_response_failed_with_empty_error_message(self):
        """A server may send an error object whose message is empty.

        The generic fallback must still apply, so the pushed error is never
        just the bare prefix.
        """
        service = _make_service()
        service.push_error = AsyncMock()

        error = MagicMock()
        error.message = None
        await _run(service, _failed_event(error))

        service.push_error.assert_called_once()
        assert "Response failed" in service.push_error.call_args.kwargs["error_msg"]

    @pytest.mark.asyncio
    async def test_response_incomplete_pushes_error(self):
        service = _make_service()
        service.push_error = AsyncMock()

        details = MagicMock()
        details.reason = "max_output_tokens"
        await _run(service, _incomplete_event(details))

        service.push_error.assert_called_once()
        assert "max_output_tokens" in service.push_error.call_args.kwargs["error_msg"]

    @pytest.mark.asyncio
    async def test_response_incomplete_without_details(self):
        service = _make_service()
        service.push_error = AsyncMock()

        await _run(service, _incomplete_event(None))

        service.push_error.assert_called_once()
        assert "Response incomplete" in service.push_error.call_args.kwargs["error_msg"]

    @pytest.mark.asyncio
    async def test_error_event_pushes_error(self):
        service = _make_service()
        service.push_error = AsyncMock()

        await _run(service, _error_event("Internal server error"))

        service.push_error.assert_called_once()
        assert "Internal server error" in service.push_error.call_args.kwargs["error_msg"]

    @pytest.mark.asyncio
    async def test_real_sdk_events_are_read_from_the_right_fields(self):
        """Drive genuine SDK event objects rather than mocks.

        ``failed`` and ``incomplete`` carry their detail on different fields,
        and a mock would satisfy any attribute path, so build the events the
        way the streaming decoder does and assert the text arrives.
        """
        failed = ResponseFailedEvent.construct(
            type="response.failed",
            sequence_number=1,
            response={"error": {"code": "server_error", "message": "Content filter triggered"}},
        )
        incomplete = ResponseIncompleteEvent.construct(
            type="response.incomplete",
            sequence_number=1,
            response={"incomplete_details": {"reason": "max_output_tokens"}},
        )
        error = ResponseErrorEvent.construct(
            type="error", sequence_number=1, message="Internal server error", code=None, param=None
        )

        for event, expected in (
            (failed, "Content filter triggered"),
            (incomplete, "max_output_tokens"),
            (error, "Internal server error"),
        ):
            service = _make_service()
            service.push_error = AsyncMock()

            await _run(service, event)

            service.push_error.assert_called_once()
            assert expected in service.push_error.call_args.kwargs["error_msg"]

    @pytest.mark.asyncio
    async def test_terminal_error_stops_consuming_the_stream(self):
        """A terminal event ends the turn: later events must not be processed."""
        service = _make_service()
        service.push_error = AsyncMock()

        error = MagicMock()
        error.message = "Content filter triggered"
        await _run(service, _failed_event(error), _text_delta_event("should not be spoken"))

        service.push_error.assert_called_once()
        service._push_llm_text.assert_not_called()

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

        await _run(service, added, delta, _failed_event(error))

        service.push_error.assert_called_once()
        service.run_function_calls.assert_not_called()

    @pytest.mark.asyncio
    async def test_terminal_error_still_runs_completed_function_calls(self):
        """Parallel tool calls: a call whose arguments finished streaming before
        the terminal error (e.g. a later item truncated by max_output_tokens)
        must still run — only the unfinished call is dropped."""
        service = _make_service()
        service.push_error = AsyncMock()
        service.run_function_calls = AsyncMock()

        def _tool_call_added(item_id, name, call_id):
            item = MagicMock(spec=ResponseFunctionToolCall)
            item.id = item_id
            item.name = name
            item.call_id = call_id
            event = MagicMock(spec=ResponseOutputItemAddedEvent)
            event.item = item
            return event

        args_done = MagicMock(spec=ResponseFunctionCallArgumentsDoneEvent)
        args_done.item_id = "item_1"
        args_done.arguments = '{"city": "SF"}'

        done_item = MagicMock(spec=ResponseFunctionToolCall)
        done_item.id = "item_1"
        done_item.name = "get_weather"
        done_item.call_id = "call_1"
        done_item.arguments = '{"city": "SF"}'
        item_done = MagicMock(spec=ResponseOutputItemDoneEvent)
        item_done.item = done_item

        partial_delta = MagicMock(spec=ResponseFunctionCallArgumentsDeltaEvent)
        partial_delta.item_id = "item_2"
        partial_delta.delta = '{"city": "NY'

        details = MagicMock()
        details.reason = "max_output_tokens"

        await _run(
            service,
            _tool_call_added("item_1", "get_weather", "call_1"),
            args_done,
            item_done,
            _tool_call_added("item_2", "get_time", "call_2"),
            partial_delta,
            _incomplete_event(details),
        )

        service.push_error.assert_called_once()
        service.run_function_calls.assert_called_once()
        fc_list = service.run_function_calls.call_args.args[0]
        assert [fc.function_name for fc in fc_list] == ["get_weather"]
        assert fc_list[0].arguments == {"city": "SF"}

    @pytest.mark.asyncio
    async def test_response_failed_reaches_pipeline_as_error_frame(self):
        """Pipeline-level check: an in-stream response.failed event must surface
        as an ErrorFrame, which is what failover strategies react to."""
        service = _make_service()

        error = MagicMock()
        error.message = "upstream provider error"
        service._client.responses.create = AsyncMock(
            return_value=_FakeAsyncStream([_failed_event(error)])
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


# ---------------------------------------------------------------------------
# retry_on_timeout
# ---------------------------------------------------------------------------


class TestHttpRetryOnTimeout:
    def test_disabled_by_default(self):
        service = _make_service()
        assert service._retry_on_timeout is False
        assert service._retry_timeout_secs == 5.0

    @pytest.mark.asyncio
    async def test_no_timeout_applied_when_disabled(self):
        service = _make_service()

        async def slow_create(**kwargs):
            await asyncio.sleep(0.1)
            return _FakeAsyncStream([])

        service._client.responses.create = AsyncMock(side_effect=slow_create)

        await service._create_stream({})

        assert service._client.responses.create.await_count == 1

    @pytest.mark.asyncio
    async def test_reissues_request_on_timeout(self):
        service = _make_service(retry_on_timeout=True, retry_timeout_secs=0.05)

        stream = _FakeAsyncStream([])
        attempts = []

        async def create(**kwargs):
            attempts.append(kwargs)
            if len(attempts) == 1:
                await asyncio.sleep(10)
            return stream

        service._client.responses.create = AsyncMock(side_effect=create)

        assert await service._create_stream({"model": "gpt-4.1"}) is stream
        assert len(attempts) == 2

    @pytest.mark.asyncio
    async def test_retry_is_unbounded(self):
        """The retry must outlive retry_timeout_secs, or it would be pointless."""
        service = _make_service(retry_on_timeout=True, retry_timeout_secs=0.05)

        stream = _FakeAsyncStream([])
        attempts = 0

        async def create(**kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                await asyncio.sleep(10)
            await asyncio.sleep(0.15)  # longer than retry_timeout_secs
            return stream

        service._client.responses.create = AsyncMock(side_effect=create)

        assert await service._create_stream({}) is stream
        assert attempts == 2

    @pytest.mark.asyncio
    async def test_api_timeout_error_also_retries(self):
        service = _make_service(retry_on_timeout=True, retry_timeout_secs=0.05)

        stream = _FakeAsyncStream([])
        attempts = 0

        async def create(**kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise APITimeoutError(request=httpx.Request("POST", "https://api.openai.com"))
            return stream

        service._client.responses.create = AsyncMock(side_effect=create)

        assert await service._create_stream({}) is stream
        assert attempts == 2
