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
    ResponseOutputItemAddedEvent,
    ResponseOutputMessage,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.responses.llm import OpenAIResponsesHttpLLMService


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


def _message_item_added(output_index):
    """Build a ResponseOutputItemAddedEvent for a message item at the index."""
    event = MagicMock(spec=ResponseOutputItemAddedEvent)
    event.output_index = output_index
    event.item = MagicMock(spec=ResponseOutputMessage)
    return event


def _text_delta(output_index, delta):
    """Build a ResponseTextDeltaEvent for the given output item and text."""
    event = MagicMock(spec=ResponseTextDeltaEvent)
    event.output_index = output_index
    event.delta = delta
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
# _process_context — duplicate message item handling
# ---------------------------------------------------------------------------


class TestHttpMessageDeduplication:
    """The Responses API can emit multiple message items with identical text.

    With ``deduplicate_output_messages`` enabled, only text from the first
    message item is forwarded; otherwise every delta is forwarded (default).
    """

    @pytest.mark.asyncio
    async def test_duplicate_messages_dropped_when_enabled(self):
        service = _make_service()
        service._settings.deduplicate_output_messages = True

        await _run(
            service,
            _message_item_added(0),
            _text_delta(0, "Hello"),
            _message_item_added(1),
            _text_delta(1, "Hello"),
        )

        service._push_llm_text.assert_awaited_once_with("Hello")

    @pytest.mark.asyncio
    async def test_duplicate_messages_kept_when_disabled(self):
        # Default behavior: forward every delta, including the duplicate.
        service = _make_service()

        await _run(
            service,
            _message_item_added(0),
            _text_delta(0, "Hello"),
            _message_item_added(1),
            _text_delta(1, "Hello"),
        )

        assert service._push_llm_text.await_count == 2

    @pytest.mark.asyncio
    async def test_single_message_streams_all_deltas_when_enabled(self):
        # A normal single-message response must not lose any text.
        service = _make_service()
        service._settings.deduplicate_output_messages = True

        await _run(
            service,
            _message_item_added(0),
            _text_delta(0, "Hello"),
            _text_delta(0, " world"),
        )

        assert service._push_llm_text.await_count == 2
        service._push_llm_text.assert_any_await("Hello")
        service._push_llm_text.assert_any_await(" world")
