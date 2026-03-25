#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the WebSocket variant of OpenAIResponsesLLMService."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.responses.llm import OpenAIResponsesLLMService


def _make_service(**kwargs):
    """Create a service with the client mocked out."""
    with patch.object(OpenAIResponsesLLMService, "_create_client"):
        service = OpenAIResponsesLLMService(
            api_key="test-key",
            **kwargs,
        )
        service._client = AsyncMock()
        return service


def _ws_events(*events):
    """Build a mock WebSocket that yields the given events from recv()."""
    ws = AsyncMock()
    # .recv() returns each event in order, then raises StopAsyncIteration
    ws.recv = AsyncMock(side_effect=[json.dumps(e) for e in events])
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    ws.close_code = None
    return ws


# ---------------------------------------------------------------------------
# Hash determinism
# ---------------------------------------------------------------------------


class TestHashInputItems:
    def test_same_input_same_hash(self):
        items = [{"role": "user", "content": "hello"}]
        h1 = OpenAIResponsesLLMService._hash_input_items(items)
        h2 = OpenAIResponsesLLMService._hash_input_items(items)
        assert h1 == h2

    def test_different_input_different_hash(self):
        h1 = OpenAIResponsesLLMService._hash_input_items([{"role": "user", "content": "hello"}])
        h2 = OpenAIResponsesLLMService._hash_input_items([{"role": "user", "content": "world"}])
        assert h1 != h2

    def test_order_independent_keys(self):
        """Keys within a dict should not affect hash (sort_keys=True)."""
        h1 = OpenAIResponsesLLMService._hash_input_items([{"a": 1, "b": 2}])
        h2 = OpenAIResponsesLLMService._hash_input_items([{"b": 2, "a": 1}])
        assert h1 == h2


# ---------------------------------------------------------------------------
# previous_response_id optimization
# ---------------------------------------------------------------------------


class TestPreviousResponseOptimization:
    def test_no_previous_state_sends_full_input(self):
        service = _make_service()
        full_input = [{"role": "user", "content": "hi"}]
        params = {"input": full_input, "model": "gpt-4.1"}

        result = service._apply_previous_response_optimization(params, full_input)

        assert result["input"] == full_input
        assert "previous_response_id" not in result

    def test_matching_prefix_sends_incremental(self):
        service = _make_service()
        prev_input = [{"role": "user", "content": "hi"}]
        service._store_previous_response_state("resp_123", prev_input)

        full_input = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "how are you?"},
        ]
        params = {"input": list(full_input), "model": "gpt-4.1"}

        result = service._apply_previous_response_optimization(params, full_input)

        assert result["previous_response_id"] == "resp_123"
        assert result["input"] == full_input[1:]

    def test_mismatched_prefix_sends_full(self):
        service = _make_service()
        prev_input = [{"role": "user", "content": "hi"}]
        service._store_previous_response_state("resp_123", prev_input)

        # Different first message
        full_input = [
            {"role": "user", "content": "different"},
            {"role": "assistant", "content": "hello"},
        ]
        params = {"input": list(full_input), "model": "gpt-4.1"}

        result = service._apply_previous_response_optimization(params, full_input)

        assert "previous_response_id" not in result
        assert result["input"] == full_input

    def test_same_length_sends_full(self):
        """When new input is same length as previous, no optimization."""
        service = _make_service()
        prev_input = [{"role": "user", "content": "hi"}]
        service._store_previous_response_state("resp_123", prev_input)

        full_input = [{"role": "user", "content": "hi"}]
        params = {"input": list(full_input), "model": "gpt-4.1"}

        result = service._apply_previous_response_optimization(params, full_input)

        assert "previous_response_id" not in result

    def test_clear_state(self):
        service = _make_service()
        service._store_previous_response_state("resp_123", [{"role": "user", "content": "hi"}])
        service._clear_previous_response_state()

        assert service._previous_response_id is None
        assert service._previous_input_hash is None
        assert service._previous_input_length is None


# ---------------------------------------------------------------------------
# _receive_response_events — text streaming
# ---------------------------------------------------------------------------


class TestReceiveResponseEventsText:
    @pytest.mark.asyncio
    async def test_text_deltas_pushed(self):
        service = _make_service()
        service._push_llm_text = AsyncMock()
        service.stop_ttfb_metrics = AsyncMock()
        service.start_llm_usage_metrics = AsyncMock()

        ws = _ws_events(
            {"type": "response.output_text.delta", "delta": "Hello"},
            {"type": "response.output_text.delta", "delta": " world"},
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "model": "gpt-4.1",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens_details": {"reasoning_tokens": 0},
                    },
                },
            },
        )
        service._websocket = ws

        context = MagicMock(spec=LLMContext)
        full_input = [{"role": "user", "content": "hi"}]
        await service._receive_response_events(context, full_input)

        assert service._push_llm_text.call_count == 2
        service._push_llm_text.assert_any_await("Hello")
        service._push_llm_text.assert_any_await(" world")

    @pytest.mark.asyncio
    async def test_response_completed_stores_state(self):
        service = _make_service()
        service._push_llm_text = AsyncMock()
        service.stop_ttfb_metrics = AsyncMock()
        service.start_llm_usage_metrics = AsyncMock()

        ws = _ws_events(
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_42",
                    "model": "gpt-4.1",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                        "input_tokens_details": {"cached_tokens": 2},
                        "output_tokens_details": {"reasoning_tokens": 1},
                    },
                },
            },
        )
        service._websocket = ws

        context = MagicMock(spec=LLMContext)
        full_input = [{"role": "user", "content": "hi"}]
        await service._receive_response_events(context, full_input)

        assert service._previous_response_id == "resp_42"
        assert service._previous_input_length == 1
        assert service._previous_input_hash is not None
        assert service.start_llm_usage_metrics.called

    @pytest.mark.asyncio
    async def test_token_usage_metrics(self):
        service = _make_service()
        service._push_llm_text = AsyncMock()
        service.stop_ttfb_metrics = AsyncMock()
        service.start_llm_usage_metrics = AsyncMock()

        ws = _ws_events(
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "model": "gpt-4.1",
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "total_tokens": 150,
                        "input_tokens_details": {"cached_tokens": 20},
                        "output_tokens_details": {"reasoning_tokens": 10},
                    },
                },
            },
        )
        service._websocket = ws

        context = MagicMock(spec=LLMContext)
        await service._receive_response_events(context, [])

        tokens = service.start_llm_usage_metrics.call_args[0][0]
        assert tokens.prompt_tokens == 100
        assert tokens.completion_tokens == 50
        assert tokens.total_tokens == 150
        assert tokens.cache_read_input_tokens == 20
        assert tokens.reasoning_tokens == 10


# ---------------------------------------------------------------------------
# _receive_response_events — function calls
# ---------------------------------------------------------------------------


class TestReceiveResponseEventsFunctionCalls:
    @pytest.mark.asyncio
    async def test_function_call_sequence(self):
        service = _make_service()
        service._push_llm_text = AsyncMock()
        service.stop_ttfb_metrics = AsyncMock()
        service.start_llm_usage_metrics = AsyncMock()
        service.run_function_calls = AsyncMock()

        ws = _ws_events(
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "name": "get_weather",
                    "call_id": "call_1",
                },
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "delta": '{"loc',
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "delta": 'ation": "SF"}',
            },
            {
                "type": "response.function_call_arguments.done",
                "item_id": "fc_1",
                "arguments": '{"location": "SF"}',
            },
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "name": "get_weather",
                    "call_id": "call_1",
                    "arguments": '{"location": "SF"}',
                },
            },
            {
                "type": "response.completed",
                "response": {"id": "resp_1", "model": "gpt-4.1", "usage": None},
            },
        )
        service._websocket = ws

        context = MagicMock(spec=LLMContext)
        await service._receive_response_events(context, [])

        service.run_function_calls.assert_called_once()
        fc_list = service.run_function_calls.call_args[0][0]
        assert len(fc_list) == 1
        assert fc_list[0].function_name == "get_weather"
        assert fc_list[0].tool_call_id == "call_1"
        assert fc_list[0].arguments == {"location": "SF"}


# ---------------------------------------------------------------------------
# _receive_response_events — errors
# ---------------------------------------------------------------------------


class TestReceiveResponseEventsErrors:
    @pytest.mark.asyncio
    async def test_response_failed_pushes_error(self):
        service = _make_service()
        service.stop_ttfb_metrics = AsyncMock()
        service.start_llm_usage_metrics = AsyncMock()
        service.push_error = AsyncMock()

        ws = _ws_events(
            {
                "type": "response.failed",
                "response": {
                    "id": "resp_1",
                    "status_details": {
                        "error": {"message": "Content filter triggered"},
                    },
                },
            },
        )
        service._websocket = ws

        context = MagicMock(spec=LLMContext)
        await service._receive_response_events(context, [])

        service.push_error.assert_called_once()
        assert "Content filter triggered" in service.push_error.call_args.kwargs["error_msg"]

    @pytest.mark.asyncio
    async def test_response_incomplete_pushes_error(self):
        service = _make_service()
        service.stop_ttfb_metrics = AsyncMock()
        service.start_llm_usage_metrics = AsyncMock()
        service.push_error = AsyncMock()

        ws = _ws_events(
            {
                "type": "response.incomplete",
                "response": {"id": "resp_1", "status_details": None},
            },
        )
        service._websocket = ws

        context = MagicMock(spec=LLMContext)
        await service._receive_response_events(context, [])

        service.push_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_previous_response_not_found_raises(self):
        from pipecat.services.openai.responses.llm import _PreviousResponseNotFoundError

        service = _make_service()
        service.stop_ttfb_metrics = AsyncMock()

        ws = _ws_events(
            {
                "type": "error",
                "error": {
                    "code": "previous_response_not_found",
                    "message": "Previous response with id 'resp_abc' not found.",
                },
            },
        )
        service._websocket = ws

        context = MagicMock(spec=LLMContext)
        with pytest.raises(_PreviousResponseNotFoundError):
            await service._receive_response_events(context, [])

    @pytest.mark.asyncio
    async def test_connection_limit_reached_raises(self):
        from pipecat.services.openai.responses.llm import _ConnectionLimitReachedError

        service = _make_service()
        service.stop_ttfb_metrics = AsyncMock()

        ws = _ws_events(
            {
                "type": "error",
                "error": {
                    "code": "websocket_connection_limit_reached",
                    "message": "Connection limit reached.",
                },
            },
        )
        service._websocket = ws

        context = MagicMock(spec=LLMContext)
        with pytest.raises(_ConnectionLimitReachedError):
            await service._receive_response_events(context, [])

    @pytest.mark.asyncio
    async def test_generic_error_pushes_error(self):
        service = _make_service()
        service.stop_ttfb_metrics = AsyncMock()
        service.start_llm_usage_metrics = AsyncMock()
        service.push_error = AsyncMock()

        ws = _ws_events(
            {
                "type": "error",
                "error": {
                    "code": "server_error",
                    "message": "Internal server error",
                },
            },
        )
        service._websocket = ws

        context = MagicMock(spec=LLMContext)
        await service._receive_response_events(context, [])

        service.push_error.assert_called_once()
        assert "Internal server error" in service.push_error.call_args.kwargs["error_msg"]


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------


class TestConnectionLifecycle:
    @pytest.mark.asyncio
    async def test_disconnect_clears_previous_response_state(self):
        service = _make_service()
        service._store_previous_response_state("resp_1", [{"role": "user", "content": "hi"}])
        service.stop_all_metrics = AsyncMock()

        await service._disconnect()

        assert service._previous_response_id is None
        assert service._previous_input_hash is None
        assert service._previous_input_length is None

    @pytest.mark.asyncio
    async def test_reconnect_clears_state_and_reconnects(self):
        service = _make_service()
        service._store_previous_response_state("resp_1", [{"role": "user", "content": "hi"}])
        service.stop_all_metrics = AsyncMock()
        service.push_error = AsyncMock()

        # Mock connect to set a websocket
        mock_ws = AsyncMock()
        mock_ws.close = AsyncMock()
        service._websocket = mock_ws

        with patch(
            "pipecat.services.openai.responses.llm.websocket_connect",
            new_callable=AsyncMock,
            return_value=AsyncMock(),
        ):
            await service._reconnect()

        assert service._previous_response_id is None
        mock_ws.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_connected_raises_on_failure(self):
        from pipecat.services.openai.responses.llm import _RetryableError

        service = _make_service()
        service._websocket = None
        service.push_error = AsyncMock()

        # Mock connect to fail
        with patch(
            "pipecat.services.openai.responses.llm.websocket_connect",
            new_callable=AsyncMock,
            side_effect=Exception("Connection refused"),
        ):
            with pytest.raises(_RetryableError):
                await service._ensure_connected()
