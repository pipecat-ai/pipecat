#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the WebSocket variant of OpenAIResponsesLLMService."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipecat.frames.frames import (
    LLMMessagesAppendFrame,
    LLMThoughtEndFrame,
    LLMThoughtStartFrame,
    LLMThoughtTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.responses.llm import (
    OpenAIResponsesLLMService,
    _model_supports_reasoning,
)


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


class TestStartsWithResponseOutput:
    def test_text_message_matches_by_role(self):
        response_output = [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello!"}],
            }
        ]
        # Adapter produces a different format, but same role
        items = [{"role": "assistant", "content": "Hello!"}, {"role": "user", "content": "hi"}]
        assert OpenAIResponsesLLMService._starts_with_response_output(items, response_output)

    def test_function_call_matches_by_call_id(self):
        response_output = [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"location": "SF"}',
            }
        ]
        # Adapter format (no "id" field)
        items = [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
        ]
        assert OpenAIResponsesLLMService._starts_with_response_output(items, response_output)

    def test_mixed_output(self):
        response_output = [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Let me check."}],
            },
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": "{}",
            },
        ]
        items = [
            {"role": "assistant", "content": "Let me check."},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
        ]
        assert OpenAIResponsesLLMService._starts_with_response_output(items, response_output)

    def test_role_mismatch(self):
        response_output = [{"type": "message", "role": "assistant", "content": []}]
        items = [{"role": "user", "content": "hi"}]
        assert not OpenAIResponsesLLMService._starts_with_response_output(items, response_output)

    def test_text_content_mismatch(self):
        response_output = [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello!"}],
            }
        ]
        items = [{"role": "assistant", "content": "Something completely different"}]
        assert not OpenAIResponsesLLMService._starts_with_response_output(items, response_output)

    def test_call_id_mismatch(self):
        response_output = [{"type": "function_call", "call_id": "call_1", "name": "f"}]
        items = [{"type": "function_call", "call_id": "call_999", "name": "f"}]
        assert not OpenAIResponsesLLMService._starts_with_response_output(items, response_output)

    def test_too_few_items(self):
        response_output = [
            {"type": "message", "role": "assistant", "content": []},
            {"type": "function_call", "call_id": "call_1", "name": "f"},
        ]
        items = [{"role": "assistant", "content": "hi"}]
        assert not OpenAIResponsesLLMService._starts_with_response_output(items, response_output)

    def test_empty_output_always_matches(self):
        assert OpenAIResponsesLLMService._starts_with_response_output([], [])
        assert OpenAIResponsesLLMService._starts_with_response_output([{"role": "user"}], [])

    def test_unknown_output_type_rejects(self):
        response_output = [{"type": "unknown_thing", "data": "something"}]
        items = [{"role": "assistant", "content": "hi"}]
        assert not OpenAIResponsesLLMService._starts_with_response_output(items, response_output)


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
        # Simulate: sent [user_msg], got assistant reply "hello"
        prev_input = [{"role": "user", "content": "hi"}]
        prev_output = [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            }
        ]
        service._store_previous_response_state("resp_123", prev_input, prev_output)

        # Next call: adapter produces full context including assistant reply + new user msg
        full_input = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "how are you?"},
        ]
        params = {"input": list(full_input), "model": "gpt-4.1"}

        result = service._apply_previous_response_optimization(params, full_input)

        assert result["previous_response_id"] == "resp_123"
        # Only the new user message should be sent
        assert result["input"] == [{"role": "user", "content": "how are you?"}]

    def test_mismatched_prefix_sends_full(self):
        service = _make_service()
        prev_input = [{"role": "user", "content": "hi"}]
        service._store_previous_response_state("resp_123", prev_input, [])

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
        service._store_previous_response_state("resp_123", prev_input, [])

        full_input = [{"role": "user", "content": "hi"}]
        params = {"input": list(full_input), "model": "gpt-4.1"}

        result = service._apply_previous_response_optimization(params, full_input)

        assert "previous_response_id" not in result

    def test_output_mismatch_sends_full_context(self):
        """When prefix matches but output doesn't, fall back to full context."""
        service = _make_service()
        prev_input = [{"role": "user", "content": "hi"}]
        prev_output = [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            }
        ]
        service._store_previous_response_state("resp_123", prev_input, prev_output)

        # Aggregator stored the output differently (e.g. different role)
        full_input = [
            {"role": "user", "content": "hi"},
            {"role": "developer", "content": "something unexpected"},
            {"role": "user", "content": "how are you?"},
        ]
        params = {"input": list(full_input), "model": "gpt-4.1"}

        result = service._apply_previous_response_optimization(params, full_input)

        assert "previous_response_id" not in result
        assert result["input"] == full_input

    def test_clear_state(self):
        service = _make_service()
        service._store_previous_response_state("resp_123", [{"role": "user", "content": "hi"}], [])
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
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "Hello!"}],
                        }
                    ],
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
        assert len(service._previous_response_output) == 1
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


class TestDrainCancelledResponse:
    @pytest.mark.asyncio
    async def test_drain_discards_events_until_terminal(self):
        """Draining should discard events until a terminal event arrives."""
        service = _make_service()
        service._needs_drain = True

        ws = _ws_events(
            {"type": "response.output_text.delta", "delta": "stale"},
            {"type": "response.output_text.delta", "delta": "also stale"},
            {"type": "response.completed", "response": {"id": "resp_old"}},
        )
        service._websocket = ws

        await service._drain_cancelled_response()

        assert not service._needs_drain

    @pytest.mark.asyncio
    async def test_drain_handles_pending_cancel(self):
        """If cancelled before response.created, drain should send cancel
        once it sees the response.created, then continue draining."""
        service = _make_service()
        service._needs_drain = True
        service._cancel_pending_response = True

        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(
            side_effect=[
                json.dumps({"type": "response.created", "response": {"id": "resp_late"}}),
                json.dumps({"type": "response.output_text.delta", "delta": "stale"}),
                json.dumps({"type": "response.failed", "response": {"id": "resp_late"}}),
            ]
        )
        mock_ws.send = AsyncMock()
        service._websocket = mock_ws

        await service._drain_cancelled_response()

        assert not service._needs_drain
        assert not service._cancel_pending_response
        # Should have sent response.cancel
        cancel_calls = [
            call for call in mock_ws.send.call_args_list if "response.cancel" in call.args[0]
        ]
        assert len(cancel_calls) == 1

    @pytest.mark.asyncio
    async def test_drain_timeout_clears_state(self):
        """If draining times out, should clear cancellation state."""
        service = _make_service()
        service._needs_drain = True

        mock_ws = AsyncMock()
        # recv() never returns a terminal event — times out
        mock_ws.recv = AsyncMock(side_effect=asyncio.TimeoutError)
        service._websocket = mock_ws

        await service._drain_cancelled_response()

        assert not service._needs_drain
        assert not service._cancel_pending_response


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------


class TestConnectionLifecycle:
    @pytest.mark.asyncio
    async def test_disconnect_clears_previous_response_state(self):
        service = _make_service()
        service._store_previous_response_state("resp_1", [{"role": "user", "content": "hi"}], [])
        service.stop_all_metrics = AsyncMock()

        await service._disconnect()

        assert service._previous_response_id is None
        assert service._previous_input_hash is None
        assert service._previous_input_length is None

    @pytest.mark.asyncio
    async def test_reconnect_clears_state_and_reconnects(self):
        service = _make_service()
        service._store_previous_response_state("resp_1", [{"role": "user", "content": "hi"}], [])
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
            # _disconnect + _connect is the lifecycle equivalent of the old _reconnect
            await service._disconnect()
            await service._connect()

        assert service._previous_response_id is None
        mock_ws.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancellation_preserves_connection_and_sets_drain(self):
        """When process_frame is cancelled (e.g. interruption), the WebSocket
        connection should be preserved and _needs_drain set."""
        service = _make_service()
        service.stop_processing_metrics = AsyncMock()
        service.push_frame = AsyncMock()

        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.CancelledError)
        mock_ws.send = AsyncMock()
        service._websocket = mock_ws

        context = MagicMock(spec=LLMContext)
        context.tools = None
        context.tool_choice = None
        context.messages = [{"role": "user", "content": "hi"}]

        from pipecat.frames.frames import LLMContextFrame

        with pytest.raises(asyncio.CancelledError):
            await service.process_frame(LLMContextFrame(context=context), FrameDirection.DOWNSTREAM)

        # Connection should be preserved, not closed
        assert service._websocket is mock_ws
        # Should be flagged for draining before next inference
        assert service._needs_drain

    @pytest.mark.asyncio
    async def test_ensure_connected_raises_on_failure(self):
        service = _make_service()
        service._websocket = None
        # Mock _try_reconnect to simulate exhausted retries
        service._try_reconnect = AsyncMock(return_value=False)

        with pytest.raises(ConnectionError):
            await service._ensure_connected()


# ---------------------------------------------------------------------------
# Reasoning — config and param building
# ---------------------------------------------------------------------------


class TestReasoningConfig:
    def test_config_accepts_object(self):
        c = OpenAIResponsesLLMService.ReasoningConfig(effort="high", summary="detailed")
        assert c.effort == "high"
        assert c.summary == "detailed"

    def test_config_forward_compat_effort(self):
        """Forward compat: an unknown effort string passes through (`| str`)."""
        c = OpenAIResponsesLLMService.ReasoningConfig(effort="ultra")
        assert c.effort == "ultra"


class TestReasoningParams:
    def _params(self, service):
        return service._build_response_params({"input": []})

    def test_default_model_gets_no_reasoning(self):
        """The default model (gpt-4.1) does not reason, so no reasoning param is set."""
        service = _make_service()
        params = self._params(service)
        assert "reasoning" not in params
        assert "include" not in params

    def test_gpt5_series_disabled_by_default(self):
        """Mainline gpt models from gpt-5 onward default to effort="none"."""
        # gpt-6.x stands in for a future mainline series we assume will reason.
        for model in ("gpt-5.4", "gpt-5.5", "gpt-6", "gpt-6.2"):
            service = _make_service(settings=OpenAIResponsesLLMService.Settings(model=model))
            params = self._params(service)
            assert params["reasoning"] == {"effort": "none"}, model
            assert "include" not in params

    def test_o_series_left_untouched(self):
        """The o-series reasons but rejects effort="none", so leave it at the default."""
        service = _make_service(settings=OpenAIResponsesLLMService.Settings(model="o3"))
        params = self._params(service)
        assert "reasoning" not in params

    def test_gpt5_chat_variant_left_untouched(self):
        """The non-reasoning gpt-5-chat variant is excluded from the default-off logic."""
        service = _make_service(
            settings=OpenAIResponsesLLMService.Settings(model="gpt-5-chat-latest")
        )
        params = self._params(service)
        assert "reasoning" not in params

    def test_explicit_reasoning_overrides_default(self):
        """An explicit config is honored as-is (not replaced by the none default)."""
        service = _make_service(
            settings=OpenAIResponsesLLMService.Settings(
                model="gpt-5.5",
                reasoning=OpenAIResponsesLLMService.ReasoningConfig(effort="high", summary="auto"),
            )
        )
        params = self._params(service)
        assert params["reasoning"] == {"effort": "high", "summary": "auto"}
        assert params["include"] == ["reasoning.encrypted_content"]

    def test_empty_reasoning_config_falls_back_to_default(self):
        """An all-unset config is treated as unconfigured (none default on gpt-5.x)."""
        service = _make_service(
            settings=OpenAIResponsesLLMService.Settings(
                model="gpt-5.5",
                reasoning=OpenAIResponsesLLMService.ReasoningConfig(),
            )
        )
        params = self._params(service)
        assert params["reasoning"] == {"effort": "none"}
        assert "include" not in params


class TestModelSupportsReasoning:
    def test_reasoning_capable_models(self):
        """The o-series and the mainline gpt series from gpt-5 onward reason."""
        for model in ("gpt-5", "gpt-5.4", "gpt-5.5", "gpt-6", "gpt-7.1", "o1", "o3", "o4-mini"):
            assert _model_supports_reasoning(model) is True, model

    def test_known_non_reasoning_models(self):
        """gpt-4.x and earlier, plus the gpt-5-chat variant, don't reason."""
        for model in ("gpt-4.1", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-5-chat-latest"):
            assert _model_supports_reasoning(model) is False, model

    def test_unrecognized_models_are_unknown(self):
        """An unrecognized model returns None so callers can leave it alone."""
        for model in ("some-custom-model", "llama-3", "mistral-large"):
            assert _model_supports_reasoning(model) is None, model


class TestReasoningUnsupportedWarning:
    """Reasoning configured on a model that can't use it should log a clear error."""

    def _service_with_reasoning(self, model):
        return _make_service(
            settings=OpenAIResponsesLLMService.Settings(
                model=model,
                reasoning=OpenAIResponsesLLMService.ReasoningConfig(effort="high"),
            )
        )

    def test_warns_on_known_non_reasoning_model(self):
        service = self._service_with_reasoning("gpt-4.1")
        with patch("pipecat.services.openai.responses.llm.logger") as mock_logger:
            service._build_response_params({"input": []})
        assert mock_logger.error.call_count == 1
        assert "does not support reasoning" in mock_logger.error.call_args.args[0]

    def test_warns_once_per_model(self):
        service = self._service_with_reasoning("gpt-4.1")
        with patch("pipecat.services.openai.responses.llm.logger") as mock_logger:
            service._build_response_params({"input": []})
            service._build_response_params({"input": []})
        assert mock_logger.error.call_count == 1

    def test_no_warning_on_reasoning_capable_model(self):
        service = self._service_with_reasoning("gpt-5.5")
        with patch("pipecat.services.openai.responses.llm.logger") as mock_logger:
            service._build_response_params({"input": []})
        mock_logger.error.assert_not_called()

    def test_no_warning_on_unrecognized_model(self):
        """We can't be sure an unrecognized model lacks reasoning, so stay quiet."""
        service = self._service_with_reasoning("some-custom-model")
        with patch("pipecat.services.openai.responses.llm.logger") as mock_logger:
            service._build_response_params({"input": []})
        mock_logger.error.assert_not_called()


# ---------------------------------------------------------------------------
# _receive_response_events — reasoning capture
# ---------------------------------------------------------------------------


class TestReceiveResponseEventsReasoning:
    @pytest.mark.asyncio
    async def test_summary_streamed_and_reasoning_item_persisted(self):
        service = _make_service()
        service.stop_ttfb_metrics = AsyncMock()
        service.start_llm_usage_metrics = AsyncMock()
        service.push_frame = AsyncMock()
        adapter = MagicMock()
        adapter.create_llm_specific_message.side_effect = lambda m: m
        service.get_llm_adapter = MagicMock(return_value=adapter)

        ws = _ws_events(
            {"type": "response.output_item.added", "item": {"type": "reasoning", "id": "rs_1"}},
            {"type": "response.reasoning_summary_text.delta", "delta": "Think"},
            {"type": "response.reasoning_summary_text.delta", "delta": "ing..."},
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "Thinking..."}],
                    "encrypted_content": "ENCRYPTED",
                },
            },
            {
                "type": "response.completed",
                "response": {"id": "resp_1", "model": "gpt-5.4-mini", "usage": None},
            },
        )
        service._websocket = ws

        context = MagicMock(spec=LLMContext)
        await service._receive_response_events(context, [])

        pushed = [c.args[0] for c in service.push_frame.call_args_list]
        # Thought frames bracket the streamed summary: start, text, text, end.
        assert sum(isinstance(f, LLMThoughtStartFrame) for f in pushed) == 1
        assert [f.text for f in pushed if isinstance(f, LLMThoughtTextFrame)] == ["Think", "ing..."]
        assert sum(isinstance(f, LLMThoughtEndFrame) for f in pushed) == 1

        # The reasoning item is persisted for round-tripping, with its encrypted
        # content, via an LLMMessagesAppendFrame.
        append_frames = [f for f in pushed if isinstance(f, LLMMessagesAppendFrame)]
        assert len(append_frames) == 1
        stored = adapter.create_llm_specific_message.call_args[0][0]
        assert stored == {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "Thinking..."}],
            "encrypted_content": "ENCRYPTED",
        }

    @pytest.mark.asyncio
    async def test_reasoning_without_encrypted_content_not_persisted(self):
        """No encrypted content (e.g. effort='none') → nothing to round-trip."""
        service = _make_service()
        service.stop_ttfb_metrics = AsyncMock()
        service.start_llm_usage_metrics = AsyncMock()
        service.push_frame = AsyncMock()
        adapter = MagicMock()
        service.get_llm_adapter = MagicMock(return_value=adapter)

        ws = _ws_events(
            {
                "type": "response.output_item.done",
                "item": {"type": "reasoning", "id": "rs_1", "summary": []},
            },
            {
                "type": "response.completed",
                "response": {"id": "resp_1", "model": "gpt-5.4-mini", "usage": None},
            },
        )
        service._websocket = ws

        context = MagicMock(spec=LLMContext)
        await service._receive_response_events(context, [])

        pushed = [c.args[0] for c in service.push_frame.call_args_list]
        assert not [f for f in pushed if isinstance(f, LLMMessagesAppendFrame)]
        adapter.create_llm_specific_message.assert_not_called()


class TestStartsWithResponseOutputReasoning:
    def test_reasoning_matches_by_id(self):
        response_output = [{"type": "reasoning", "id": "rs_1", "encrypted_content": "ENC"}]
        items = [
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "ENC"},
            {"role": "user", "content": "next"},
        ]
        assert OpenAIResponsesLLMService._starts_with_response_output(items, response_output)

    def test_reasoning_id_mismatch_rejects(self):
        response_output = [{"type": "reasoning", "id": "rs_1"}]
        items = [{"type": "reasoning", "id": "rs_2"}]
        assert not OpenAIResponsesLLMService._starts_with_response_output(items, response_output)
