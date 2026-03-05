#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for OpenAI WebSocket LLM service initialization and basic properties."""

import json
import unittest
from unittest.mock import patch

from pipecat.services.openai.websocket_llm import (
    InputParams,
    OpenAIWebSocketLLMService,
    OpenAIWebSocketLLMSettings,
)


class TestOpenAIWebSocketLLMServiceInit(unittest.TestCase):
    """Test OpenAIWebSocketLLMService constructor and basic configuration."""

    @patch("pipecat.services.openai.websocket_llm.websocket_connect")
    def test_default_init(self, _mock_ws):
        """Test service initializes with required model parameter."""
        service = OpenAIWebSocketLLMService(model="gpt-4o")
        self.assertEqual(service._settings.model, "gpt-4o")
        self.assertEqual(service._api_key, "")
        self.assertEqual(service._base_url, "wss://api.openai.com/v1/responses")
        self.assertFalse(service._settings.store)
        self.assertTrue(service._settings.use_previous_response_id)
        self.assertIsNone(service._previous_response_id)

    @patch("pipecat.services.openai.websocket_llm.websocket_connect")
    def test_custom_api_key_and_url(self, _mock_ws):
        """Test service initializes with custom API key and base URL."""
        service = OpenAIWebSocketLLMService(
            model="gpt-4o",
            api_key="test-key-123",
            base_url="wss://custom.endpoint/v1/responses",
        )
        self.assertEqual(service._api_key, "test-key-123")
        self.assertEqual(service._base_url, "wss://custom.endpoint/v1/responses")

    @patch("pipecat.services.openai.websocket_llm.websocket_connect")
    def test_store_and_previous_response_id_settings(self, _mock_ws):
        """Test store and use_previous_response_id settings."""
        service = OpenAIWebSocketLLMService(
            model="gpt-4o",
            store=True,
            use_previous_response_id=False,
        )
        self.assertTrue(service._settings.store)
        self.assertFalse(service._settings.use_previous_response_id)

    @patch("pipecat.services.openai.websocket_llm.websocket_connect")
    def test_input_params(self, _mock_ws):
        """Test input parameters are applied to settings."""
        params = InputParams(
            temperature=0.7,
            max_tokens=1024,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.3,
        )
        service = OpenAIWebSocketLLMService(model="gpt-4o", params=params)
        self.assertEqual(service._settings.temperature, 0.7)
        self.assertEqual(service._settings.max_tokens, 1024)
        self.assertEqual(service._settings.top_p, 0.9)
        self.assertEqual(service._settings.frequency_penalty, 0.5)
        self.assertEqual(service._settings.presence_penalty, 0.3)

    @patch("pipecat.services.openai.websocket_llm.websocket_connect")
    def test_default_input_params(self, _mock_ws):
        """Test default input parameters are None."""
        service = OpenAIWebSocketLLMService(model="gpt-4o")
        self.assertIsNone(service._settings.temperature)
        self.assertIsNone(service._settings.max_tokens)
        self.assertIsNone(service._settings.top_p)
        self.assertIsNone(service._settings.frequency_penalty)
        self.assertIsNone(service._settings.presence_penalty)

    @patch("pipecat.services.openai.websocket_llm.websocket_connect")
    def test_can_generate_metrics(self, _mock_ws):
        """Test that metrics generation is reported as supported."""
        service = OpenAIWebSocketLLMService(model="gpt-4o")
        self.assertTrue(service.can_generate_metrics())

    @patch("pipecat.services.openai.websocket_llm.websocket_connect")
    def test_adapter_class(self, _mock_ws):
        """Test that the correct adapter class is used."""
        from pipecat.adapters.services.open_ai_websocket_adapter import (
            OpenAIWebSocketLLMAdapter,
        )

        service = OpenAIWebSocketLLMService(model="gpt-4o")
        self.assertIsInstance(service.get_llm_adapter(), OpenAIWebSocketLLMAdapter)


class TestOpenAIWebSocketLLMSettings(unittest.TestCase):
    """Test OpenAIWebSocketLLMSettings dataclass."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = OpenAIWebSocketLLMSettings(
            model="gpt-4o",
            temperature=None,
            max_tokens=None,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=False,
            user_turn_completion_config=None,
        )
        self.assertFalse(settings.store)
        self.assertTrue(settings.use_previous_response_id)

    def test_custom_settings(self):
        """Test settings with custom values."""
        settings = OpenAIWebSocketLLMSettings(
            model="gpt-4o-mini",
            temperature=0.5,
            max_tokens=2048,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=False,
            user_turn_completion_config=None,
            store=True,
            use_previous_response_id=False,
        )
        self.assertEqual(settings.model, "gpt-4o-mini")
        self.assertEqual(settings.temperature, 0.5)
        self.assertTrue(settings.store)
        self.assertFalse(settings.use_previous_response_id)


class TestInputParams(unittest.TestCase):
    """Test InputParams Pydantic model."""

    def test_default_params(self):
        """Test default InputParams values are all None."""
        params = InputParams()
        self.assertIsNone(params.temperature)
        self.assertIsNone(params.max_tokens)
        self.assertIsNone(params.top_p)
        self.assertIsNone(params.frequency_penalty)
        self.assertIsNone(params.presence_penalty)

    def test_custom_params(self):
        """Test InputParams with custom values."""
        params = InputParams(temperature=0.8, max_tokens=500)
        self.assertEqual(params.temperature, 0.8)
        self.assertEqual(params.max_tokens, 500)


class TestTextStreamingEndToEnd(unittest.IsolatedAsyncioTestCase):
    """Test text streaming: LLMContextFrame -> mock WebSocket -> LLMTextFrame output."""

    @patch("pipecat.services.openai.websocket_llm.websocket_connect")
    async def test_text_deltas_produce_llm_text_frames(self, _mock_ws):
        """Simulate text delta messages and verify LLMTextFrame output via _on_message."""
        from unittest.mock import AsyncMock

        from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame
        from pipecat.processors.aggregators.llm_context import LLMContext

        service = OpenAIWebSocketLLMService(model="gpt-4o", api_key="test")
        service._context = LLMContext(messages=[{"role": "user", "content": "Hi"}])

        # Mock push_frame and _push_llm_text so we can track calls
        service.push_frame = AsyncMock()
        service._push_llm_text = AsyncMock()
        service.stop_ttfb_metrics = AsyncMock()
        service.stop_processing_metrics = AsyncMock()
        service.start_llm_usage_metrics = AsyncMock()

        # Simulate the sequence of messages from the WebSocket
        messages = [
            json.dumps({"type": "response.created", "response": {"id": "resp_1"}}),
            json.dumps({"type": "response.output_text.delta", "delta": "Hello"}),
            json.dumps({"type": "response.output_text.delta", "delta": " world"}),
            json.dumps({"type": "response.output_text.delta", "delta": "!"}),
            json.dumps(
                {
                    "type": "response.done",
                    "response": {
                        "id": "resp_1",
                        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                    },
                }
            ),
        ]

        for msg in messages:
            await service._on_message(msg)

        # Verify text deltas were pushed
        self.assertEqual(service._push_llm_text.call_count, 3)
        service._push_llm_text.assert_any_call("Hello")
        service._push_llm_text.assert_any_call(" world")
        service._push_llm_text.assert_any_call("!")

        # Verify response end frame was pushed
        push_frame_calls = service.push_frame.call_args_list
        end_frames = [c for c in push_frame_calls if isinstance(c[0][0], LLMFullResponseEndFrame)]
        self.assertEqual(len(end_frames), 1)

        # Verify TTFB metrics stopped on response.created
        service.stop_ttfb_metrics.assert_called_once()

    @patch("pipecat.services.openai.websocket_llm.websocket_connect")
    async def test_previous_response_id_stored_after_done(self, _mock_ws):
        """Verify previous_response_id is stored after response.done."""
        from unittest.mock import AsyncMock

        from pipecat.processors.aggregators.llm_context import LLMContext

        service = OpenAIWebSocketLLMService(model="gpt-4o", api_key="test")
        service._context = LLMContext(messages=[])
        service.push_frame = AsyncMock()
        service.stop_processing_metrics = AsyncMock()
        service.start_llm_usage_metrics = AsyncMock()

        self.assertIsNone(service._previous_response_id)

        await service._on_message(
            json.dumps(
                {
                    "type": "response.done",
                    "response": {"id": "resp_abc123", "usage": {}},
                }
            )
        )

        self.assertEqual(service._previous_response_id, "resp_abc123")

    @patch("pipecat.services.openai.websocket_llm.websocket_connect")
    async def test_previous_response_id_not_stored_when_disabled(self, _mock_ws):
        """Verify previous_response_id is NOT stored when use_previous_response_id is False."""
        from unittest.mock import AsyncMock

        service = OpenAIWebSocketLLMService(
            model="gpt-4o", api_key="test", use_previous_response_id=False
        )
        service.push_frame = AsyncMock()
        service.stop_processing_metrics = AsyncMock()
        service.start_llm_usage_metrics = AsyncMock()

        await service._on_message(
            json.dumps(
                {
                    "type": "response.done",
                    "response": {"id": "resp_xyz", "usage": {}},
                }
            )
        )

        self.assertIsNone(service._previous_response_id)

    @patch("pipecat.services.openai.websocket_llm.websocket_connect")
    async def test_empty_delta_is_ignored(self, _mock_ws):
        """Verify that an empty text delta does not produce a text frame."""
        from unittest.mock import AsyncMock

        service = OpenAIWebSocketLLMService(model="gpt-4o", api_key="test")
        service._push_llm_text = AsyncMock()

        await service._on_message(json.dumps({"type": "response.output_text.delta", "delta": ""}))

        service._push_llm_text.assert_not_called()

    @patch("pipecat.services.openai.websocket_llm.websocket_connect")
    async def test_send_response_create_format(self, _mock_ws):
        """Verify the response.create message format sent to the WebSocket."""
        from unittest.mock import AsyncMock

        from pipecat.processors.aggregators.llm_context import LLMContext

        service = OpenAIWebSocketLLMService(model="gpt-4o", api_key="test", store=True)
        service._previous_response_id = "resp_prev"
        service.push_frame = AsyncMock()
        service.start_processing_metrics = AsyncMock()
        service.start_ttfb_metrics = AsyncMock()
        service.send_with_retry = AsyncMock()

        context = LLMContext(messages=[{"role": "user", "content": "Hello"}])
        await service._send_response_create(context)

        # Verify send_with_retry was called
        service.send_with_retry.assert_called_once()
        sent_json = json.loads(service.send_with_retry.call_args[0][0])

        self.assertEqual(sent_json["type"], "response.create")
        self.assertEqual(sent_json["response"]["model"], "gpt-4o")
        self.assertTrue(sent_json["response"]["store"])
        self.assertEqual(sent_json["response"]["previous_response_id"], "resp_prev")
        self.assertIn("input", sent_json["response"])


class TestFunctionCalling(unittest.IsolatedAsyncioTestCase):
    """Test function call argument accumulation and execution via mock WebSocket."""

    @patch("pipecat.services.openai.websocket_llm.websocket_connect")
    async def test_single_function_call(self, _mock_ws):
        """Test a single function call flow: added -> delta -> done."""
        from unittest.mock import AsyncMock

        from pipecat.frames.frames import FunctionCallFromLLM
        from pipecat.processors.aggregators.llm_context import LLMContext

        service = OpenAIWebSocketLLMService(model="gpt-4o", api_key="test")
        service._context = LLMContext(messages=[{"role": "user", "content": "weather?"}])
        service.run_function_calls = AsyncMock()

        # Step 1: output_item.added with function_call
        await service._on_message(
            json.dumps(
                {
                    "type": "response.output_item.added",
                    "item": {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "get_weather",
                    },
                }
            )
        )

        self.assertIn("call_1", service._pending_function_calls)
        self.assertEqual(service._pending_function_calls["call_1"]["name"], "get_weather")

        # Step 2: argument deltas
        await service._on_message(
            json.dumps(
                {
                    "type": "response.function_call_arguments.delta",
                    "call_id": "call_1",
                    "delta": '{"city":',
                }
            )
        )
        await service._on_message(
            json.dumps(
                {
                    "type": "response.function_call_arguments.delta",
                    "call_id": "call_1",
                    "delta": ' "NYC"}',
                }
            )
        )

        self.assertEqual(service._pending_function_calls["call_1"]["arguments"], '{"city": "NYC"}')

        # Step 3: arguments done
        await service._on_message(
            json.dumps(
                {
                    "type": "response.function_call_arguments.done",
                    "call_id": "call_1",
                    "arguments": '{"city": "NYC"}',
                }
            )
        )

        # Verify run_function_calls was called
        service.run_function_calls.assert_called_once()
        func_calls = service.run_function_calls.call_args[0][0]
        self.assertEqual(len(func_calls), 1)
        self.assertEqual(func_calls[0].function_name, "get_weather")
        self.assertEqual(func_calls[0].arguments, {"city": "NYC"})
        self.assertEqual(func_calls[0].tool_call_id, "call_1")

        # Verify pending call was removed
        self.assertNotIn("call_1", service._pending_function_calls)

    @patch("pipecat.services.openai.websocket_llm.websocket_connect")
    async def test_parallel_function_calls(self, _mock_ws):
        """Test two parallel function calls in one response."""
        from unittest.mock import AsyncMock

        from pipecat.processors.aggregators.llm_context import LLMContext

        service = OpenAIWebSocketLLMService(model="gpt-4o", api_key="test")
        service._context = LLMContext(messages=[])
        service.run_function_calls = AsyncMock()

        # Add two function calls
        await service._on_message(
            json.dumps(
                {
                    "type": "response.output_item.added",
                    "item": {
                        "type": "function_call",
                        "call_id": "call_a",
                        "name": "get_weather",
                    },
                }
            )
        )
        await service._on_message(
            json.dumps(
                {
                    "type": "response.output_item.added",
                    "item": {
                        "type": "function_call",
                        "call_id": "call_b",
                        "name": "get_time",
                    },
                }
            )
        )

        self.assertEqual(len(service._pending_function_calls), 2)

        # Complete first function call
        await service._on_message(
            json.dumps(
                {
                    "type": "response.function_call_arguments.delta",
                    "call_id": "call_a",
                    "delta": '{"city": "NYC"}',
                }
            )
        )
        await service._on_message(
            json.dumps(
                {
                    "type": "response.function_call_arguments.done",
                    "call_id": "call_a",
                    "arguments": '{"city": "NYC"}',
                }
            )
        )

        # Complete second function call
        await service._on_message(
            json.dumps(
                {
                    "type": "response.function_call_arguments.delta",
                    "call_id": "call_b",
                    "delta": '{"tz": "EST"}',
                }
            )
        )
        await service._on_message(
            json.dumps(
                {
                    "type": "response.function_call_arguments.done",
                    "call_id": "call_b",
                    "arguments": '{"tz": "EST"}',
                }
            )
        )

        # Both function calls should have been executed
        self.assertEqual(service.run_function_calls.call_count, 2)

        first_call = service.run_function_calls.call_args_list[0][0][0]
        self.assertEqual(first_call[0].function_name, "get_weather")

        second_call = service.run_function_calls.call_args_list[1][0][0]
        self.assertEqual(second_call[0].function_name, "get_time")
        self.assertEqual(second_call[0].arguments, {"tz": "EST"})

        # All pending calls cleared
        self.assertEqual(len(service._pending_function_calls), 0)

    @patch("pipecat.services.openai.websocket_llm.websocket_connect")
    async def test_function_call_arguments_done_unknown_call_id(self, _mock_ws):
        """Test arguments.done with unknown call_id is handled gracefully."""
        from unittest.mock import AsyncMock

        service = OpenAIWebSocketLLMService(model="gpt-4o", api_key="test")
        service.run_function_calls = AsyncMock()

        # Should not raise, just log a warning
        await service._on_message(
            json.dumps(
                {
                    "type": "response.function_call_arguments.done",
                    "call_id": "nonexistent",
                    "arguments": "{}",
                }
            )
        )

        service.run_function_calls.assert_not_called()

    @patch("pipecat.services.openai.websocket_llm.websocket_connect")
    async def test_interruption_clears_pending_function_calls(self, _mock_ws):
        """Test that InterruptionFrame clears pending function calls."""
        from unittest.mock import AsyncMock

        from pipecat.frames.frames import InterruptionFrame
        from pipecat.processors.frame_processor import FrameDirection

        service = OpenAIWebSocketLLMService(model="gpt-4o", api_key="test")
        service.push_frame = AsyncMock()
        service.stop_all_metrics = AsyncMock()

        # Add a pending function call
        service._pending_function_calls["call_1"] = {"name": "fn", "arguments": ""}

        frame = InterruptionFrame()
        await service.process_frame(frame, FrameDirection.DOWNSTREAM)

        self.assertEqual(len(service._pending_function_calls), 0)


class TestErrorHandlingAndResponseId(unittest.IsolatedAsyncioTestCase):
    """Test error handling, previous_response_not_found fallback, response ID storage."""

    @patch("pipecat.services.openai.websocket_llm.websocket_connect")
    async def test_previous_response_not_found_clears_and_retries(self, _mock_ws):
        """Test that previous_response_not_found error clears ID and retries."""
        from unittest.mock import AsyncMock

        from pipecat.processors.aggregators.llm_context import LLMContext

        service = OpenAIWebSocketLLMService(model="gpt-4o", api_key="test")
        context = LLMContext(messages=[{"role": "user", "content": "Hi"}])
        service._context = context
        service._previous_response_id = "resp_stale"
        service._send_response_create = AsyncMock()

        await service._on_message(
            json.dumps(
                {
                    "type": "error",
                    "error": {
                        "code": "previous_response_not_found",
                        "message": "Previous response not found",
                    },
                }
            )
        )

        # previous_response_id should be cleared
        self.assertIsNone(service._previous_response_id)
        # _send_response_create should be retried with the same context
        service._send_response_create.assert_called_once_with(context)

    @patch("pipecat.services.openai.websocket_llm.websocket_connect")
    async def test_previous_response_not_found_no_retry_without_context(self, _mock_ws):
        """Test that retry does not happen when there is no context."""
        from unittest.mock import AsyncMock

        service = OpenAIWebSocketLLMService(model="gpt-4o", api_key="test")
        service._previous_response_id = "resp_stale"
        service._context = None
        service._send_response_create = AsyncMock()

        await service._on_message(
            json.dumps(
                {
                    "type": "error",
                    "error": {
                        "code": "previous_response_not_found",
                        "message": "Previous response not found",
                    },
                }
            )
        )

        self.assertIsNone(service._previous_response_id)
        service._send_response_create.assert_not_called()

    @patch("pipecat.services.openai.websocket_llm.websocket_connect")
    async def test_generic_error_pushes_error_upstream(self, _mock_ws):
        """Test that generic errors are pushed upstream via push_error."""
        from unittest.mock import AsyncMock

        service = OpenAIWebSocketLLMService(model="gpt-4o", api_key="test")
        service.push_error = AsyncMock()

        await service._on_message(
            json.dumps(
                {
                    "type": "error",
                    "error": {
                        "code": "rate_limit_exceeded",
                        "message": "Too many requests",
                    },
                }
            )
        )

        service.push_error.assert_called_once()
        call_kwargs = service.push_error.call_args[1]
        self.assertIn("rate_limit_exceeded", call_kwargs["error_msg"])
        self.assertIn("Too many requests", call_kwargs["error_msg"])

    @patch("pipecat.services.openai.websocket_llm.websocket_connect")
    async def test_response_done_with_failed_status_pushes_error(self, _mock_ws):
        """Test that response.done with status=failed pushes an error."""
        from unittest.mock import AsyncMock

        service = OpenAIWebSocketLLMService(model="gpt-4o", api_key="test")
        service.push_frame = AsyncMock()
        service.push_error = AsyncMock()
        service.stop_processing_metrics = AsyncMock()
        service.start_llm_usage_metrics = AsyncMock()

        await service._on_message(
            json.dumps(
                {
                    "type": "response.done",
                    "response": {
                        "id": "resp_fail",
                        "status": "failed",
                        "status_details": {"error": {"message": "Content filter triggered"}},
                        "usage": {},
                    },
                }
            )
        )

        service.push_error.assert_called_once()
        self.assertIn(
            "Content filter triggered",
            service.push_error.call_args[1]["error_msg"],
        )

    @patch("pipecat.services.openai.websocket_llm.websocket_connect")
    async def test_previous_response_id_persists_across_requests(self, _mock_ws):
        """Test previous_response_id is stored and used across requests."""
        from unittest.mock import AsyncMock

        from pipecat.processors.aggregators.llm_context import LLMContext

        service = OpenAIWebSocketLLMService(model="gpt-4o", api_key="test")
        service.push_frame = AsyncMock()
        service.stop_processing_metrics = AsyncMock()
        service.start_llm_usage_metrics = AsyncMock()
        service.start_processing_metrics = AsyncMock()
        service.start_ttfb_metrics = AsyncMock()
        service.send_with_retry = AsyncMock()

        # First response stores the ID
        await service._on_message(
            json.dumps(
                {
                    "type": "response.done",
                    "response": {"id": "resp_1", "usage": {}},
                }
            )
        )
        self.assertEqual(service._previous_response_id, "resp_1")

        # Second request should include the previous_response_id
        context = LLMContext(messages=[{"role": "user", "content": "More"}])
        await service._send_response_create(context)

        sent_json = json.loads(service.send_with_retry.call_args[0][0])
        self.assertEqual(sent_json["response"]["previous_response_id"], "resp_1")

        # Second response updates the ID
        await service._on_message(
            json.dumps(
                {
                    "type": "response.done",
                    "response": {"id": "resp_2", "usage": {}},
                }
            )
        )
        self.assertEqual(service._previous_response_id, "resp_2")

    @patch("pipecat.services.openai.websocket_llm.websocket_connect")
    async def test_invalid_json_message_handled_gracefully(self, _mock_ws):
        """Test that invalid JSON messages don't crash the service."""
        service = OpenAIWebSocketLLMService(model="gpt-4o", api_key="test")

        # Should not raise
        await service._on_message("not valid json {{{")

    @patch("pipecat.services.openai.websocket_llm.websocket_connect")
    async def test_usage_metrics_pushed_on_response_done(self, _mock_ws):
        """Test that usage metrics are pushed when response.done contains usage."""
        from unittest.mock import AsyncMock

        from pipecat.metrics.metrics import LLMTokenUsage

        service = OpenAIWebSocketLLMService(model="gpt-4o", api_key="test")
        service.push_frame = AsyncMock()
        service.stop_processing_metrics = AsyncMock()
        service.start_llm_usage_metrics = AsyncMock()

        await service._on_message(
            json.dumps(
                {
                    "type": "response.done",
                    "response": {
                        "id": "resp_m",
                        "usage": {
                            "input_tokens": 100,
                            "output_tokens": 50,
                            "total_tokens": 150,
                        },
                    },
                }
            )
        )

        service.start_llm_usage_metrics.assert_called_once()
        tokens = service.start_llm_usage_metrics.call_args[0][0]
        self.assertEqual(tokens.prompt_tokens, 100)
        self.assertEqual(tokens.completion_tokens, 50)
        self.assertEqual(tokens.total_tokens, 150)


if __name__ == "__main__":
    unittest.main()
