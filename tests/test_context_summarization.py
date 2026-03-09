#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for context summarization feature."""

import asyncio
import unittest
from unittest.mock import AsyncMock

from pipecat.frames.frames import LLMContextSummaryRequestFrame, LLMContextSummaryResultFrame
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage
from pipecat.services.llm_service import LLMService
from pipecat.utils.context.llm_context_summarization import (
    LLMAutoContextSummarizationConfig,
    LLMContextSummarizationConfig,
    LLMContextSummarizationUtil,
    LLMContextSummaryConfig,
)


class TestContextSummarizationMixin(unittest.TestCase):
    """Tests for LLMContextSummarizationUtil."""

    def test_estimate_tokens_simple_text(self):
        """Test token estimation with simple text."""
        # Simple sentence: "Hello world" = 11 chars / 4 = 2.75 -> 2 tokens
        tokens = LLMContextSummarizationUtil.estimate_tokens("Hello world")
        self.assertEqual(tokens, 2)

        # More words: "This is a test message" = 22 chars / 4 = 5.5 -> 5 tokens
        tokens = LLMContextSummarizationUtil.estimate_tokens("This is a test message")
        self.assertEqual(tokens, 5)

    def test_estimate_tokens_empty(self):
        """Test token estimation with empty text."""
        tokens = LLMContextSummarizationUtil.estimate_tokens("")
        self.assertEqual(tokens, 0)

    def test_estimate_context_tokens(self):
        """Test context token estimation."""
        context = LLMContext()

        # Empty context
        self.assertEqual(LLMContextSummarizationUtil.estimate_context_tokens(context), 0)

        # Add messages
        context.add_message({"role": "system", "content": "You are helpful"})  # ~4 words
        context.add_message({"role": "user", "content": "Hello"})  # ~1 word
        context.add_message({"role": "assistant", "content": "Hi there"})  # ~2 words

        # Each message has ~10 token overhead
        # Total content: ~7 words * 1.3 = ~9 tokens
        # Total overhead: 3 * 10 = 30 tokens
        # Expected: ~39 tokens
        total = LLMContextSummarizationUtil.estimate_context_tokens(context)
        self.assertGreater(total, 30)  # At least overhead
        self.assertLess(total, 50)  # Not too much

    def test_get_messages_to_summarize_basic(self):
        """Test basic message extraction for summarization."""
        context = LLMContext()

        # Add messages
        context.add_message({"role": "system", "content": "System prompt"})
        context.add_message({"role": "user", "content": "Message 1"})
        context.add_message({"role": "assistant", "content": "Response 1"})
        context.add_message({"role": "user", "content": "Message 2"})
        context.add_message({"role": "assistant", "content": "Response 2"})
        context.add_message({"role": "user", "content": "Message 3"})
        context.add_message({"role": "assistant", "content": "Response 3"})

        # Keep last 2 messages
        result = LLMContextSummarizationUtil.get_messages_to_summarize(context, 2)

        # Get first system message from context
        first_system = None
        for msg in context.messages:
            if msg.get("role") == "system":
                first_system = msg
                break

        # Should get system message
        self.assertIsNotNone(first_system)
        self.assertEqual(first_system["content"], "System prompt")

        # Should get middle messages (indices 1-4)
        self.assertEqual(len(result.messages), 4)
        self.assertEqual(result.messages[0]["content"], "Message 1")
        self.assertEqual(result.messages[-1]["content"], "Response 2")

        # Last index should be 4 (0-indexed)
        self.assertEqual(result.last_summarized_index, 4)

    def test_get_messages_to_summarize_no_system(self):
        """Test message extraction when there's no system message."""
        context = LLMContext()

        # Add messages without system prompt
        context.add_message({"role": "user", "content": "Message 1"})
        context.add_message({"role": "assistant", "content": "Response 1"})
        context.add_message({"role": "user", "content": "Message 2"})
        context.add_message({"role": "assistant", "content": "Response 2"})

        # Keep last 1 message
        result = LLMContextSummarizationUtil.get_messages_to_summarize(context, 1)

        # Get first system message from context
        first_system = None
        for msg in context.messages:
            if msg.get("role") == "system":
                first_system = msg
                break

        # Should have no system message
        self.assertIsNone(first_system)

        # Should get first 3 messages
        self.assertEqual(len(result.messages), 3)
        self.assertEqual(result.last_summarized_index, 2)

    def test_get_messages_to_summarize_insufficient(self):
        """Test when there aren't enough messages to summarize."""
        context = LLMContext()

        # Add only 2 messages
        context.add_message({"role": "user", "content": "Message 1"})
        context.add_message({"role": "assistant", "content": "Response 1"})

        # Try to keep 2 messages (same as total)
        result = LLMContextSummarizationUtil.get_messages_to_summarize(context, 2)

        # Should return empty
        self.assertEqual(len(result.messages), 0)
        self.assertEqual(result.last_summarized_index, -1)

    def test_format_messages_for_summary(self):
        """Test message formatting for summary."""

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]

        transcript = LLMContextSummarizationUtil.format_messages_for_summary(messages)

        self.assertIn("USER: Hello", transcript)
        self.assertIn("ASSISTANT: Hi there", transcript)
        self.assertIn("USER: How are you?", transcript)

    def test_format_messages_with_list_content(self):
        """Test formatting messages with list content."""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First part"},
                    {"type": "text", "text": "Second part"},
                ],
            }
        ]

        transcript = LLMContextSummarizationUtil.format_messages_for_summary(messages)

        self.assertIn("USER: First part Second part", transcript)


class TestLLMContextSummaryConfig(unittest.TestCase):
    """Tests for LLMContextSummaryConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LLMContextSummaryConfig()

        self.assertEqual(config.target_context_tokens, 6000)
        self.assertEqual(config.min_messages_after_summary, 4)
        self.assertIsNone(config.summarization_prompt)

    def test_custom_config(self):
        """Test custom configuration."""
        config = LLMContextSummaryConfig(
            target_context_tokens=2000,
            min_messages_after_summary=4,
            summarization_prompt="Custom prompt",
        )

        self.assertEqual(config.target_context_tokens, 2000)
        self.assertEqual(config.min_messages_after_summary, 4)
        self.assertEqual(config.summary_prompt, "Custom prompt")

    def test_summary_prompt_property(self):
        """Test summary_prompt property uses default when None."""
        config = LLMContextSummaryConfig()
        self.assertIn("summarizing a conversation", config.summary_prompt.lower())

        config_with_custom = LLMContextSummaryConfig(summarization_prompt="Custom")
        self.assertEqual(config_with_custom.summary_prompt, "Custom")


class TestLLMAutoContextSummarizationConfig(unittest.TestCase):
    """Tests for LLMAutoContextSummarizationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LLMAutoContextSummarizationConfig()

        self.assertEqual(config.max_context_tokens, 8000)
        self.assertEqual(config.max_unsummarized_messages, 20)
        self.assertEqual(config.summary_config.target_context_tokens, 6000)
        self.assertEqual(config.summary_config.min_messages_after_summary, 4)

    def test_custom_config(self):
        """Test custom configuration."""
        config = LLMAutoContextSummarizationConfig(
            max_context_tokens=2500,
            max_unsummarized_messages=15,
            summary_config=LLMContextSummaryConfig(
                target_context_tokens=2000,
                min_messages_after_summary=4,
                summarization_prompt="Custom prompt",
            ),
        )

        self.assertEqual(config.max_context_tokens, 2500)
        self.assertEqual(config.max_unsummarized_messages, 15)
        self.assertEqual(config.summary_config.target_context_tokens, 2000)
        self.assertEqual(config.summary_config.min_messages_after_summary, 4)
        self.assertEqual(config.summary_config.summary_prompt, "Custom prompt")

    def test_target_tokens_auto_adjusted(self):
        """Test that target_context_tokens is auto-adjusted when it exceeds max."""
        config = LLMAutoContextSummarizationConfig(
            max_context_tokens=1000,
            summary_config=LLMContextSummaryConfig(target_context_tokens=9000),
        )
        self.assertLessEqual(config.summary_config.target_context_tokens, config.max_context_tokens)

    def test_max_context_tokens_none(self):
        """Test that max_context_tokens can be None when max_unsummarized_messages is set."""
        config = LLMAutoContextSummarizationConfig(
            max_context_tokens=None,
            max_unsummarized_messages=20,
        )
        self.assertIsNone(config.max_context_tokens)
        self.assertEqual(config.max_unsummarized_messages, 20)

    def test_max_unsummarized_messages_none(self):
        """Test that max_unsummarized_messages can be None when max_context_tokens is set."""
        config = LLMAutoContextSummarizationConfig(
            max_context_tokens=8000,
            max_unsummarized_messages=None,
        )
        self.assertEqual(config.max_context_tokens, 8000)
        self.assertIsNone(config.max_unsummarized_messages)

    def test_both_none_raises(self):
        """Test that setting both thresholds to None raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            LLMAutoContextSummarizationConfig(
                max_context_tokens=None,
                max_unsummarized_messages=None,
            )
        self.assertIn("at least one", str(cm.exception).lower())

    def test_target_tokens_not_auto_adjusted_when_max_none(self):
        """Test that target_context_tokens is not auto-adjusted when max_context_tokens is None."""
        config = LLMAutoContextSummarizationConfig(
            max_context_tokens=None,
            max_unsummarized_messages=10,
            summary_config=LLMContextSummaryConfig(target_context_tokens=9000),
        )
        # target_context_tokens should remain unchanged since there's no max to compare against
        self.assertEqual(config.summary_config.target_context_tokens, 9000)


class TestLLMContextSummarizationConfigDeprecated(unittest.TestCase):
    """Tests for deprecated LLMContextSummarizationConfig."""

    def test_emits_deprecation_warning(self):
        """Test that instantiating the deprecated config emits a DeprecationWarning."""
        with self.assertWarns(DeprecationWarning):
            LLMContextSummarizationConfig()

    def test_to_auto_config(self):
        """Test conversion to the new LLMAutoContextSummarizationConfig."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            old_config = LLMContextSummarizationConfig(
                max_context_tokens=2500,
                target_context_tokens=2000,
                max_unsummarized_messages=15,
                min_messages_after_summary=4,
                summarization_prompt="Custom",
            )

        new_config = old_config.to_auto_config()

        self.assertIsInstance(new_config, LLMAutoContextSummarizationConfig)
        self.assertEqual(new_config.max_context_tokens, 2500)
        self.assertEqual(new_config.max_unsummarized_messages, 15)
        self.assertEqual(new_config.summary_config.target_context_tokens, 2000)
        self.assertEqual(new_config.summary_config.min_messages_after_summary, 4)
        self.assertEqual(new_config.summary_config.summarization_prompt, "Custom")


class TestFunctionCallHandling(unittest.TestCase):
    """Tests for function call handling in summarization."""

    def test_function_call_in_progress_not_summarized(self):
        """Test that messages with function calls in progress are not summarized."""
        context = LLMContext()

        # Add messages including a function call without result
        context.add_message({"role": "system", "content": "System prompt"})
        context.add_message({"role": "user", "content": "What time is it?"})
        context.add_message(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": "{}"},
                    }
                ],
            }
        )
        # No tool result yet - function call is in progress
        context.add_message({"role": "user", "content": "Latest message"})

        # Try to keep last 1 message
        result = LLMContextSummarizationUtil.get_messages_to_summarize(context, 1)

        # Should only get the first user message, stopping before the function call
        self.assertEqual(len(result.messages), 1)
        self.assertEqual(result.messages[0]["content"], "What time is it?")
        self.assertEqual(result.last_summarized_index, 1)

    def test_completed_function_call_can_be_summarized(self):
        """Test that completed function calls can be summarized."""
        context = LLMContext()

        # Add messages including a complete function call sequence
        context.add_message({"role": "system", "content": "System prompt"})
        context.add_message({"role": "user", "content": "What time is it?"})
        context.add_message(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": "{}"},
                    }
                ],
            }
        )
        # Tool result completes the function call
        context.add_message(
            {"role": "tool", "tool_call_id": "call_123", "content": '{"time": "10:30 AM"}'}
        )
        context.add_message({"role": "assistant", "content": "It's 10:30 AM"})
        context.add_message({"role": "user", "content": "Latest message"})

        # Try to keep last 1 message
        result = LLMContextSummarizationUtil.get_messages_to_summarize(context, 1)

        # Should get all messages except the last one (complete function call is included)
        self.assertEqual(len(result.messages), 4)
        self.assertEqual(result.messages[0]["content"], "What time is it?")
        self.assertEqual(result.messages[-1]["content"], "It's 10:30 AM")
        self.assertEqual(result.last_summarized_index, 4)

    def test_multiple_function_calls_in_progress(self):
        """Test handling of multiple function calls in progress."""
        context = LLMContext()

        # Add messages with multiple function calls
        context.add_message({"role": "system", "content": "System prompt"})
        context.add_message({"role": "user", "content": "Message 1"})
        context.add_message({"role": "assistant", "content": "Response 1"})
        context.add_message({"role": "user", "content": "What's the time and date?"})
        context.add_message(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_time",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": "{}"},
                    },
                    {
                        "id": "call_date",
                        "type": "function",
                        "function": {"name": "get_date", "arguments": "{}"},
                    },
                ],
            }
        )
        # Only one tool result - other call still in progress
        context.add_message(
            {"role": "tool", "tool_call_id": "call_time", "content": '{"time": "10:30 AM"}'}
        )
        context.add_message({"role": "user", "content": "Latest message"})

        # Try to keep last 1 message
        result = LLMContextSummarizationUtil.get_messages_to_summarize(context, 1)

        # Should stop before the function call that's in progress
        # Messages to summarize: indices 1, 2, 3 (stops before index 4 where incomplete call is)
        self.assertEqual(len(result.messages), 3)
        self.assertEqual(result.last_summarized_index, 3)

    def test_multiple_completed_function_calls(self):
        """Test that multiple completed function calls can be summarized."""
        context = LLMContext()

        # Add messages with multiple completed function calls
        context.add_message({"role": "system", "content": "System prompt"})
        context.add_message({"role": "user", "content": "What's the time and date?"})
        context.add_message(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_time",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": "{}"},
                    },
                    {
                        "id": "call_date",
                        "type": "function",
                        "function": {"name": "get_date", "arguments": "{}"},
                    },
                ],
            }
        )
        # Both tool results provided
        context.add_message(
            {"role": "tool", "tool_call_id": "call_time", "content": '{"time": "10:30 AM"}'}
        )
        context.add_message(
            {
                "role": "tool",
                "tool_call_id": "call_date",
                "content": '{"date": "January 1, 2024"}',
            }
        )
        context.add_message({"role": "assistant", "content": "It's 10:30 AM on January 1, 2024"})
        context.add_message({"role": "user", "content": "Latest message"})

        # Try to keep last 1 message
        result = LLMContextSummarizationUtil.get_messages_to_summarize(context, 1)

        # Should get all messages except the last one (all function calls completed)
        self.assertEqual(len(result.messages), 5)
        self.assertEqual(result.last_summarized_index, 5)

    def test_sequential_function_calls_mixed_completion(self):
        """Test sequential function calls with mixed completion states."""
        context = LLMContext()

        # Add messages with sequential function calls
        context.add_message({"role": "system", "content": "System prompt"})

        # First function call - completed
        context.add_message({"role": "user", "content": "What time is it?"})
        context.add_message(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": "{}"},
                    }
                ],
            }
        )
        context.add_message(
            {"role": "tool", "tool_call_id": "call_1", "content": '{"time": "10:30 AM"}'}
        )
        context.add_message({"role": "assistant", "content": "It's 10:30 AM"})

        # Second function call - in progress
        context.add_message({"role": "user", "content": "What's the date?"})
        context.add_message(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "get_date", "arguments": "{}"},
                    }
                ],
            }
        )
        # No result for call_2 yet
        context.add_message({"role": "user", "content": "Latest message"})

        # Try to keep last 1 message
        result = LLMContextSummarizationUtil.get_messages_to_summarize(context, 1)

        # Should get messages up to and including the first completed function call
        # but stop before the second function call that's in progress
        # Messages to summarize: indices 1, 2, 3, 4, 5 (stops before index 6 where incomplete call is)
        self.assertEqual(len(result.messages), 5)
        self.assertEqual(result.messages[-1]["content"], "What's the date?")
        self.assertEqual(result.last_summarized_index, 5)

    def test_function_call_formatting_in_transcript(self):
        """Test that function calls are properly formatted in transcript."""

        messages = [
            {"role": "user", "content": "What time is it?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": '{"time": "10:30 AM"}'},
            {"role": "assistant", "content": "It's 10:30 AM"},
        ]

        transcript = LLMContextSummarizationUtil.format_messages_for_summary(messages)

        # Check that function call is included
        self.assertIn("TOOL_CALL: get_time({})", transcript)
        # Check that tool result is included
        self.assertIn('TOOL_RESULT[call_123]: {"time": "10:30 AM"}', transcript)

    def test_no_function_calls(self):
        """Test that summarization works normally without function calls."""
        context = LLMContext()

        # Add normal conversation without function calls
        context.add_message({"role": "system", "content": "System prompt"})
        context.add_message({"role": "user", "content": "Hello"})
        context.add_message({"role": "assistant", "content": "Hi"})
        context.add_message({"role": "user", "content": "How are you?"})
        context.add_message({"role": "assistant", "content": "I'm good"})
        context.add_message({"role": "user", "content": "Latest message"})

        # Try to keep last 1 message
        result = LLMContextSummarizationUtil.get_messages_to_summarize(context, 1)

        # Should get all messages except the last one
        self.assertEqual(len(result.messages), 4)
        self.assertEqual(result.last_summarized_index, 4)


class TestSummaryGenerationExceptions(unittest.IsolatedAsyncioTestCase):
    """Tests for summary generation exception handling."""

    async def test_generate_summary_raises_on_no_messages(self):
        """Test that _generate_summary raises RuntimeError when there are no messages to summarize."""
        llm_service = LLMService()
        context = LLMContext()

        # Add only one message (system), which isn't enough to summarize
        context.add_message({"role": "system", "content": "System prompt"})

        frame = LLMContextSummaryRequestFrame(
            request_id="test",
            context=context,
            min_messages_to_keep=1,
            target_context_tokens=1000,
            summarization_prompt="Summarize this",
        )

        with self.assertRaises(RuntimeError) as cm:
            await llm_service._generate_summary(frame)

        self.assertEqual(str(cm.exception), "No messages to summarize")

    async def test_generate_summary_raises_on_no_run_inference(self):
        """Test that _generate_summary raises RuntimeError when run_inference is not implemented."""
        # Create a minimal LLM service - base class raises NotImplementedError
        llm_service = LLMService()

        context = LLMContext()
        context.add_message({"role": "user", "content": "Message 1"})
        context.add_message({"role": "assistant", "content": "Response 1"})
        context.add_message({"role": "user", "content": "Message 2"})

        frame = LLMContextSummaryRequestFrame(
            request_id="test",
            context=context,
            min_messages_to_keep=1,
            target_context_tokens=1000,
            summarization_prompt="Summarize this",
        )

        with self.assertRaises(RuntimeError) as cm:
            await llm_service._generate_summary(frame)

        self.assertIn("does not implement run_inference", str(cm.exception))
        self.assertIn("LLMService", str(cm.exception))

    async def test_generate_summary_raises_on_empty_response(self):
        """Test that _generate_summary raises RuntimeError when LLM returns empty summary."""
        llm_service = LLMService()
        # Mock run_inference to return None
        llm_service.run_inference = AsyncMock(return_value=None)

        context = LLMContext()
        context.add_message({"role": "user", "content": "Message 1"})
        context.add_message({"role": "assistant", "content": "Response 1"})
        context.add_message({"role": "user", "content": "Message 2"})

        frame = LLMContextSummaryRequestFrame(
            request_id="test",
            context=context,
            min_messages_to_keep=1,
            target_context_tokens=1000,
            summarization_prompt="Summarize this",
        )

        with self.assertRaises(RuntimeError) as cm:
            await llm_service._generate_summary(frame)

        self.assertEqual(str(cm.exception), "LLM returned empty summary")

    async def test_generate_summary_task_handles_exceptions(self):
        """Test that _generate_summary_task properly handles exceptions from _generate_summary."""
        llm_service = LLMService()

        # Mock broadcast_frame to capture the result
        broadcast_calls = []

        async def mock_broadcast(frame_class, **kwargs):
            broadcast_calls.append((frame_class, kwargs))

        llm_service.broadcast_frame = mock_broadcast

        # Mock push_error
        llm_service.push_error = AsyncMock()

        context = LLMContext()
        context.add_message({"role": "system", "content": "System prompt"})

        frame = LLMContextSummaryRequestFrame(
            request_id="test_123",
            context=context,
            min_messages_to_keep=1,
            target_context_tokens=1000,
            summarization_prompt="Summarize this",
        )

        # Execute the task
        await llm_service._generate_summary_task(frame)

        # Verify broadcast_frame was called with error
        self.assertEqual(len(broadcast_calls), 1)
        frame_class, kwargs = broadcast_calls[0]
        self.assertEqual(kwargs["request_id"], "test_123")
        self.assertEqual(kwargs["summary"], "")
        self.assertEqual(kwargs["last_summarized_index"], -1)
        self.assertEqual(
            kwargs["error"], "Error generating context summary: No messages to summarize"
        )

        # Verify push_error was called
        llm_service.push_error.assert_called_once()

    async def test_generate_summary_success(self):
        """Test that _generate_summary returns successfully with valid input."""
        llm_service = LLMService()
        # Mock run_inference to return a summary
        llm_service.run_inference = AsyncMock(return_value="This is a summary of the conversation")

        context = LLMContext()
        context.add_message({"role": "user", "content": "Message 1"})
        context.add_message({"role": "assistant", "content": "Response 1"})
        context.add_message({"role": "user", "content": "Message 2"})

        frame = LLMContextSummaryRequestFrame(
            request_id="test",
            context=context,
            min_messages_to_keep=1,
            target_context_tokens=1000,
            summarization_prompt="Summarize this",
        )

        summary, last_index = await llm_service._generate_summary(frame)

        self.assertEqual(summary, "This is a summary of the conversation")
        self.assertGreater(last_index, -1)
        self.assertEqual(last_index, 1)  # Should be the index of the last summarized message

    async def test_generate_summary_task_timeout(self):
        """Test that _generate_summary_task handles timeout correctly."""
        llm_service = LLMService()

        # Mock _generate_summary to hang
        async def slow_summary(frame):
            await asyncio.sleep(10)
            return ("summary", 1)

        llm_service._generate_summary = slow_summary

        broadcast_calls = []

        async def mock_broadcast(frame_class, **kwargs):
            broadcast_calls.append((frame_class, kwargs))

        llm_service.broadcast_frame = mock_broadcast
        llm_service.push_error = AsyncMock()

        context = LLMContext()
        context.add_message({"role": "user", "content": "Message 1"})
        context.add_message({"role": "assistant", "content": "Response 1"})
        context.add_message({"role": "user", "content": "Message 2"})

        frame = LLMContextSummaryRequestFrame(
            request_id="timeout_test",
            context=context,
            min_messages_to_keep=1,
            target_context_tokens=1000,
            summarization_prompt="Summarize this",
            summarization_timeout=0.1,  # Very short timeout
        )

        await llm_service._generate_summary_task(frame)

        # Should have broadcast an error result
        self.assertEqual(len(broadcast_calls), 1)
        _, kwargs = broadcast_calls[0]
        self.assertEqual(kwargs["request_id"], "timeout_test")
        self.assertEqual(kwargs["summary"], "")
        self.assertEqual(kwargs["last_summarized_index"], -1)
        # error is None for timeout path (push_error is called instead)
        self.assertIsNone(kwargs["error"])

        # push_error should have been called with timeout message
        llm_service.push_error.assert_called_once()
        call_args = llm_service.push_error.call_args
        error_msg = call_args.kwargs.get("error_msg") or call_args.args[0]
        self.assertIn("timed out", error_msg)


class TestDedicatedLLMSummarization(unittest.IsolatedAsyncioTestCase):
    """Tests for dedicated LLM summarization in LLMContextSummarizer."""

    async def asyncSetUp(self):
        from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

        self.task_manager = TaskManager()
        self.task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

    def _create_context_and_config(self, dedicated_llm):
        """Create a context with enough messages and a config with a dedicated LLM."""
        context = LLMContext()
        for i in range(10):
            context.add_message(
                {"role": "user", "content": f"Test message {i} that adds tokens to context."}
            )

        config = LLMAutoContextSummarizationConfig(
            max_context_tokens=50,  # Very low to trigger easily
            summary_config=LLMContextSummaryConfig(
                llm=dedicated_llm,
                summarization_timeout=5.0,
            ),
        )
        return context, config

    async def test_dedicated_llm_success(self):
        """Test that dedicated LLM generates summary and applies result."""
        from pipecat.processors.aggregators.llm_context_summarizer import LLMContextSummarizer

        dedicated_llm = LLMService()
        dedicated_llm._generate_summary = AsyncMock(return_value=("Dedicated summary", 5))

        context, config = self._create_context_and_config(dedicated_llm)
        original_message_count = len(context.messages)
        summarizer = LLMContextSummarizer(context=context, config=config)
        await summarizer.setup(self.task_manager)

        # Track whether on_request_summarization event fires (it should NOT)
        event_fired = False

        @summarizer.event_handler("on_request_summarization")
        async def on_request_summarization(summarizer, frame):
            nonlocal event_fired
            event_fired = True

        # Trigger summarization via LLM response start
        from pipecat.frames.frames import LLMFullResponseStartFrame

        await summarizer.process_frame(LLMFullResponseStartFrame())

        # Wait for the background task to complete
        await asyncio.sleep(0.1)

        # The event should NOT have fired (dedicated LLM handles it internally)
        self.assertFalse(event_fired)

        # Verify the dedicated LLM was called
        dedicated_llm._generate_summary.assert_called_once()

        # Verify summary was applied to context (message count should decrease)
        self.assertLess(len(context.messages), original_message_count)

        # Verify summary message is present
        summary_messages = [
            msg for msg in context.messages if "Conversation summary:" in msg.get("content", "")
        ]
        self.assertEqual(len(summary_messages), 1)
        self.assertIn("Dedicated summary", summary_messages[0]["content"])

        await summarizer.cleanup()

    async def test_dedicated_llm_timeout(self):
        """Test that dedicated LLM timeout produces error and clears state."""
        from pipecat.processors.aggregators.llm_context_summarizer import LLMContextSummarizer

        dedicated_llm = LLMService()

        async def slow_summary(frame):
            await asyncio.sleep(10)
            return ("summary", 1)

        dedicated_llm._generate_summary = slow_summary

        context, config = self._create_context_and_config(dedicated_llm)
        config.summary_config.summarization_timeout = 0.1  # Very short timeout
        summarizer = LLMContextSummarizer(context=context, config=config)
        await summarizer.setup(self.task_manager)

        original_message_count = len(context.messages)

        # Trigger summarization
        from pipecat.frames.frames import LLMFullResponseStartFrame

        await summarizer.process_frame(LLMFullResponseStartFrame())

        # Wait for the background task to complete (timeout + some buffer)
        await asyncio.sleep(0.3)

        # Context should be unchanged (timeout = error = no summary applied)
        self.assertEqual(len(context.messages), original_message_count)

        # Summarization state should be cleared so new requests can be made
        self.assertFalse(summarizer._summarization_in_progress)

        await summarizer.cleanup()

    async def test_dedicated_llm_exception(self):
        """Test that dedicated LLM exceptions produce error and clear state."""
        from pipecat.processors.aggregators.llm_context_summarizer import LLMContextSummarizer

        dedicated_llm = LLMService()
        dedicated_llm._generate_summary = AsyncMock(
            side_effect=RuntimeError("LLM connection failed")
        )

        context, config = self._create_context_and_config(dedicated_llm)
        summarizer = LLMContextSummarizer(context=context, config=config)
        await summarizer.setup(self.task_manager)

        original_message_count = len(context.messages)

        # Trigger summarization
        from pipecat.frames.frames import LLMFullResponseStartFrame

        await summarizer.process_frame(LLMFullResponseStartFrame())

        # Wait for the background task to complete
        await asyncio.sleep(0.1)

        # Context should be unchanged (exception = error = no summary applied)
        self.assertEqual(len(context.messages), original_message_count)

        # Summarization state should be cleared
        self.assertFalse(summarizer._summarization_in_progress)

        await summarizer.cleanup()

    async def test_dedicated_llm_does_not_emit_event(self):
        """Test that summarizer does NOT emit on_request_summarization when dedicated LLM is set."""
        from pipecat.processors.aggregators.llm_context_summarizer import LLMContextSummarizer

        dedicated_llm = LLMService()
        dedicated_llm._generate_summary = AsyncMock(return_value=("Summary", 1))

        context, config = self._create_context_and_config(dedicated_llm)
        summarizer = LLMContextSummarizer(context=context, config=config)
        await summarizer.setup(self.task_manager)

        event_fired = False

        @summarizer.event_handler("on_request_summarization")
        async def on_request_summarization(summarizer, frame):
            nonlocal event_fired
            event_fired = True

        from pipecat.frames.frames import LLMFullResponseStartFrame

        await summarizer.process_frame(LLMFullResponseStartFrame())
        await asyncio.sleep(0.1)

        self.assertFalse(event_fired)

        await summarizer.cleanup()

    async def test_no_dedicated_llm_emits_event(self):
        """Test that summarizer emits on_request_summarization when no dedicated LLM."""
        from pipecat.processors.aggregators.llm_context_summarizer import LLMContextSummarizer

        context = LLMContext()
        for i in range(10):
            context.add_message(
                {"role": "user", "content": f"Test message {i} that adds tokens to context."}
            )

        config = LLMAutoContextSummarizationConfig(max_context_tokens=50)
        summarizer = LLMContextSummarizer(context=context, config=config)
        await summarizer.setup(self.task_manager)

        request_frame = None

        @summarizer.event_handler("on_request_summarization")
        async def on_request_summarization(summarizer, frame):
            nonlocal request_frame
            request_frame = frame

        from pipecat.frames.frames import LLMFullResponseStartFrame

        await summarizer.process_frame(LLMFullResponseStartFrame())

        self.assertIsNotNone(request_frame)
        self.assertIsInstance(request_frame, LLMContextSummaryRequestFrame)

        await summarizer.cleanup()


class TestOrphanedToolResponseDetection(unittest.TestCase):
    """Tests that tool responses in the kept range are treated as orphans.

    The scan in _get_function_calls_in_progress_index is bounded by summary_end,
    so a tool response that falls in the kept portion (>= summary_end) never
    resolves its matching tool call.  This ensures the assistant+tool_calls
    message and all its responses stay together in the kept range.
    """

    def test_tool_response_in_kept_range_is_treated_as_orphan(self):
        """Tool response in the kept range causes the tool call to be kept too."""
        context = LLMContext()
        context.add_message({"role": "system", "content": "System prompt"})  # idx 0
        context.add_message({"role": "user", "content": "Hello"})  # idx 1
        context.add_message(  # idx 2: assistant with tool_call
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "fn", "arguments": "{}"},
                    }
                ],
            }
        )
        context.add_message(
            {"role": "tool", "tool_call_id": "call_1", "content": "result"}
        )  # idx 3 (kept)
        context.add_message({"role": "user", "content": "Thanks"})  # idx 4 (kept)

        # Keep 2: summary_end=3. The tool response at idx 3 is outside the scan
        # range → call_1 stays pending → boundary moves back to idx 2.
        result = LLMContextSummarizationUtil.get_messages_to_summarize(context, 2)
        self.assertEqual(result.last_summarized_index, 1)
        self.assertEqual(result.messages[-1]["content"], "Hello")

    def test_tool_response_in_summarized_range_is_not_orphan(self):
        """Tool response within the summarized range correctly resolves its call."""
        context = LLMContext()
        context.add_message({"role": "system", "content": "System prompt"})  # idx 0
        context.add_message({"role": "user", "content": "Hello"})  # idx 1
        context.add_message(  # idx 2: assistant with tool_call
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "fn", "arguments": "{}"},
                    }
                ],
            }
        )
        context.add_message(
            {"role": "tool", "tool_call_id": "call_1", "content": "result"}
        )  # idx 3
        context.add_message({"role": "assistant", "content": "Done"})  # idx 4
        context.add_message({"role": "user", "content": "Thanks"})  # idx 5 (kept)

        # Keep 1: summary_end=5. Both the tool call (idx 2) and its response
        # (idx 3) are within the scan range → resolved → no adjustment.
        result = LLMContextSummarizationUtil.get_messages_to_summarize(context, 1)
        self.assertEqual(result.last_summarized_index, 4)
        self.assertEqual(len(result.messages), 4)

    def test_partial_responses_in_kept_range_moves_back(self):
        """When only some tool responses are in the kept range the whole group is kept."""
        context = LLMContext()
        context.add_message({"role": "system", "content": "System prompt"})  # idx 0
        context.add_message({"role": "user", "content": "Hello"})  # idx 1
        context.add_message(  # idx 2: assistant with two tool_calls
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "fn_a", "arguments": "{}"},
                    },
                    {
                        "id": "call_b",
                        "type": "function",
                        "function": {"name": "fn_b", "arguments": "{}"},
                    },
                ],
            }
        )
        context.add_message(
            {"role": "tool", "tool_call_id": "call_a", "content": "result_a"}
        )  # idx 3
        context.add_message(
            {"role": "tool", "tool_call_id": "call_b", "content": "result_b"}
        )  # idx 4 (kept)
        context.add_message({"role": "user", "content": "Thanks"})  # idx 5 (kept)

        # Keep 2: summary_end=4. call_a is resolved (idx 3 is in scan range) but
        # call_b's response (idx 4) is outside → call_b stays pending →
        # function_call_start=2 → boundary moves back to idx 2.
        result = LLMContextSummarizationUtil.get_messages_to_summarize(context, 2)
        self.assertEqual(result.last_summarized_index, 1)
        self.assertEqual(result.messages[-1]["content"], "Hello")

    def test_non_adjacent_orphan_in_kept_range_moves_back(self):
        """Orphaned tool response deeper in the kept range (not at the boundary) is detected."""
        context = LLMContext()
        context.add_message({"role": "system", "content": "System prompt"})  # idx 0
        context.add_message({"role": "user", "content": "Hello"})  # idx 1
        context.add_message(  # idx 2: assistant with two tool_calls
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "fn_a", "arguments": "{}"},
                    },
                    {
                        "id": "call_b",
                        "type": "function",
                        "function": {"name": "fn_b", "arguments": "{}"},
                    },
                ],
            }
        )
        context.add_message(
            {"role": "tool", "tool_call_id": "call_a", "content": "result_a"}
        )  # idx 3
        context.add_message({"role": "user", "content": "Intermediate"})  # idx 4 (kept)
        context.add_message(
            {"role": "tool", "tool_call_id": "call_b", "content": "result_b"}
        )  # idx 5 (kept) — NOT adjacent to the boundary
        context.add_message({"role": "user", "content": "Latest"})  # idx 6 (kept)

        # Keep 3: summary_end=4. call_b's response is at idx 5, two hops into
        # the kept range. The scan stops at idx 4, so call_b is never resolved →
        # function_call_start=2 → boundary moves back to idx 2.
        result = LLMContextSummarizationUtil.get_messages_to_summarize(context, 3)
        self.assertEqual(result.last_summarized_index, 1)
        self.assertEqual(result.messages[-1]["content"], "Hello")


class TestLLMSpecificMessageHandling(unittest.TestCase):
    """Tests that LLMSpecificMessage objects are correctly skipped in summarization."""

    def test_estimate_context_tokens_skips_specific_messages(self):
        """Test that estimate_context_tokens skips LLMSpecificMessage objects."""
        context = LLMContext()
        context.add_message({"role": "user", "content": "Hello"})
        context.add_message(LLMSpecificMessage(llm="google", message={}))
        context.add_message({"role": "assistant", "content": "Hi there"})

        tokens_with_specific = LLMContextSummarizationUtil.estimate_context_tokens(context)

        context_without = LLMContext()
        context_without.add_message({"role": "user", "content": "Hello"})
        context_without.add_message({"role": "assistant", "content": "Hi there"})
        tokens_without = LLMContextSummarizationUtil.estimate_context_tokens(context_without)

        self.assertEqual(tokens_with_specific, tokens_without)

    def test_get_messages_to_summarize_with_specific_messages(self):
        """Test that get_messages_to_summarize handles LLMSpecificMessage objects."""
        context = LLMContext()
        context.add_message({"role": "system", "content": "System prompt"})
        context.add_message(LLMSpecificMessage(llm="google", message={}))
        context.add_message({"role": "user", "content": "Message 1"})
        context.add_message({"role": "assistant", "content": "Response 1"})
        context.add_message(LLMSpecificMessage(llm="google", message={}))
        context.add_message({"role": "user", "content": "Message 2"})
        context.add_message({"role": "assistant", "content": "Response 2"})

        result = LLMContextSummarizationUtil.get_messages_to_summarize(context, 2)

        self.assertEqual(len(result.messages), 4)
        self.assertEqual(result.last_summarized_index, 4)

    def test_format_messages_skips_specific_messages(self):
        """Test that format_messages_for_summary skips LLMSpecificMessage objects."""
        messages = [
            {"role": "user", "content": "Hello"},
            LLMSpecificMessage(llm="google", message={}),
            {"role": "assistant", "content": "Hi there"},
        ]

        transcript = LLMContextSummarizationUtil.format_messages_for_summary(messages)

        self.assertIn("USER: Hello", transcript)
        self.assertIn("ASSISTANT: Hi there", transcript)

    def test_function_call_tracking_skips_specific_messages(self):
        """Test that _get_function_calls_in_progress_index skips LLMSpecificMessage."""
        messages = [
            {"role": "user", "content": "What time is it?"},
            LLMSpecificMessage(llm="google", message={}),
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": "{}"},
                    }
                ],
            },
            LLMSpecificMessage(llm="google", message={}),
            {"role": "tool", "tool_call_id": "call_123", "content": '{"time": "10:30 AM"}'},
        ]

        result = LLMContextSummarizationUtil._get_earliest_function_call_not_resolved_in_range(
            messages, 0, len(messages)
        )
        self.assertEqual(result, -1)


if __name__ == "__main__":
    unittest.main()
