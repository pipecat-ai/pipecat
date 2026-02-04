#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for text ordering before tool calls and duplicate prevention.

These tests verify that:
1. Assistant text is added to context BEFORE tool calls (correct ordering)
2. No duplicate assistant messages are created

The LLM service pushes LLMFullResponseEndFrame BEFORE FunctionCallsStartedFrame
to ensure the aggregator flushes text to context before tool calls are processed.
"""

import unittest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    FunctionCallCancelFrame,
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    LLMContextAssistantTimestampFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
from pipecat.tests.utils import SleepFrame, run_test


class TestTextBeforeToolCall(unittest.IsolatedAsyncioTestCase):
    """Tests for text ordering and duplicate prevention with tool calls."""

    async def test_text_before_tool_call_correct_ordering(self):
        """Test that assistant text appears before tool_calls in context.

        Frame order (as sent by LLM service):
        1. LLMFullResponseStartFrame
        2. LLMTextFrame(s)
        3. LLMFullResponseEndFrame  <- triggers text flush to context
        4. FunctionCallsStartedFrame
        5. FunctionCallInProgressFrame
        """
        context = LLMContext()
        aggregator = LLMAssistantAggregator(context)

        frames_to_send = [
            LLMFullResponseStartFrame(),
            LLMTextFrame("Let me check the weather for you."),
            # LLMFullResponseEndFrame is sent BEFORE function calls
            LLMFullResponseEndFrame(),
            FunctionCallsStartedFrame(
                function_calls=[
                    FunctionCallFromLLM(
                        context=context,
                        tool_call_id="call_1",
                        function_name="get_weather",
                        arguments={"location": "LA"},
                    )
                ]
            ),
            FunctionCallInProgressFrame(
                function_name="get_weather",
                tool_call_id="call_1",
                arguments={"location": "LA"},
                cancel_on_interruption=False,
            ),
            SleepFrame(),
            FunctionCallResultFrame(
                function_name="get_weather",
                tool_call_id="call_1",
                arguments={"location": "LA"},
                result={"temp": 72},
            ),
        ]

        await run_test(aggregator, frames_to_send=frames_to_send)

        # Find indices of assistant text and tool_calls messages
        text_index = None
        tool_calls_index = None

        for i, msg in enumerate(context.messages):
            if msg.get("role") == "assistant":
                if msg.get("content") and "tool_calls" not in msg:
                    text_index = i
                elif "tool_calls" in msg:
                    tool_calls_index = i

        self.assertIsNotNone(text_index, "Assistant text message not found")
        self.assertIsNotNone(tool_calls_index, "Assistant tool_calls message not found")
        self.assertLess(
            text_index,
            tool_calls_index,
            f"Text (index {text_index}) should come before tool_calls (index {tool_calls_index})",
        )

    async def test_no_duplicate_text_messages(self):
        """Test that only one text message is added to context."""
        context = LLMContext()
        aggregator = LLMAssistantAggregator(context)

        frames_to_send = [
            LLMFullResponseStartFrame(),
            LLMTextFrame("Checking "),
            LLMTextFrame("now..."),
            LLMFullResponseEndFrame(),
            FunctionCallsStartedFrame(
                function_calls=[
                    FunctionCallFromLLM(
                        context=context,
                        tool_call_id="call_2",
                        function_name="check_status",
                        arguments={},
                    )
                ]
            ),
            FunctionCallInProgressFrame(
                function_name="check_status",
                tool_call_id="call_2",
                arguments={},
                cancel_on_interruption=False,
            ),
            SleepFrame(),
            FunctionCallResultFrame(
                function_name="check_status",
                tool_call_id="call_2",
                arguments={},
                result={"status": "ok"},
            ),
        ]

        await run_test(aggregator, frames_to_send=frames_to_send)

        # Count assistant messages with text content (not tool_calls)
        text_messages = [
            m
            for m in context.messages
            if m.get("role") == "assistant" and m.get("content") and "tool_calls" not in m
        ]

        self.assertEqual(
            len(text_messages),
            1,
            f"Expected 1 text message, got {len(text_messages)}: {text_messages}",
        )
        self.assertEqual(text_messages[0]["content"], "Checking now...")

    async def test_multiple_text_frames_aggregated_correctly(self):
        """Test that multiple text frames are aggregated into single message."""
        context = LLMContext()
        aggregator = LLMAssistantAggregator(context)

        frames_to_send = [
            LLMFullResponseStartFrame(),
            LLMTextFrame("I "),
            LLMTextFrame("will "),
            LLMTextFrame("check "),
            LLMTextFrame("that."),
            LLMFullResponseEndFrame(),
            FunctionCallsStartedFrame(
                function_calls=[
                    FunctionCallFromLLM(
                        context=context,
                        tool_call_id="call_5",
                        function_name="check",
                        arguments={},
                    )
                ]
            ),
            FunctionCallInProgressFrame(
                function_name="check",
                tool_call_id="call_5",
                arguments={},
                cancel_on_interruption=False,
            ),
            SleepFrame(),
            FunctionCallResultFrame(
                function_name="check",
                tool_call_id="call_5",
                arguments={},
                result={"done": True},
            ),
        ]

        await run_test(aggregator, frames_to_send=frames_to_send)

        text_messages = [
            m
            for m in context.messages
            if m.get("role") == "assistant" and m.get("content") and "tool_calls" not in m
        ]

        self.assertEqual(len(text_messages), 1)
        # Text should be aggregated with spaces handled
        self.assertIn("check", text_messages[0]["content"])

    async def test_tool_call_cancellation_preserves_text(self):
        """Test that text is preserved when tool call is cancelled."""
        context = LLMContext()
        aggregator = LLMAssistantAggregator(context)

        frames_to_send = [
            LLMFullResponseStartFrame(),
            LLMTextFrame("Let me search for that."),
            LLMFullResponseEndFrame(),
            FunctionCallsStartedFrame(
                function_calls=[
                    FunctionCallFromLLM(
                        context=context,
                        tool_call_id="call_cancel",
                        function_name="search",
                        arguments={"q": "test"},
                    )
                ]
            ),
            FunctionCallInProgressFrame(
                function_name="search",
                tool_call_id="call_cancel",
                arguments={"q": "test"},
                cancel_on_interruption=True,
            ),
            SleepFrame(),
            FunctionCallCancelFrame(
                function_name="search",
                tool_call_id="call_cancel",
            ),
        ]

        await run_test(aggregator, frames_to_send=frames_to_send)

        # Text should still be in context
        text_messages = [
            m
            for m in context.messages
            if m.get("role") == "assistant" and m.get("content") and "tool_calls" not in m
        ]

        self.assertEqual(len(text_messages), 1)
        self.assertEqual(text_messages[0]["content"], "Let me search for that.")


if __name__ == "__main__":
    unittest.main()
