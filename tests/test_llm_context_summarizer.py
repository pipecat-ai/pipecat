#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.frames.frames import (
    InterruptionFrame,
    LLMContextSummaryRequestFrame,
    LLMContextSummaryResultFrame,
    LLMFullResponseStartFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_context_summarizer import LLMContextSummarizer
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams
from pipecat.utils.context.llm_context_summarization import LLMContextSummarizationConfig


class TestLLMContextSummarizer(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.task_manager = TaskManager()
        self.task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

        self.context = LLMContext(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
            ]
        )

    async def test_summarization_triggered_by_token_limit(self):
        """Test that summarization is triggered when token limit is reached."""
        config = LLMContextSummarizationConfig(
            max_context_tokens=100,  # Very low to trigger easily
            max_unsummarized_messages=100,  # High so it doesn't trigger by message count
        )

        summarizer = LLMContextSummarizer(context=self.context, config=config)
        await summarizer.setup(self.task_manager)

        request_frame = None

        @summarizer.event_handler("on_request_summarization")
        async def on_request_summarization(summarizer, frame):
            nonlocal request_frame
            request_frame = frame

        # Add messages to exceed token limit
        for i in range(10):
            self.context.add_message(
                {
                    "role": "user",
                    "content": "This is a test message that adds tokens to the context.",
                }
            )

        # Trigger check by processing LLMFullResponseStartFrame
        await summarizer.process_frame(LLMFullResponseStartFrame())

        # Should have triggered summarization
        self.assertIsNotNone(request_frame)
        self.assertIsInstance(request_frame, LLMContextSummaryRequestFrame)
        self.assertEqual(request_frame.context, self.context)

        await summarizer.cleanup()

    async def test_summarization_triggered_by_message_count(self):
        """Test that summarization is triggered when message count threshold is reached."""
        config = LLMContextSummarizationConfig(
            max_context_tokens=100000,  # Very high so it doesn't trigger by tokens
            max_unsummarized_messages=5,  # Low to trigger easily
        )

        summarizer = LLMContextSummarizer(context=self.context, config=config)
        await summarizer.setup(self.task_manager)

        request_frame = None

        @summarizer.event_handler("on_request_summarization")
        async def on_request_summarization(summarizer, frame):
            nonlocal request_frame
            request_frame = frame

        # Add messages to exceed message count
        for i in range(6):
            self.context.add_message({"role": "user", "content": f"Message {i}"})

        # Trigger check
        await summarizer.process_frame(LLMFullResponseStartFrame())

        # Should have triggered summarization
        self.assertIsNotNone(request_frame)
        self.assertIsInstance(request_frame, LLMContextSummaryRequestFrame)

        await summarizer.cleanup()

    async def test_summarization_not_triggered_below_thresholds(self):
        """Test that summarization is not triggered when below thresholds."""
        config = LLMContextSummarizationConfig(
            max_context_tokens=10000,
            max_unsummarized_messages=20,
        )

        summarizer = LLMContextSummarizer(context=self.context, config=config)
        await summarizer.setup(self.task_manager)

        request_frame = None

        @summarizer.event_handler("on_request_summarization")
        async def on_request_summarization(summarizer, frame):
            nonlocal request_frame
            request_frame = frame

        # Add a few messages (below threshold)
        for i in range(3):
            self.context.add_message({"role": "user", "content": "Short message"})

        # Trigger check
        await summarizer.process_frame(LLMFullResponseStartFrame())

        # Should NOT have triggered summarization
        self.assertIsNone(request_frame)

        await summarizer.cleanup()

    async def test_summarization_in_progress_prevents_duplicate(self):
        """Test that a summarization in progress prevents triggering another."""
        config = LLMContextSummarizationConfig(
            max_context_tokens=50,  # Very low
            max_unsummarized_messages=100,
        )

        summarizer = LLMContextSummarizer(context=self.context, config=config)
        await summarizer.setup(self.task_manager)

        request_count = 0

        @summarizer.event_handler("on_request_summarization")
        async def on_request_summarization(summarizer, frame):
            nonlocal request_count
            request_count += 1

        # Add enough messages to trigger
        for i in range(10):
            self.context.add_message({"role": "user", "content": "Test message to add tokens."})

        # First trigger - should request summarization
        await summarizer.process_frame(LLMFullResponseStartFrame())
        self.assertEqual(request_count, 1)

        # Second trigger while first is in progress - should NOT request again
        await summarizer.process_frame(LLMFullResponseStartFrame())
        self.assertEqual(request_count, 1)

        await summarizer.cleanup()

    async def test_summary_result_handling(self):
        """Test that summary results are processed and applied correctly."""
        config = LLMContextSummarizationConfig(max_context_tokens=50, min_messages_after_summary=2)

        summarizer = LLMContextSummarizer(context=self.context, config=config)
        await summarizer.setup(self.task_manager)

        # Add messages and trigger summarization
        for i in range(10):
            self.context.add_message({"role": "user", "content": "Test message."})

        request_frame = None

        @summarizer.event_handler("on_request_summarization")
        async def on_request_summarization(summarizer, frame):
            nonlocal request_frame
            request_frame = frame

        original_message_count = len(self.context.messages)
        await summarizer.process_frame(LLMFullResponseStartFrame())
        self.assertIsNotNone(request_frame)

        # Simulate receiving a summary result
        summary_result = LLMContextSummaryResultFrame(
            request_id=request_frame.request_id,
            summary="This is a test summary.",
            last_summarized_index=5,
            error=None,
        )

        await summarizer.process_frame(summary_result)

        # Should have applied the summary and reduced message count
        # Expected: system message + summary message + 2 recent messages = 4 messages
        # (since last_summarized_index=5, we keep messages after index 5)
        self.assertLess(len(self.context.messages), original_message_count)

        # Check that summary was added
        summary_messages = [
            msg
            for msg in self.context.messages
            if "Conversation summary:" in msg.get("content", "")
        ]
        self.assertEqual(len(summary_messages), 1)

        await summarizer.cleanup()

    async def test_interruption_cancels_summarization(self):
        """Test that an interruption cancels pending summarization."""
        config = LLMContextSummarizationConfig(max_context_tokens=50)

        summarizer = LLMContextSummarizer(context=self.context, config=config)
        await summarizer.setup(self.task_manager)

        # Add messages and trigger summarization
        for i in range(10):
            self.context.add_message({"role": "user", "content": "Test message."})

        request_count = 0

        @summarizer.event_handler("on_request_summarization")
        async def on_request_summarization(summarizer, frame):
            nonlocal request_count
            request_count += 1

        await summarizer.process_frame(LLMFullResponseStartFrame())
        self.assertEqual(request_count, 1)

        # Process interruption
        await summarizer.process_frame(InterruptionFrame())

        # Try to trigger again - should work since the previous one was canceled
        await summarizer.process_frame(LLMFullResponseStartFrame())
        self.assertEqual(request_count, 2)

        await summarizer.cleanup()

    async def test_stale_summary_result_ignored(self):
        """Test that stale summary results are ignored."""
        config = LLMContextSummarizationConfig(max_context_tokens=50, min_messages_after_summary=2)

        summarizer = LLMContextSummarizer(context=self.context, config=config)
        await summarizer.setup(self.task_manager)

        # Add messages and trigger summarization
        for i in range(10):
            self.context.add_message({"role": "user", "content": "Test message."})

        request_frame = None

        @summarizer.event_handler("on_request_summarization")
        async def on_request_summarization(summarizer, frame):
            nonlocal request_frame
            request_frame = frame

        original_message_count = len(self.context.messages)
        await summarizer.process_frame(LLMFullResponseStartFrame())
        valid_request_id = request_frame.request_id

        # Send a stale summary result (wrong request_id)
        stale_result = LLMContextSummaryResultFrame(
            request_id="stale-id-123",
            summary="Stale summary",
            last_summarized_index=3,
            error=None,
        )

        await summarizer.process_frame(stale_result)

        # Should be ignored - message count should not change
        self.assertEqual(len(self.context.messages), original_message_count)

        # Send the correct summary result
        valid_result = LLMContextSummaryResultFrame(
            request_id=valid_request_id,
            summary="Valid summary",
            last_summarized_index=5,
            error=None,
        )

        await summarizer.process_frame(valid_result)

        # Should be processed - message count should decrease
        self.assertLess(len(self.context.messages), original_message_count)

        # Check that summary was added
        summary_messages = [
            msg
            for msg in self.context.messages
            if "Conversation summary:" in msg.get("content", "")
        ]
        self.assertEqual(len(summary_messages), 1)

        await summarizer.cleanup()


if __name__ == "__main__":
    unittest.main()
