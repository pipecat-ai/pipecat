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
    LLMSummarizeContextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_context_summarizer import (
    LLMContextSummarizer,
    SummaryAppliedEvent,
)
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams
from pipecat.utils.context.llm_context_summarization import (
    LLMAutoContextSummarizationConfig,
    LLMContextSummaryConfig,
)


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
        config = LLMAutoContextSummarizationConfig(
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
        config = LLMAutoContextSummarizationConfig(
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
        config = LLMAutoContextSummarizationConfig(
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
        config = LLMAutoContextSummarizationConfig(
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
        config = LLMAutoContextSummarizationConfig(
            max_context_tokens=50,
            summary_config=LLMContextSummaryConfig(min_messages_after_summary=2),
        )

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
        config = LLMAutoContextSummarizationConfig(max_context_tokens=50)

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
        config = LLMAutoContextSummarizationConfig(
            max_context_tokens=50,
            summary_config=LLMContextSummaryConfig(min_messages_after_summary=2),
        )

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

    async def test_manual_summarization_via_frame(self):
        """Test that LLMSummarizeContextFrame triggers summarization on demand."""
        config = LLMAutoContextSummarizationConfig(
            max_context_tokens=100000,  # High — auto trigger would never fire
            max_unsummarized_messages=100,
        )

        summarizer = LLMContextSummarizer(
            context=self.context,
            config=config,
            auto_trigger=False,  # Disable auto; only manual requests should work
        )
        await summarizer.setup(self.task_manager)

        request_frame = None

        @summarizer.event_handler("on_request_summarization")
        async def on_request_summarization(summarizer, frame):
            nonlocal request_frame
            request_frame = frame

        # Add messages
        for i in range(5):
            self.context.add_message({"role": "user", "content": f"Message {i}"})

        # Auto-trigger should NOT fire even on LLMFullResponseStartFrame
        await summarizer.process_frame(LLMFullResponseStartFrame())
        self.assertIsNone(request_frame)

        # Manual trigger via LLMSummarizeContextFrame should fire
        await summarizer.process_frame(LLMSummarizeContextFrame())
        self.assertIsNotNone(request_frame)
        self.assertIsInstance(request_frame, LLMContextSummaryRequestFrame)

        # The request must have a valid request_id and carry the current context
        self.assertTrue(request_frame.request_id)
        self.assertEqual(request_frame.context, self.context)

        await summarizer.cleanup()

    async def test_manual_summarization_with_config_override(self):
        """Test that LLMSummarizeContextFrame can override default summary config."""
        config = LLMAutoContextSummarizationConfig(
            max_context_tokens=100000,
            summary_config=LLMContextSummaryConfig(
                target_context_tokens=6000,
                min_messages_after_summary=4,
            ),
        )

        summarizer = LLMContextSummarizer(context=self.context, config=config)
        await summarizer.setup(self.task_manager)

        request_frame = None

        @summarizer.event_handler("on_request_summarization")
        async def on_request_summarization(summarizer, frame):
            nonlocal request_frame
            request_frame = frame

        for i in range(5):
            self.context.add_message({"role": "user", "content": f"Message {i}"})

        # Push a manual frame with custom config overrides
        custom_config = LLMContextSummaryConfig(
            target_context_tokens=500,
            min_messages_after_summary=1,
        )
        await summarizer.process_frame(LLMSummarizeContextFrame(config=custom_config))

        self.assertIsNotNone(request_frame)
        # The request should use the overridden values
        self.assertEqual(request_frame.target_context_tokens, 500)
        self.assertEqual(request_frame.min_messages_to_keep, 1)

        await summarizer.cleanup()

    async def test_manual_summarization_blocked_when_in_progress(self):
        """Test that a second LLMSummarizeContextFrame is ignored while one is in progress."""
        config = LLMAutoContextSummarizationConfig(max_context_tokens=100000)

        summarizer = LLMContextSummarizer(context=self.context, config=config)
        await summarizer.setup(self.task_manager)

        request_count = 0

        @summarizer.event_handler("on_request_summarization")
        async def on_request_summarization(summarizer, frame):
            nonlocal request_count
            request_count += 1

        for i in range(5):
            self.context.add_message({"role": "user", "content": f"Message {i}"})

        # First manual request
        await summarizer.process_frame(LLMSummarizeContextFrame())
        self.assertEqual(request_count, 1)

        # Second manual request while first is in progress — should be ignored
        await summarizer.process_frame(LLMSummarizeContextFrame())
        self.assertEqual(request_count, 1)

        await summarizer.cleanup()

    async def test_summary_message_role_is_user(self):
        """Test that the summary message uses the user role."""
        config = LLMAutoContextSummarizationConfig(
            max_context_tokens=50,
            summary_config=LLMContextSummaryConfig(min_messages_after_summary=2),
        )

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

        await summarizer.process_frame(LLMFullResponseStartFrame())
        self.assertIsNotNone(request_frame)

        # Simulate receiving a summary result
        summary_result = LLMContextSummaryResultFrame(
            request_id=request_frame.request_id,
            summary="This is a test summary.",
            last_summarized_index=5,
        )
        await summarizer.process_frame(summary_result)

        # Find the summary message and verify its role is "user"
        summary_msg = next(
            (msg for msg in self.context.messages if "summary" in msg.get("content", "").lower()),
            None,
        )
        self.assertIsNotNone(summary_msg)
        self.assertEqual(summary_msg["role"], "user")

        await summarizer.cleanup()

    async def test_summary_message_default_template(self):
        """Test that the default summary_message_template is used."""
        config = LLMAutoContextSummarizationConfig(
            max_context_tokens=50,
            summary_config=LLMContextSummaryConfig(min_messages_after_summary=2),
        )

        summarizer = LLMContextSummarizer(context=self.context, config=config)
        await summarizer.setup(self.task_manager)

        for i in range(10):
            self.context.add_message({"role": "user", "content": "Test message."})

        request_frame = None

        @summarizer.event_handler("on_request_summarization")
        async def on_request_summarization(summarizer, frame):
            nonlocal request_frame
            request_frame = frame

        await summarizer.process_frame(LLMFullResponseStartFrame())

        summary_result = LLMContextSummaryResultFrame(
            request_id=request_frame.request_id,
            summary="Key facts from conversation.",
            last_summarized_index=5,
        )
        await summarizer.process_frame(summary_result)

        # Default template wraps with "Conversation summary: {summary}"
        summary_msg = next(
            (
                msg
                for msg in self.context.messages
                if "Conversation summary:" in msg.get("content", "")
            ),
            None,
        )
        self.assertIsNotNone(summary_msg)
        self.assertEqual(
            summary_msg["content"], "Conversation summary: Key facts from conversation."
        )

        await summarizer.cleanup()

    async def test_summary_message_custom_template(self):
        """Test that a custom summary_message_template is applied."""
        config = LLMAutoContextSummarizationConfig(
            max_context_tokens=50,
            summary_config=LLMContextSummaryConfig(
                min_messages_after_summary=2,
                summary_message_template="<context_summary>\n{summary}\n</context_summary>",
            ),
        )

        summarizer = LLMContextSummarizer(context=self.context, config=config)
        await summarizer.setup(self.task_manager)

        for i in range(10):
            self.context.add_message({"role": "user", "content": "Test message."})

        request_frame = None

        @summarizer.event_handler("on_request_summarization")
        async def on_request_summarization(summarizer, frame):
            nonlocal request_frame
            request_frame = frame

        await summarizer.process_frame(LLMFullResponseStartFrame())

        summary_result = LLMContextSummaryResultFrame(
            request_id=request_frame.request_id,
            summary="Key facts from conversation.",
            last_summarized_index=5,
        )
        await summarizer.process_frame(summary_result)

        # Custom template wraps with XML tags
        summary_msg = next(
            (msg for msg in self.context.messages if "<context_summary>" in msg.get("content", "")),
            None,
        )
        self.assertIsNotNone(summary_msg)
        self.assertEqual(
            summary_msg["content"],
            "<context_summary>\nKey facts from conversation.\n</context_summary>",
        )

        await summarizer.cleanup()

    async def test_on_summary_applied_event(self):
        """Test that on_summary_applied event fires with correct data."""
        config = LLMAutoContextSummarizationConfig(
            max_context_tokens=50,
            summary_config=LLMContextSummaryConfig(min_messages_after_summary=2),
        )

        summarizer = LLMContextSummarizer(context=self.context, config=config)
        await summarizer.setup(self.task_manager)

        # Add messages (1 system + 10 user = 11 total)
        for i in range(10):
            self.context.add_message({"role": "user", "content": "Test message."})

        request_frame = None
        applied_event = None

        @summarizer.event_handler("on_request_summarization")
        async def on_request_summarization(summarizer, frame):
            nonlocal request_frame
            request_frame = frame

        @summarizer.event_handler("on_summary_applied")
        async def on_summary_applied(summarizer, event):
            nonlocal applied_event
            applied_event = event

        original_count = len(self.context.messages)  # 11
        await summarizer.process_frame(LLMFullResponseStartFrame())

        # Summarize up to index 7 (system=0, user1..user7), keep last 3 (user8, user9, user10)
        summary_result = LLMContextSummaryResultFrame(
            request_id=request_frame.request_id,
            summary="Test summary.",
            last_summarized_index=7,
        )
        await summarizer.process_frame(summary_result)

        # Allow async event handler to complete
        await asyncio.sleep(0.05)

        # Verify event was fired
        self.assertIsNotNone(applied_event)
        self.assertIsInstance(applied_event, SummaryAppliedEvent)
        self.assertEqual(applied_event.original_message_count, original_count)

        # After summarization: system + summary + 3 recent = 5
        self.assertEqual(applied_event.new_message_count, 5)

        # Summarized messages: indices 1-7 = 7 messages (excluding system at index 0)
        self.assertEqual(applied_event.summarized_message_count, 7)

        # Preserved: system (1) + recent messages after index 7 (3) = 4
        self.assertEqual(applied_event.preserved_message_count, 4)

        await summarizer.cleanup()

    async def test_on_summary_applied_not_fired_on_error(self):
        """Test that on_summary_applied event is NOT fired when summarization fails."""
        config = LLMAutoContextSummarizationConfig(
            max_context_tokens=50,
            summary_config=LLMContextSummaryConfig(min_messages_after_summary=2),
        )

        summarizer = LLMContextSummarizer(context=self.context, config=config)
        await summarizer.setup(self.task_manager)

        for i in range(10):
            self.context.add_message({"role": "user", "content": "Test message."})

        request_frame = None
        applied_event = None

        @summarizer.event_handler("on_request_summarization")
        async def on_request_summarization(summarizer, frame):
            nonlocal request_frame
            request_frame = frame

        @summarizer.event_handler("on_summary_applied")
        async def on_summary_applied(summarizer, event):
            nonlocal applied_event
            applied_event = event

        await summarizer.process_frame(LLMFullResponseStartFrame())

        # Send a result with an error
        error_result = LLMContextSummaryResultFrame(
            request_id=request_frame.request_id,
            summary="",
            last_summarized_index=-1,
            error="Summarization timed out",
        )
        await summarizer.process_frame(error_result)

        await asyncio.sleep(0.05)

        # Event should NOT have fired
        self.assertIsNone(applied_event)

        await summarizer.cleanup()

    async def test_request_frame_includes_timeout(self):
        """Test that the request frame includes the configured summarization_timeout."""
        config = LLMAutoContextSummarizationConfig(
            max_context_tokens=50,
            summary_config=LLMContextSummaryConfig(summarization_timeout=60.0),
        )

        summarizer = LLMContextSummarizer(context=self.context, config=config)
        await summarizer.setup(self.task_manager)

        request_frame = None

        @summarizer.event_handler("on_request_summarization")
        async def on_request_summarization(summarizer, frame):
            nonlocal request_frame
            request_frame = frame

        for i in range(10):
            self.context.add_message({"role": "user", "content": "Test message to add tokens."})

        await summarizer.process_frame(LLMFullResponseStartFrame())

        self.assertIsNotNone(request_frame)
        self.assertEqual(request_frame.summarization_timeout, 60.0)

        await summarizer.cleanup()


if __name__ == "__main__":
    unittest.main()
