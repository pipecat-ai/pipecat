#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.frames.frames import (
    BotSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.tests.utils import SleepFrame, run_test


class TestUserIdleProcessor(unittest.IsolatedAsyncioTestCase):
    async def test_basic_idle_detection(self):
        """Test that idle callback is triggered after timeout when user stops speaking."""
        callback_called = asyncio.Event()

        async def idle_callback(processor: UserIdleProcessor) -> None:
            callback_called.set()

        # Create processor with a short timeout for testing
        processor = UserIdleProcessor(callback=idle_callback, timeout=0.1)  # 100ms timeout

        frames_to_send = [
            # Start conversation
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            # Wait 200ms - double the idle timeout to ensure it triggers
            SleepFrame(sleep=0.2),
        ]

        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        assert callback_called.is_set(), "Idle callback was not called"

    async def test_active_listening_resets_idle(self):
        """Test that bot speaking frames reset the idle timer because user is actively listening."""
        callback_called = asyncio.Event()

        async def idle_callback(processor: UserIdleProcessor) -> None:
            callback_called.set()

        processor = UserIdleProcessor(callback=idle_callback, timeout=0.2)

        frames_to_send = [
            # Start conversation
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            # Wait almost long enough for idle timeout
            SleepFrame(sleep=0.1),
            # Bot speaking frame should reset idle timer
            BotSpeakingFrame(),
            # Wait almost long enough for idle timeout again
            SleepFrame(sleep=0.1),
            # Another bot speaking frame resets timer again
            BotSpeakingFrame(),
            # Give some time for the idle timeout task to start (Python 3.10
            # doesn't really like when you create a task and then cancel it
            # right away).
            SleepFrame(sleep=0.1),
        ]

        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            BotSpeakingFrame,
            BotSpeakingFrame,
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        assert not callback_called.is_set(), (
            "Idle callback was called even though bot speaking frames reset the timer"
        )

    async def test_idle_retry_callback(self):
        """Test that retry count increases until user activity resets it."""
        retry_counts = []

        async def retry_callback(processor: UserIdleProcessor, retry_count: int) -> bool:
            retry_counts.append(retry_count)
            return True  # Keep monitoring for idle events

        processor = UserIdleProcessor(callback=retry_callback, timeout=0.4)

        frames_to_send = [
            # Start conversation
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            # Wait for first idle timeout (count=1)
            SleepFrame(sleep=0.5),
            # Wait for second idle timeout (count=2)
            SleepFrame(sleep=0.5),
            # User activity resets the count
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            # Wait for new idle timeout (count should be 1 again)
            SleepFrame(sleep=0.5),
        ]

        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        assert retry_counts == [1, 2, 1], f"Expected retry counts [1, 2, 1], got {retry_counts}"

    async def test_idle_monitoring_stops_on_false_return(self):
        """Test that idle monitoring stops when callback returns False."""
        retry_counts = []

        async def retry_callback(processor: UserIdleProcessor, retry_count: int) -> bool:
            retry_counts.append(retry_count)
            return retry_count < 2  # Stop after second retry

        processor = UserIdleProcessor(callback=retry_callback, timeout=0.4)

        frames_to_send = [
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.5),  # First retry (count=1, returns True)
            SleepFrame(sleep=0.5),  # Second retry (count=2, returns False)
            SleepFrame(sleep=0.5),  # Should not trigger callback
        ]

        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        assert retry_counts == [1, 2], f"Expected retry counts [1, 2], got {retry_counts}"

    async def test_no_idle_before_conversation(self):
        """Test that idle monitoring doesn't start before first conversation activity."""
        callback_called = asyncio.Event()

        async def idle_callback(processor: UserIdleProcessor) -> None:
            callback_called.set()

        processor = UserIdleProcessor(callback=idle_callback, timeout=0.1)

        frames_to_send = [
            SleepFrame(sleep=0.2),  # Should not trigger callback
            # No conversation activity yet
        ]

        expected_down_frames = []

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        assert not callback_called.is_set(), "Idle callback was called before conversation started"

    async def test_idle_starts_with_bot_speech(self):
        """Test that monitoring starts with bot speaking frames, not just user speech."""
        callback_called = asyncio.Event()

        async def idle_callback(processor: UserIdleProcessor) -> None:
            callback_called.set()

        processor = UserIdleProcessor(callback=idle_callback, timeout=0.1)

        frames_to_send = [
            BotStartedSpeakingFrame(),
            BotSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            SleepFrame(sleep=0.2),
        ]

        expected_down_frames = [
            BotStartedSpeakingFrame,
            BotSpeakingFrame,
            BotStoppedSpeakingFrame,
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        assert callback_called.is_set(), "Idle callback not called after bot speech"
