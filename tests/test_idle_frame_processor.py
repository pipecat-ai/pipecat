#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.frames.frames import (
    StartFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.idle_frame_processor import IdleFrameProcessor
from pipecat.tests.utils import SleepFrame, run_test


class TestIdleFrameProcessor(unittest.IsolatedAsyncioTestCase):
    async def test_basic_idle_detection(self):
        """Test that idle callback is triggered after timeout with no frames."""
        callback_called = asyncio.Event()

        async def idle_callback(processor: IdleFrameProcessor) -> None:
            callback_called.set()

        processor = IdleFrameProcessor(callback=idle_callback, timeout=0.1)

        frames_to_send = [
            # Wait longer than timeout to trigger idle callback
            SleepFrame(sleep=0.2),
        ]

        expected_down_frames = []

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        assert callback_called.is_set(), "Idle callback was not called"

    async def test_activity_resets_timer(self):
        """Test that receiving frames resets the idle timer."""
        callback_called = asyncio.Event()

        async def idle_callback(processor: IdleFrameProcessor) -> None:
            callback_called.set()

        processor = IdleFrameProcessor(callback=idle_callback, timeout=0.2)

        frames_to_send = [
            # Send frames at intervals shorter than timeout
            SleepFrame(sleep=0.1),
            TextFrame(text="hello"),
            SleepFrame(sleep=0.1),
            TextFrame(text="world"),
            # Give some time for the idle task to start before shutdown
            SleepFrame(sleep=0.1),
        ]

        expected_down_frames = [
            TextFrame,
            TextFrame,
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        assert not callback_called.is_set(), (
            "Idle callback was called even though frames kept resetting the timer"
        )

    async def test_signal_preserved_during_callback(self):
        """Regression: event set during callback must not be cleared by finally."""

        class _Sentinel(Exception):
            pass

        async def callback(proc):
            proc._idle_event.set()  # Simulate activity arriving during callback
            raise _Sentinel()  # Exit handler to observe event state

        processor = IdleFrameProcessor(callback=callback, timeout=0.01)
        processor._idle_event = asyncio.Event()

        with self.assertRaises(_Sentinel):
            await processor._idle_task_handler()

        # Bug (finally: clear): event is cleared -> False -> FAILS
        # Fix (no finally):     event is preserved -> True -> PASSES
        self.assertTrue(
            processor._idle_event.is_set(),
            "Signal set during callback was lost — finally: clear() bug",
        )

    async def test_specific_frame_types(self):
        """Test that only monitored frame types reset the idle timer."""
        callback_called = asyncio.Event()

        async def idle_callback(processor: IdleFrameProcessor) -> None:
            callback_called.set()

        # Only monitor TranscriptionFrame
        processor = IdleFrameProcessor(
            callback=idle_callback,
            timeout=0.15,
            types=[TranscriptionFrame],
        )

        frames_to_send = [
            # TextFrame should NOT reset the timer
            SleepFrame(sleep=0.1),
            TextFrame(text="this should not reset"),
            # Wait for timeout to trigger
            SleepFrame(sleep=0.2),
        ]

        expected_down_frames = [
            TextFrame,
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        assert callback_called.is_set(), (
            "Idle callback was not called even though non-monitored frame types were sent"
        )


if __name__ == "__main__":
    unittest.main()
