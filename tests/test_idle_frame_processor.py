#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.frames.frames import (
    TextFrame,
    TranscriptionFrame,
)
from pipecat.processors.idle_frame_processor import IdleFrameProcessor
from pipecat.tests.utils import SleepFrame, run_test


class TestIdleFrameProcessor(unittest.IsolatedAsyncioTestCase):
    async def test_basic_idle_detection(self):
        """Test that idle callback is triggered after timeout."""
        callback_called = asyncio.Event()

        async def idle_callback(processor):
            callback_called.set()

        processor = IdleFrameProcessor(callback=idle_callback, timeout=0.1)

        await run_test(
            processor,
            frames_to_send=[SleepFrame(sleep=0.2)],
            expected_down_frames=[],
        )

        assert callback_called.is_set(), "Idle callback was not called"

    async def test_activity_resets_timer(self):
        """Test that receiving frames resets the idle timer."""
        callback_called = asyncio.Event()

        async def idle_callback(processor):
            callback_called.set()

        processor = IdleFrameProcessor(callback=idle_callback, timeout=0.2)

        await run_test(
            processor,
            frames_to_send=[
                SleepFrame(sleep=0.1),
                TextFrame(text="hello"),
                SleepFrame(sleep=0.1),
                TextFrame(text="world"),
                SleepFrame(sleep=0.1),
            ],
            expected_down_frames=[TextFrame, TextFrame],
        )

        assert not callback_called.is_set()

    async def test_signal_preserved_during_callback(self):
        """Regression: event set during callback must not be cleared."""

        class _Sentinel(Exception):
            pass

        async def callback(proc):
            proc._idle_event.set()
            raise _Sentinel()

        processor = IdleFrameProcessor(callback=callback, timeout=0.01)
        processor._idle_event = asyncio.Event()

        with self.assertRaises(_Sentinel):
            await processor._idle_task_handler()

        self.assertTrue(processor._idle_event.is_set())

    async def test_specific_frame_types(self):
        """Test that only monitored frame types reset the idle timer."""
        callback_called = asyncio.Event()

        async def idle_callback(processor):
            callback_called.set()

        processor = IdleFrameProcessor(
            callback=idle_callback,
            timeout=0.15,
            types=[TranscriptionFrame],
        )

        await run_test(
            processor,
            frames_to_send=[
                SleepFrame(sleep=0.1),
                TextFrame(text="this should not reset"),
                SleepFrame(sleep=0.2),
            ],
            expected_down_frames=[TextFrame],
        )

        assert callback_called.is_set()


if __name__ == "__main__":
    unittest.main()
