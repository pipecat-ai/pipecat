#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InterruptionFrame,
    OutputTransportMessageUrgentFrame,
    TextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test


class TestFrameProcessor(unittest.IsolatedAsyncioTestCase):
    async def test_before_after_events(self):
        identity = IdentityFilter()

        before_process_called = False
        after_process_called = False
        before_push_called = False
        after_push_called = False

        @identity.event_handler("on_before_process_frame")
        async def on_before_process_frame(filter, frame):
            nonlocal before_process_called
            before_process_called = True

        @identity.event_handler("on_after_process_frame")
        async def on_after_process_frame(filter, frame):
            nonlocal after_process_called
            after_process_called = True

        @identity.event_handler("on_before_push_frame")
        async def on_before_push_frame(filter, frame):
            nonlocal before_push_called
            before_push_called = True

        @identity.event_handler("on_after_push_frame")
        async def on_after_push_frame(filter, frame):
            nonlocal after_push_called
            after_push_called = True

        pipeline = Pipeline([identity])

        frames_to_send = [TextFrame(text="Hello cat!")]

        expected_down_frames = [TextFrame]

        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        assert before_process_called
        assert after_process_called
        assert before_push_called
        assert after_push_called

    async def test_interruption_and_wait(self):
        class DelayFrameProcessor(FrameProcessor):
            """This processors just gives time to the event loop to change
            between tasks. Otherwise things happen to fast."""

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                await asyncio.sleep(0.1)
                await self.push_frame(frame, direction)

        class InterruptFrameProcessor(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)

                if isinstance(frame, TextFrame):
                    await self.push_interruption_task_frame_and_wait()
                    await self.push_frame(OutputTransportMessageUrgentFrame(message=frame.text))
                else:
                    await self.push_frame(frame, direction)

        pipeline = Pipeline([DelayFrameProcessor(), InterruptFrameProcessor()])

        frames_to_send = [
            # Just a random interruption to make sure we don't clear anything
            # before the actual `InterruptionTaskFrame` interruption.
            InterruptionFrame(),
            # This will generate an `InterruptionTaskFrame` and will wait for an
            # `InterruptionFrame`.
            TextFrame(text="Hello from Pipecat!"),
            # Just give time for everything to complete.
            SleepFrame(sleep=0.5),
            EndFrame(),
        ]
        expected_down_frames = [
            InterruptionFrame,
            InterruptionFrame,
            OutputTransportMessageUrgentFrame,
            EndFrame,
        ]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            send_end_frame=False,
        )
