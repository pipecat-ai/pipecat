#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import (
    EndFrame,
    EndWorkerFrame,
    Frame,
    StopFrame,
    StopWorkerFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.filters.frame_filter import FrameFilter
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.filters.null_filter import NullFilter
from pipecat.processors.filters.wake_check_filter import WakeCheckFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import run_test


async def _block_all(frame: Frame) -> bool:
    return False


async def _pushed_by(processor: FrameProcessor, frame: Frame, direction: FrameDirection):
    """The frames a processor pushes while processing one frame."""
    pushed = []

    async def record(frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        pushed.append((frame, direction))

    processor.push_frame = record
    await processor.process_frame(frame, direction)
    return pushed


class TestFiltersPassFramesThatEndOrStopTheWorker(unittest.IsolatedAsyncioTestCase):
    """The worker waits for these frames, so a filter that drops one hangs it."""

    async def test_stop_and_worker_frames_pass(self):
        filters = {
            "FrameFilter": lambda: FrameFilter(types=(TextFrame,)),
            "NullFilter": NullFilter,
            "FunctionFilter": lambda: FunctionFilter(filter=_block_all, direction=None),
        }
        cases = [
            (StopFrame, FrameDirection.DOWNSTREAM),
            (EndWorkerFrame, FrameDirection.UPSTREAM),
            (StopWorkerFrame, FrameDirection.UPSTREAM),
        ]
        for name, make_filter in filters.items():
            for frame_cls, direction in cases:
                with self.subTest(filter=name, frame=frame_cls.__name__):
                    frame = frame_cls()
                    pushed = await _pushed_by(make_filter(), frame, direction)
                    self.assertEqual(pushed, [(frame, direction)])


class TestIdentifyFilter(unittest.IsolatedAsyncioTestCase):
    async def test_identity(self):
        filter = IdentityFilter()
        frames_to_send = [UserStartedSpeakingFrame(), UserStoppedSpeakingFrame()]
        expected_down_frames = [UserStartedSpeakingFrame, UserStoppedSpeakingFrame]
        await run_test(
            filter,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )


class TestFrameFilter(unittest.IsolatedAsyncioTestCase):
    async def test_text_frame(self):
        filter = FrameFilter(types=(TextFrame,))
        frames_to_send = [TextFrame(text="Hello Pipecat!")]
        expected_down_frames = [TextFrame]
        await run_test(
            filter,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

    async def test_end_frame(self):
        filter = FrameFilter(types=(EndFrame,))
        frames_to_send = [EndFrame()]
        expected_down_frames = [EndFrame]
        await run_test(
            filter,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            send_end_frame=False,
        )

    async def test_system_frame(self):
        filter = FrameFilter(types=())
        frames_to_send = [UserStartedSpeakingFrame()]
        expected_down_frames = [UserStartedSpeakingFrame]
        await run_test(
            filter,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )


class TestFunctionFilter(unittest.IsolatedAsyncioTestCase):
    async def test_passthrough(self):
        async def passthrough(frame: Frame):
            return True

        filter = FunctionFilter(filter=passthrough)
        frames_to_send = [TextFrame(text="Hello Pipecat!")]
        expected_down_frames = [TextFrame]
        await run_test(
            filter,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

    async def test_no_passthrough(self):
        async def no_passthrough(frame: Frame):
            return False

        filter = FunctionFilter(filter=no_passthrough)
        frames_to_send = [TextFrame(text="Hello Pipecat!")]
        expected_down_frames = []
        await run_test(
            filter,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

    async def test_no_direction_filters_both_directions(self):
        """When direction is None, frames in both directions are filtered."""

        class UpstreamPusher(FrameProcessor):
            """Pushes a TextFrame upstream when it receives a system frame."""

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                await self.push_frame(frame, direction)
                if isinstance(frame, UserStartedSpeakingFrame):
                    await self.push_frame(TextFrame(text="upstream"), FrameDirection.UPSTREAM)

        async def block_text(frame: Frame):
            return not isinstance(frame, TextFrame)

        # direction=None: filter applies in both directions. The downstream
        # TextFrame is blocked and the upstream TextFrame pushed by
        # UpstreamPusher is also blocked.
        filter = FunctionFilter(filter=block_text, direction=None)
        pipeline = Pipeline([filter, UpstreamPusher()])
        frames_to_send = [
            TextFrame(text="Hello!"),
            UserStartedSpeakingFrame(),
        ]
        expected_down_frames = [UserStartedSpeakingFrame]
        expected_up_frames = []
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )

    async def test_downstream_direction_passes_upstream(self):
        """When direction is DOWNSTREAM, upstream frames pass through unfiltered."""

        class UpstreamPusher(FrameProcessor):
            """Pushes a TextFrame upstream when it receives a system frame."""

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                await self.push_frame(frame, direction)
                if isinstance(frame, UserStartedSpeakingFrame):
                    await self.push_frame(TextFrame(text="upstream"), FrameDirection.UPSTREAM)

        async def block_text(frame: Frame):
            return not isinstance(frame, TextFrame)

        # direction=DOWNSTREAM: filter only applies downstream, so the
        # upstream TextFrame pushed by UpstreamPusher passes through.
        filter = FunctionFilter(filter=block_text)
        pipeline = Pipeline([filter, UpstreamPusher()])
        frames_to_send = [UserStartedSpeakingFrame()]
        expected_down_frames = [UserStartedSpeakingFrame]
        expected_up_frames = [TextFrame]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )

    async def test_upstream_direction_passes_downstream(self):
        """When direction is UPSTREAM, downstream frames pass through unfiltered."""

        class UpstreamPusher(FrameProcessor):
            """Pushes a TextFrame upstream when it receives a system frame."""

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                await self.push_frame(frame, direction)
                if isinstance(frame, UserStartedSpeakingFrame):
                    await self.push_frame(TextFrame(text="upstream"), FrameDirection.UPSTREAM)

        async def block_text(frame: Frame):
            return not isinstance(frame, TextFrame)

        # direction=UPSTREAM: filter only applies upstream, so the
        # downstream TextFrame passes through but the upstream TextFrame
        # pushed by UpstreamPusher is blocked.
        filter = FunctionFilter(filter=block_text, direction=FrameDirection.UPSTREAM)
        pipeline = Pipeline([filter, UpstreamPusher()])
        frames_to_send = [TextFrame(text="Hello!"), UserStartedSpeakingFrame()]
        expected_down_frames = [UserStartedSpeakingFrame, TextFrame]
        expected_up_frames = []
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )


class TestWakeCheckFilter(unittest.IsolatedAsyncioTestCase):
    async def test_blank_wake_phrases_do_not_pass_unrelated_speech(self):
        wake_filter = WakeCheckFilter(wake_phrases=["", " \t", "Hey, Pipecat"])
        frames_to_send = [
            TranscriptionFrame(user_id="test", text="Unrelated speech ", timestamp=""),
            TranscriptionFrame(user_id="test", text="Hey, Pipecat", timestamp=""),
        ]

        received_down, _ = await run_test(
            wake_filter,
            frames_to_send=frames_to_send,
            expected_down_frames=[TranscriptionFrame],
        )

        self.assertEqual(received_down[0].text, "Hey, Pipecat")

    async def test_no_wake_word(self):
        filter = WakeCheckFilter(wake_phrases=["Hey, Pipecat"])
        frames_to_send = [TranscriptionFrame(user_id="test", text="Phrase 1", timestamp="")]
        expected_down_frames = []
        await run_test(
            filter,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

    async def test_wake_word(self):
        filter = WakeCheckFilter(wake_phrases=["Hey, Pipecat"])
        frames_to_send = [
            TranscriptionFrame(user_id="test", text="Hey, Pipecat", timestamp=""),
            TranscriptionFrame(user_id="test", text="Phrase 1", timestamp=""),
        ]
        expected_down_frames = [TranscriptionFrame, TranscriptionFrame]
        (received_down, _) = await run_test(
            filter,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        assert received_down[-1].text == "Phrase 1"


if __name__ == "__main__":
    unittest.main()
