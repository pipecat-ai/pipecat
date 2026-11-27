#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A context frame overtaken by an interruption must not start a run."""

import asyncio
import unittest

from pipecat.frames.frames import Frame, InterruptionFrame, LLMContextFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import run_test
from tests.test_llm_service import MockLLMService


class _ContextThenInterruption(FrameProcessor):
    """Pushes a context frame and an interruption back to back, as the user
    aggregator does when the user resumes right after inference was triggered."""

    def __init__(self, yield_between: bool):
        super().__init__()
        self._yield_between = yield_between

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame) and frame.text == "go":
            await self.push_frame(LLMContextFrame(context=LLMContext()))
            if self._yield_between:
                await asyncio.sleep(0.05)
            await self.broadcast_interruption()
            return
        await self.push_frame(frame, direction)


class _RecordingLLM(MockLLMService):
    def __init__(self):
        super().__init__()
        self.runs = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            self.runs.append(frame)
        elif not isinstance(frame, InterruptionFrame):
            await self.push_frame(frame, direction)


class TestStaleContextAfterInterruption(unittest.IsolatedAsyncioTestCase):
    async def _run(self, yield_between: bool) -> list:
        llm = _RecordingLLM()
        await run_test(
            Pipeline([_ContextThenInterruption(yield_between), llm]),
            frames_to_send=[TextFrame("go"), TextFrame("pad")],
            expected_down_frames=None,
        )
        return llm.runs

    async def test_context_overtaken_by_interruption_is_dropped(self):
        self.assertEqual(await self._run(yield_between=False), [])

    async def test_context_ahead_of_interruption_still_reaches_the_service(self):
        # Processed before the interruption arrives, so the service sees it and
        # the interruption cancels the run the usual way.
        self.assertEqual(len(await self._run(yield_between=True)), 1)


if __name__ == "__main__":
    unittest.main()
