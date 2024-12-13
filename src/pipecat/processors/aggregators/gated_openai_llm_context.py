#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio

from pipecat.frames.frames import CancelFrame, EndFrame, Frame, StartFrame
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.sync.base_notifier import BaseNotifier


class GatedOpenAILLMContextAggregator(FrameProcessor):
    """This aggregator keeps the last received OpenAI LLM context frame and it
    doesn't let it through until the notifier is notified.

    """

    def __init__(self, notifier: BaseNotifier, **kwargs):
        super().__init__(**kwargs)
        self._notifier = notifier
        self._last_context_frame = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame)
            await self._start()
        if isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop()
            await self.push_frame(frame)
        elif isinstance(frame, OpenAILLMContextFrame):
            self._last_context_frame = frame
        else:
            await self.push_frame(frame, direction)

    async def _start(self):
        self._gate_task = self.get_event_loop().create_task(self._gate_task_handler())

    async def _stop(self):
        self._gate_task.cancel()
        await self._gate_task

    async def _gate_task_handler(self):
        while True:
            try:
                await self._notifier.wait()
                if self._last_context_frame:
                    await self.push_frame(self._last_context_frame)
                    self._last_context_frame = None
            except asyncio.CancelledError:
                break
