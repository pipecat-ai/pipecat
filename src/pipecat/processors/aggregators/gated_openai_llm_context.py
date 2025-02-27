#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from pipecat.frames.frames import CancelFrame, EndFrame, Frame, StartFrame
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.sync.base_notifier import BaseNotifier


class GatedOpenAILLMContextAggregator(FrameProcessor):
    """This aggregator keeps the last received OpenAI LLM context frame and it
    doesn't let it through until the notifier is notified.

    """

    def __init__(self, *, notifier: BaseNotifier, start_open: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._notifier = notifier
        self._start_open = start_open
        self._last_context_frame = None
        self._gate_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame)
            await self._start()
        if isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop()
            await self.push_frame(frame)
        elif isinstance(frame, OpenAILLMContextFrame):
            if self._start_open:
                self._start_open = False
                await self.push_frame(frame, direction)
            else:
                self._last_context_frame = frame
        else:
            await self.push_frame(frame, direction)

    async def _start(self):
        if not self._gate_task:
            self._gate_task = self.create_task(self._gate_task_handler())

    async def _stop(self):
        if self._gate_task:
            await self.cancel_task(self._gate_task)
            self._gate_task = None

    async def _gate_task_handler(self):
        while True:
            await self._notifier.wait()
            if self._last_context_frame:
                await self.push_frame(self._last_context_frame)
                self._last_context_frame = None
