#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Turns user speech into ACP prompts."""

import asyncio

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.acp.frames import ACPPromptFrame
from pipecat.services.acp.types import text_block


class ACPUserAggregator(FrameProcessor):
    """Collects transcriptions into one prompt per user turn.

    The ACP equivalent of an LLM user aggregator, minus the context: the agent
    keeps the conversation history, so each turn is sent on its own rather than
    appended to a message list.

    A turn is closed by a quiet period rather than by
    ``UserStoppedSpeakingFrame`` directly. Speaking frames are system frames and
    transcriptions are data frames, so the end of a turn arrives before the
    words that finish it. Every transcription received after the user stops
    speaking restarts the timer, and the prompt goes out once they stop coming.
    """

    def __init__(self, *, aggregation_timeout: float = 0.5, **kwargs):
        """Initialize the aggregator.

        Args:
            aggregation_timeout: Seconds of quiet after the user stops speaking
                before the turn is sent. Raise it for slow transcription,
                lower it for snappier hand-off.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(**kwargs)
        self._aggregation_timeout = aggregation_timeout
        self._parts: list[str] = []
        self._speaking = False
        self._text_event = asyncio.Event()
        self._aggregation_task: asyncio.Task | None = None

    async def cleanup(self):
        """Clean up resources."""
        await super().cleanup()
        await self._stop_aggregation_task()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._create_aggregation_task()
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._speaking = True
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._speaking = False
            self._text_event.set()
        elif isinstance(frame, TranscriptionFrame):
            if frame.text.strip():
                self._parts.append(frame.text.strip())
            if not self._speaking:
                self._text_event.set()
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._flush()
            await self._stop_aggregation_task()

        await self.push_frame(frame, direction)

    def _create_aggregation_task(self):
        if not self._aggregation_task:
            self._aggregation_task = self.create_task(self._aggregation_task_handler())

    async def _stop_aggregation_task(self):
        if self._aggregation_task:
            await self.cancel_task(self._aggregation_task)
            self._aggregation_task = None

    async def _aggregation_task_handler(self):
        """Flush the turn once transcriptions stop arriving."""
        while True:
            try:
                await asyncio.wait_for(self._text_event.wait(), timeout=self._aggregation_timeout)
                self._text_event.clear()
            except TimeoutError:
                if not self._speaking:
                    await self._flush()

    async def _flush(self):
        if not self._parts:
            return
        text = " ".join(self._parts)
        self._parts = []
        await self.push_frame(ACPPromptFrame(blocks=[text_block(text)]))
