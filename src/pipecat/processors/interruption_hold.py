#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Frame processor that holds bot output during tentative user interruptions."""

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    InterruptionHoldFrame,
    InterruptionResumeFrame,
    SystemFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.frame_queue import FrameQueue


class InterruptionHoldProcessor(FrameProcessor):
    """Buffers downstream output while a user turn interruption is held.

    When placed after TTS (or any response-producing branch), this processor
    buffers downstream non-system frames between :class:`InterruptionHoldFrame`
    and its resolution:

    - :class:`InterruptionResumeFrame` flushes buffered frames in arrival order
      and resumes normal pass-through.
    - :class:`InterruptionFrame` discards interruptible buffered frames while
      preserving uninterruptible frames (for example
      :class:`~pipecat.frames.frames.FunctionCallInProgressFrame`).
    - :class:`EndFrame` flushes remaining buffered frames so shutdown never hangs.
    - :class:`CancelFrame` discards buffered frames immediately.
    """

    def __init__(self, **kwargs):
        """Initialize the interruption hold processor.

        Args:
            **kwargs: Additional keyword arguments passed to :class:`FrameProcessor`.
        """
        super().__init__(**kwargs)
        self._held = False
        self._flushing = False
        self._buffer: FrameQueue = FrameQueue()

    @property
    def is_holding(self) -> bool:
        """Whether downstream non-system frames are currently being buffered."""
        return self._held

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and buffer, flush, or discard as directed.

        Args:
            frame: The incoming frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionHoldFrame):
            self._held = True
            logger.debug(f"{self}: holding downstream frames")
            await self.push_frame(frame, direction)
        elif isinstance(frame, InterruptionResumeFrame):
            was_held = self._held
            already_flushing = self._flushing
            self._held = False
            logger.debug(f"{self}: resuming downstream frames (was_held={was_held})")
            if was_held and not already_flushing:
                self._flushing = True
                await self.push_frame(frame, direction)
                await self._flush_buffer()
            else:
                await self.push_frame(frame, direction)
        elif isinstance(frame, InterruptionFrame):
            already_flushing = self._flushing
            self._held = False
            self._buffer.reset()
            has_uninterruptible = not self._buffer.empty()
            if has_uninterruptible and not already_flushing:
                self._flushing = True
            await self.push_frame(frame, direction)
            if has_uninterruptible and not already_flushing:
                await self._flush_buffer()
        elif isinstance(frame, CancelFrame):
            self._held = False
            self._flushing = False
            self._discard_all()
            await self.push_frame(frame, direction)
        elif isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
        elif direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
        elif isinstance(frame, EndFrame):
            self._held = False
            if not self._flushing:
                await self._flush_buffer()
            await self.push_frame(frame, direction)
        elif self._held or self._flushing:
            self._buffer.put_nowait(frame)
        else:
            await self.push_frame(frame, direction)

    async def _flush_buffer(self):
        """Flush buffered frames in arrival order unless re-held mid-flush."""
        self._flushing = True
        try:
            while not self._buffer.empty():
                if self._held:
                    return
                buffered = self._buffer.get_nowait()
                self._buffer.task_done()
                await self.push_frame(buffered, FrameDirection.DOWNSTREAM)
        finally:
            self._flushing = False

    def _discard_all(self):
        """Drop every frame currently held in the buffer."""
        while not self._buffer.empty():
            self._buffer.get_nowait()
            self._buffer.task_done()
