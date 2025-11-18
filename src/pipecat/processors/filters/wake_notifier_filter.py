#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Wake notifier filter for conditional frame-based notifications."""

from typing import Awaitable, Callable, Tuple, Type

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.sync.base_notifier import BaseNotifier


class WakeNotifierFilter(FrameProcessor):
    """Frame processor that conditionally triggers notifications based on frame types and filters.

    This processor monitors frames of specified types and executes a callback predicate
    when such frames are processed. If the callback returns True, the associated
    notifier is triggered, allowing for conditional wake-up or notification scenarios.
    """

    def __init__(
        self,
        notifier: BaseNotifier,
        *,
        types: Tuple[Type[Frame], ...],
        filter: Callable[[Frame], Awaitable[bool]],
        **kwargs,
    ):
        """Initialize the wake notifier filter.

        Args:
            notifier: The notifier to trigger when conditions are met.
            types: Tuple of frame types to monitor for potential notifications.
            filter: Async callback that determines whether to trigger notification.
                   Should return True to trigger notification, False otherwise.
            **kwargs: Additional arguments passed to parent FrameProcessor.
        """
        super().__init__(**kwargs)
        self._notifier = notifier
        self._types = types
        self._filter = filter

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and conditionally trigger notifications.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, self._types) and await self._filter(frame):
            await self._notifier.notify()

        await self.push_frame(frame, direction)
