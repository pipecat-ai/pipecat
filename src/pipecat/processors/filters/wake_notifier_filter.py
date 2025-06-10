#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Awaitable, Callable, Tuple, Type

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.sync.base_notifier import BaseNotifier


class WakeNotifierFilter(FrameProcessor):
    """This processor expects a list of frame types and will execute a given
    callback predicate when a frame of any of those type is being processed. If
    the callback returns true the notifier will be notified.

    """

    def __init__(
        self,
        notifier: BaseNotifier,
        *,
        types: Tuple[Type[Frame], ...],
        filter: Callable[[Frame], Awaitable[bool]],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._notifier = notifier
        self._types = types
        self._filter = filter

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, self._types) and await self._filter(frame):
            await self._notifier.notify()

        await self.push_frame(frame, direction)
