#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Frame queue utilities for Pipecat pipeline processors."""

import asyncio
from collections.abc import Callable
from typing import Any

from pipecat.frames.frames import Frame, UninterruptibleFrame


class FrameQueue(asyncio.Queue):
    """An asyncio.Queue that tracks whether any UninterruptibleFrame is enqueued.

    Extends ``asyncio.Queue`` and maintains an O(1) ``has_uninterruptible``
    flag so interrupt-handling code can decide whether to cancel a task or
    merely drain non-uninterruptible items without scanning the queue.

    Items may be raw ``Frame`` objects or tuples whose first element is a
    ``Frame`` (e.g. ``(frame, direction, callback)``).  Pass a ``frame_getter``
    callable to extract the frame from each item; the default treats the item
    itself as the frame.

    Also exposes a ``reset()`` helper that drains all non-``UninterruptibleFrame``
    items while keeping uninterruptible ones in place.
    """

    def __init__(self, frame_getter: Callable[[Any], Frame] = lambda item: item):
        """Initialize the FrameQueue.

        Args:
            frame_getter: Callable that extracts a ``Frame`` from a queue item.
                Defaults to the identity function (item is a raw ``Frame``).
                Pass ``lambda item: item[0]`` when items are
                ``(frame, direction, callback)`` tuples.
        """
        super().__init__()
        self._frame_getter = frame_getter
        self._uninterruptible_count: int = 0

    def has_frame(self, frame_type: type[Frame] | type[UninterruptibleFrame]) -> bool:
        """Return True if any frame of the given type is in the queue.

        ``frame_type`` may be ``Frame``, ``UninterruptibleFrame`` (a mixin, not a
        ``Frame`` subclass), or any concrete frame type.

        Note:
            This inspects the internal `_queue` (deque) of asyncio.Queue.
            This is not part of the public API but is stable in CPython.

        Args:
            frame_type: The frame class to check for.

        Returns:
            True if at least one enqueued frame is an instance of ``frame_type``.
        """
        for item in self._queue:  # pyright: ignore[reportAttributeAccessIssue]
            if isinstance(self._frame_getter(item), frame_type):
                return True
        return False

    @property
    def has_uninterruptible(self) -> bool:
        """Return True if any UninterruptibleFrame is currently in the queue."""
        return self._uninterruptible_count > 0

    def _put(self, item: Any) -> None:
        if isinstance(self._frame_getter(item), UninterruptibleFrame):
            self._uninterruptible_count += 1
        super()._put(item)

    def _get(self) -> Any:
        item = super()._get()
        if isinstance(self._frame_getter(item), UninterruptibleFrame):
            self._uninterruptible_count -= 1
        return item

    def reset(self) -> None:
        """Remove all non-UninterruptibleFrame items, keeping uninterruptible ones."""
        kept: asyncio.Queue = asyncio.Queue()
        while not self.empty():
            item = self.get_nowait()
            if isinstance(self._frame_getter(item), UninterruptibleFrame):
                kept.put_nowait(item)
            self.task_done()
        while not kept.empty():
            item = kept.get_nowait()
            self.put_nowait(item)
            kept.task_done()
