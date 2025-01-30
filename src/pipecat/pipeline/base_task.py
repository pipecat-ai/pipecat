#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from abc import ABC, abstractmethod
from typing import AsyncIterable, Iterable

from pipecat.frames.frames import Frame


class BaseTask(ABC):
    @property
    @abstractmethod
    def id(self) -> int:
        """Returns the unique indetifier for this task."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of this task."""
        pass

    @abstractmethod
    def has_finished(self) -> bool:
        """Indicates whether the tasks has finished. That is, all processors
        have stopped.

        """
        pass

    @abstractmethod
    async def stop_when_done(self):
        """This is a helper function that sends an EndFrame to the pipeline in
        order to stop the task after everything in it has been processed.

        """
        pass

    @abstractmethod
    async def cancel(self):
        """
        Stops the running pipeline immediately.
        """
        pass

    @abstractmethod
    async def run(self):
        """
        Starts running the given pipeline.
        """
        pass

    @abstractmethod
    async def queue_frame(self, frame: Frame):
        """
        Queue a frame to be pushed down the pipeline.
        """
        pass

    @abstractmethod
    async def queue_frames(self, frames: Iterable[Frame] | AsyncIterable[Frame]):
        """
        Queues multiple frames to be pushed down the pipeline.
        """
        pass
