#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from abc import abstractmethod
from dataclasses import dataclass
from typing import AsyncIterable, Iterable

from pipecat.frames.frames import Frame
from pipecat.utils.base_object import BaseObject


@dataclass
class PipelineTaskParams:
    """Specific configuration for the pipeline task."""

    loop: asyncio.AbstractEventLoop


class BasePipelineTask(BaseObject):
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
        """Stops the running pipeline immediately."""
        pass

    @abstractmethod
    async def run(self, params: PipelineTaskParams):
        """Starts running the given pipeline."""
        pass

    @abstractmethod
    async def queue_frame(self, frame: Frame):
        """Queue a frame to be pushed down the pipeline."""
        pass

    @abstractmethod
    async def queue_frames(self, frames: Iterable[Frame] | AsyncIterable[Frame]):
        """Queues multiple frames to be pushed down the pipeline."""
        pass
