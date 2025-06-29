#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base pipeline task implementation for managing pipeline execution.

This module provides the abstract base class and configuration for pipeline
tasks that manage the lifecycle and execution of frame processing pipelines.
"""

import asyncio
from abc import abstractmethod
from dataclasses import dataclass
from typing import AsyncIterable, Iterable

from pipecat.frames.frames import Frame
from pipecat.utils.base_object import BaseObject


@dataclass
class PipelineTaskParams:
    """Configuration parameters for pipeline task execution.

    Parameters:
        loop: The asyncio event loop to use for task execution.
    """

    loop: asyncio.AbstractEventLoop


class BasePipelineTask(BaseObject):
    """Abstract base class for pipeline task implementations.

    Defines the interface for managing pipeline execution lifecycle,
    including starting, stopping, and frame queuing operations.
    """

    @abstractmethod
    def has_finished(self) -> bool:
        """Check if the pipeline task has finished execution.

        Returns:
            True if all processors have stopped and the task is complete.
        """
        pass

    @abstractmethod
    async def stop_when_done(self):
        """Schedule the pipeline to stop after processing all queued frames.

        Implementing classes should send an EndFrame or equivalent signal to
        gracefully terminate the pipeline once all current processing is complete.
        """
        pass

    @abstractmethod
    async def cancel(self):
        """Immediately stop the running pipeline.

        Implementing classes should cancel all running tasks and stop frame
        processing without waiting for completion.
        """
        pass

    @abstractmethod
    async def run(self, params: PipelineTaskParams):
        """Start and run the pipeline with the given parameters.

        Implementing classes should initialize and execute the pipeline using
        the provided configuration parameters.

        Args:
            params: Configuration parameters for pipeline execution.
        """
        pass

    @abstractmethod
    async def queue_frame(self, frame: Frame):
        """Queue a single frame for processing by the pipeline.

        Implementing classes should add the frame to their processing queue
        for downstream handling.

        Args:
            frame: The frame to be processed.
        """
        pass

    @abstractmethod
    async def queue_frames(self, frames: Iterable[Frame] | AsyncIterable[Frame]):
        """Queue multiple frames for processing by the pipeline.

        Implementing classes should process the iterable/async iterable and
        add all frames to their processing queue.

        Args:
            frames: An iterable or async iterable of frames to be processed.
        """
        pass
