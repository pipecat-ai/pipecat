#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""This module defines pipeline nodes.

A pipeline node (`PipelineNode`) wraps a frame processor (`FrameProcessor`) and
can link to previous and next nodes in the pipeline. Pipeline nodes allow
linking frame processors together with the benefit that stateless frame
processors can be re-used in different pipelines, since what is linked is the
actual pipeline node, not the frame processor itself.

"""

import asyncio
from typing import Optional

from loguru import logger

from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.utils.base_object import BaseObject


class PipelineNode(BaseObject):
    """A node in a pipeline that hosts a frame processor.

    A `PipelineNode` wraps a single `FrameProcessor` and is responsible for
    connecting it to previous and next nodes in a pipeline. It pushes frames
    emitted by its processor to the appropriate neighbor based on frame
    direction (UPSTREAM or DOWNSTREAM).
    """

    def __init__(self, processor: FrameProcessor):
        """Initialize the pipeline node with a given FrameProcessor.

        Args:
            processor: The FrameProcessor instance that this node will host.
        """
        super().__init__()
        self._processor = processor

        self._prev: Optional["PipelineNode"] = None
        self._next: Optional["PipelineNode"] = None

        self.__push_task: Optional[asyncio.Task] = None

    @property
    def processor(self) -> FrameProcessor:
        """Returns the frame processor of this pipeline node."""
        return self._processor

    @property
    def next(self) -> Optional["PipelineNode"]:
        """Get the next pipeline node.

        Returns:
            The next node, or None if there's no next node.
        """
        return self._next

    @property
    def previous(self) -> Optional["PipelineNode"]:
        """Get the previous pipeline node.

        Returns:
            The previous node, or None if there's no previous node.
        """
        return self._prev

    async def setup(self, setup: FrameProcessorSetup):
        """Set up this pipeline node.

        This sets up the wrapped frame processor with required components.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await self.processor.setup(setup)
        self._clock = setup.clock
        self._task_manager = setup.task_manager
        self._observer = setup.observer

        self.__create_push_task()

    async def cleanup(self):
        """Clean up this pipeline node."""
        await super().cleanup()
        await self.processor.cleanup()
        if self.__push_task:
            await self.__push_task
            self.__push_task = None

    def link(self, node: "PipelineNode"):
        """Link this node to the next node in the pipeline.

        Args:
            node: The node to link to.
        """
        self._next = node
        node._prev = self
        logger.debug(f"Linking {self.processor} -> {node.processor}")

    def __create_push_task(self):
        """Create the frame push task."""
        if not self.__push_task:
            self.__push_task = self._task_manager.create_task(
                self.__push_task_handler(), f"{self.processor}::_push_task"
            )

    async def __push_task_handler(self):
        """Push task handler.

        Receive frames from the wrapped frame processor and push them to the
        next or previous node depending on the direction.
        """
        async for frame, direction in self.processor:
            destination = None
            if direction == FrameDirection.DOWNSTREAM and self.next:
                logger.trace(f"Pushing {frame} from {self.processor} to {self.next.processor}")
                destination = self.next.processor
            elif direction == FrameDirection.UPSTREAM and self.previous:
                logger.trace(f"Pushing {frame} upstream from {self} to {self._prev}")
                destination = self.previous.processor

            if destination:
                await destination.queue_frame(frame, direction)

            if self._observer and destination:
                timestamp = self._clock.get_time() if self._clock else 0
                data = FramePushed(
                    source=self.processor,
                    destination=destination,
                    frame=frame,
                    direction=direction,
                    timestamp=timestamp,
                )
                await self._observer.on_push_frame(data)
