#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from abc import ABC, abstractmethod

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class BaseObserver(ABC):
    """This is the base class for pipeline frame observers. Observers can view
    all the frames that go through the pipeline without the need to inject
    processors in the pipeline. This can be useful, for example, to implement
    frame loggers or debuggers among other things.

    """

    @abstractmethod
    async def on_push_frame(
        self,
        src: FrameProcessor,
        dst: FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
        timestamp: int,
    ):
        """Abstract method to handle the event when a frame is pushed from one
        processor to another.

        Args:
            src (FrameProcessor): The source frame processor that is sending the frame.
            dst (FrameProcessor): The destination frame processor that will receive the frame.
            frame (Frame): The frame being transferred between processors.
            direction (FrameDirection): The direction of the frame transfer.
            timestamp (int): The timestamp when the frame was pushed (based on the pipeline clock).

        This method should be implemented by subclasses to define specific behavior
        when a frame is pushed.

        """
        pass
