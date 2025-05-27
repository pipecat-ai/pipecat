#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from abc import abstractmethod
from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from pipecat.frames.frames import Frame
from pipecat.utils.base_object import BaseObject

if TYPE_CHECKING:
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


@dataclass
class FramePushed:
    """Represents an event where a frame is pushed from one processor to another
    within the pipeline.

    This data structure is typically used by observers to track the flow of
    frames through the pipeline for logging, debugging, or analytics purposes.

    Attributes:
        source (FrameProcessor): The processor sending the frame.
        destination (FrameProcessor): The processor receiving the frame.
        frame (Frame): The frame being transferred.
        direction (FrameDirection): The direction of the transfer (e.g., downstream or upstream).
        timestamp (int): The time when the frame was pushed, based on the pipeline clock.

    """

    source: "FrameProcessor"
    destination: "FrameProcessor"
    frame: Frame
    direction: "FrameDirection"
    timestamp: int


class BaseObserver(BaseObject):
    """This is the base class for pipeline frame observers. Observers can view
    all the frames that go through the pipeline without the need to inject
    processors in the pipeline. This can be useful, for example, to implement
    frame loggers or debuggers among other things.

    """

    @abstractmethod
    async def on_push_frame(self, data: FramePushed):
        """Handle the event when a frame is pushed from one processor to another.

        This method should be implemented by subclasses to define specific
        behavior (e.g., logging, monitoring, debugging) when a frame is
        transferred through the pipeline.

        Args:
            data (FramePushed): The event data containing details about the frame transfer.

        """
        pass
