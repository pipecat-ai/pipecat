#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Frame serialization interfaces for Pipecat."""

from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel

from pipecat.frames.frames import (
    Frame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
)
from pipecat.processors.frameworks.rtvi import RTVI_MESSAGE_LABEL
from pipecat.utils.base_object import BaseObject


class FrameSerializer(BaseObject):
    """Abstract base class for frame serialization implementations.

    Defines the interface for converting frames to/from serialized formats
    for transmission or storage. Subclasses must implement the core
    serialize/deserialize methods.
    """

    class InputParams(BaseModel):
        """Base configuration parameters for FrameSerializer.

        Parameters:
            ignore_rtvi_messages: Whether to ignore RTVI protocol messages during serialization.
                Defaults to True to prevent RTVI messages from being sent to external transports.
        """

        ignore_rtvi_messages: bool = True

    def __init__(self, params: Optional[InputParams] = None, **kwargs):
        """Initialize the FrameSerializer.

        Args:
            params: Configuration parameters.
            **kwargs: Additional arguments passed to BaseObject (e.g., name).
        """
        super().__init__(**kwargs)
        self._params = params or FrameSerializer.InputParams()

    def should_ignore_frame(self, frame: Frame) -> bool:
        """Check if a frame should be ignored during serialization.

        This method filters out RTVI protocol messages when ignore_rtvi_messages is enabled.
        Subclasses can override this to add additional filtering logic.

        Args:
            frame: The frame to check.

        Returns:
            True if the frame should be ignored, False otherwise.
        """
        if (
            self._params.ignore_rtvi_messages
            and isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame))
            and frame.message.get("label") == RTVI_MESSAGE_LABEL
        ):
            return True
        return False

    async def setup(self, frame: StartFrame):
        """Initialize the serializer with startup configuration.

        Args:
            frame: StartFrame containing initialization parameters.
        """
        pass

    @abstractmethod
    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Convert a frame to its serialized representation.

        Args:
            frame: The frame to serialize.

        Returns:
            Serialized frame data as string, bytes, or None if serialization fails.
        """
        pass

    @abstractmethod
    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Convert serialized data back to a frame object.

        Args:
            data: Serialized frame data as string or bytes.

        Returns:
            Reconstructed Frame object, or None if deserialization fails.
        """
        pass
