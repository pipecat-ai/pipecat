#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Frame serialization interfaces for Pipecat."""

from abc import ABC, abstractmethod

from pipecat.frames.frames import Frame, StartFrame
from pipecat.utils.base_object import BaseObject


class FrameSerializer(BaseObject):
    """Abstract base class for frame serialization implementations.

    Defines the interface for converting frames to/from serialized formats
    for transmission or storage. Subclasses must implement the core
    serialize/deserialize methods.
    """

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
