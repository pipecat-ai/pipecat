#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Frame serialization interfaces for Pipecat."""

from abc import ABC, abstractmethod
from enum import Enum

from pipecat.frames.frames import Frame, StartFrame


class FrameSerializerType(Enum):
    """Enumeration of supported frame serialization formats.

    Parameters:
        BINARY: Binary serialization format for compact representation.
        TEXT: Text-based serialization format for human-readable output.
    """

    BINARY = "binary"
    TEXT = "text"


class FrameSerializer(ABC):
    """Abstract base class for frame serialization implementations.

    Defines the interface for converting frames to/from serialized formats
    for transmission or storage. Subclasses must implement serialization
    type detection and the core serialize/deserialize methods.
    """

    @property
    @abstractmethod
    def type(self) -> FrameSerializerType:
        """Get the serialization type supported by this serializer.

        Returns:
            The FrameSerializerType indicating binary or text format.
        """
        pass

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
