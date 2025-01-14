#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from abc import ABC, abstractmethod
from enum import Enum

from pipecat.frames.frames import Frame


class FrameSerializerType(Enum):
    BINARY = "binary"
    TEXT = "text"


class FrameSerializer(ABC):
    @property
    @abstractmethod
    def type(self) -> FrameSerializerType:
        pass

    @abstractmethod
    def serialize(self, frame: Frame) -> str | bytes | None:
        pass

    @abstractmethod
    def deserialize(self, data: str | bytes) -> Frame | None:
        pass
