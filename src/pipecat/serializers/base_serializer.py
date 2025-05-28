#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from abc import ABC, abstractmethod
from enum import Enum

from pipecat.frames.frames import EndFrame, Frame, StartFrame, TransferCallFrame


class FrameSerializerType(Enum):
    BINARY = "binary"
    TEXT = "text"


class FrameSerializer(ABC):
    @property
    @abstractmethod
    def type(self) -> FrameSerializerType:
        pass

    async def setup(self, frame: StartFrame):
        pass

    @abstractmethod
    async def serialize(self, frame: Frame) -> str | bytes | None:
        pass

    @abstractmethod
    async def deserialize(self, data: str | bytes) -> Frame | None:
        pass

    @abstractmethod
    async def _transfer_call(self, frame: TransferCallFrame):
        pass

    @abstractmethod
    async def _hang_up_call(self, frame: EndFrame):
        pass
