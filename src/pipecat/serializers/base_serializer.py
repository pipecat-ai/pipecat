#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional

from pipecat.frames.frames import Frame
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport


class FrameSerializerType(Enum):
    BINARY = "binary"
    TEXT = "text"


class FrameSerializer(ABC):
    input_transport: Optional[BaseInputTransport] = None
    output_transport: Optional[BaseOutputTransport] = None

    def set_input_output_transports(
        self, input_transport: BaseInputTransport, output_transport: BaseOutputTransport
    ):
        self.input_transport = input_transport
        self.output_transport = output_transport

    @property
    @abstractmethod
    def type(self) -> FrameSerializerType:
        pass

    @abstractmethod
    def serialize(self, frame: Frame) -> str | bytes | None | List[str] | List[bytes]:
        pass

    @abstractmethod
    def deserialize(self, data: str | bytes) -> Frame | List[Frame] | None:
        pass


class AsyncFrameSerializer(FrameSerializer):
    @abstractmethod
    async def serialize(self, frame: Frame) -> str | bytes | None:
        pass

    @abstractmethod
    async def deserialize(self, data: str | bytes) -> Frame | List[Frame] | None:
        pass
