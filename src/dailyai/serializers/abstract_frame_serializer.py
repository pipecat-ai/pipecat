from abc import abstractmethod

from dailyai.pipeline.frames import Frame


class FrameSerializer:
    def __init__(self):
        pass

    @abstractmethod
    def serialize(self, frame: Frame) -> bytes:
        raise NotImplementedError()

    @abstractmethod
    def deserialize(self, data: bytes) -> Frame:
        raise NotImplementedError
