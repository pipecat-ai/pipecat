from dataclasses import dataclass
from typing import Any


class QueueFrame:
    def __eq__(self, other):
        return isinstance(other, self.__class__)


class ControlQueueFrame(QueueFrame):
    pass


class StartStreamQueueFrame(ControlQueueFrame):
    pass


class EndStreamQueueFrame(ControlQueueFrame):
    pass


class LLMResponseStartQueueFrame(QueueFrame):
    pass


class LLMResponseEndQueueFrame(QueueFrame):
    pass


@dataclass()
class AudioQueueFrame(QueueFrame):
    data: bytes


@dataclass()
class ImageQueueFrame(QueueFrame):
    url: str | None
    image: bytes


@dataclass()
class SpriteQueueFrame(QueueFrame):
    images: list[bytes]


@dataclass()
class TextQueueFrame(QueueFrame):
    text: str


@dataclass()
class TranscriptionQueueFrame(TextQueueFrame):
    participantId: str
    timestamp: str


@dataclass()
class LLMMessagesQueueFrame(QueueFrame):
    messages: list[dict[str, str]]  # TODO: define this more concretely!


class AppMessageQueueFrame(QueueFrame):
    message: Any
    participantId: str

class UserStartedSpeakingFrame(QueueFrame):
    pass

class UserStoppedSpeakingFrame(QueueFrame):
    pass
