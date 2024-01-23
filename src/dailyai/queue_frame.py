from enum import Enum
from dataclasses import dataclass
from typing import Any

"""
class FrameType(Enum):
    NOOP = -1
    START_STREAM = 0
    END_STREAM = 1
    AUDIO = 2
    IMAGE = 3
    TEXT = 4
    TRANSCRIPTION = 5
    LLM_MESSAGE = 6
    APP_MESSAGE = 7

@dataclass(frozen=True)
class QueueFrame:
    frame_type: FrameType
    frame_data: str | dict | bytes | list | None
"""
class QueueFrame:
    pass

class StartStreamQueueFrame(QueueFrame):
    pass

class EndStreamQueueFrame(QueueFrame):
    pass

@dataclass()
class AudioQueueFrame(QueueFrame):
    data: bytes

@dataclass()
class ImageQueueFrame(QueueFrame):
    url: str | None
    image: bytes

@dataclass()
class TextQueueFrame(QueueFrame):
    text: str

@dataclass()
class TranscriptionQueueFrame(TextQueueFrame):
    participantId: str
    timestamp: str

@dataclass()
class LLMMessagesQueueFrame(QueueFrame):
    messages: list[dict[str,str]] # TODO: define this more concretely!

class AppMessageQueueFrame(QueueFrame):
    message: Any
    participantId: str
