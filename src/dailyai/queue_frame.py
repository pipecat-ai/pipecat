from enum import Enum
from dataclasses import dataclass

class FrameType(Enum):
    START_STREAM = 0
    END_STREAM = 1
    AUDIO = 2
    IMAGE = 3
    SENTENCE = 4
    TEXT_CHUNK = 5
    LLM_MESSAGE = 6
    APP_MESSAGE = 7
    IMAGE_DESCRIPTION = 8
    TRANSCRIPTION = 9

@dataclass(frozen=True)
class QueueFrame:
    frame_type: FrameType
    frame_data: str | dict | bytes | list | None
