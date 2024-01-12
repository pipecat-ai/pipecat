from enum import Enum
from dataclasses import dataclass

class FrameType(Enum):
    START_STREAM = 0
    END_STREAM = 1
    AUDIO_FRAME = 2
    IMAGE_FRAME = 3
    SENTENCE_FRAME = 4
    TEXT_CHUNK_FRAME = 5
    LLM_MESSAGE_FRAME = 6
    APP_MESSAGE_FRAME = 7
    IMAGE_DESCRIPTION = 8

@dataclass(frozen=True)
class QueueFrame:
    frame_type: FrameType
    frame_data: str | dict | bytes | list | None
