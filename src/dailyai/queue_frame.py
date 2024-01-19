from enum import Enum
from dataclasses import dataclass

class FrameType(Enum):
    NOOP = -1
    START_STREAM = 0
    END_STREAM = 1
    AUDIO = 2
    IMAGE = 3
    TEXT = 4
    LLM_MESSAGE = 5
    APP_MESSAGE = 6

@dataclass(frozen=True)
class QueueFrame:
    frame_type: FrameType
    frame_data: str | dict | bytes | list | None
