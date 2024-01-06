from enum import Enum
from dataclasses import dataclass

class FrameType(Enum):
    AUDIO_FRAME = 1
    IMAGE_FRAME = 2
    START_STREAM = 3
    END_STREAM = 4


@dataclass(frozen=True)
class OutputQueueFrame:
    frame_type: FrameType
    frame_data: bytes | None
