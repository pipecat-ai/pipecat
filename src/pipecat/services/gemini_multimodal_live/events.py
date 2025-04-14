#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
#

import base64
import io
import json
from enum import Enum
from typing import List, Literal, Optional

from PIL import Image
from pydantic import BaseModel, Field

from pipecat.frames.frames import ImageRawFrame

#
# Client events
#


class MediaChunk(BaseModel):
    mimeType: str
    data: str


class ContentPart(BaseModel):
    text: Optional[str] = Field(default=None, validate_default=False)
    inlineData: Optional[MediaChunk] = Field(default=None, validate_default=False)


class Turn(BaseModel):
    role: Literal["user", "model"] = "user"
    parts: List[ContentPart]


class StartSensitivity(str, Enum):
    """Determines how start of speech is detected."""

    UNSPECIFIED = "START_SENSITIVITY_UNSPECIFIED"  # Default is HIGH
    HIGH = "START_SENSITIVITY_HIGH"  # Detect start of speech more often
    LOW = "START_SENSITIVITY_LOW"  # Detect start of speech less often


class EndSensitivity(str, Enum):
    """Determines how end of speech is detected."""

    UNSPECIFIED = "END_SENSITIVITY_UNSPECIFIED"  # Default is HIGH
    HIGH = "END_SENSITIVITY_HIGH"  # End speech more often
    LOW = "END_SENSITIVITY_LOW"  # End speech less often


class AutomaticActivityDetection(BaseModel):
    """Configures automatic detection of activity."""

    disabled: Optional[bool] = None
    start_of_speech_sensitivity: Optional[StartSensitivity] = None
    prefix_padding_ms: Optional[int] = None
    end_of_speech_sensitivity: Optional[EndSensitivity] = None
    silence_duration_ms: Optional[int] = None


class RealtimeInputConfig(BaseModel):
    """Configures the realtime input behavior."""

    automatic_activity_detection: Optional[AutomaticActivityDetection] = None


class RealtimeInput(BaseModel):
    mediaChunks: List[MediaChunk]


class ClientContent(BaseModel):
    turns: Optional[List[Turn]] = None
    turnComplete: bool = False


class AudioInputMessage(BaseModel):
    realtimeInput: RealtimeInput

    @classmethod
    def from_raw_audio(cls, raw_audio: bytes, sample_rate: int) -> "AudioInputMessage":
        data = base64.b64encode(raw_audio).decode("utf-8")
        return cls(
            realtimeInput=RealtimeInput(
                mediaChunks=[MediaChunk(mimeType=f"audio/pcm;rate={sample_rate}", data=data)]
            )
        )


class VideoInputMessage(BaseModel):
    realtimeInput: RealtimeInput

    @classmethod
    def from_image_frame(cls, frame: ImageRawFrame) -> "VideoInputMessage":
        buffer = io.BytesIO()
        Image.frombytes(frame.format, frame.size, frame.image).save(buffer, format="JPEG")
        data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return cls(
            realtimeInput=RealtimeInput(mediaChunks=[MediaChunk(mimeType=f"image/jpeg", data=data)])
        )


class ClientContentMessage(BaseModel):
    clientContent: ClientContent


class SystemInstruction(BaseModel):
    parts: List[ContentPart]


class AudioTranscriptionConfig(BaseModel):
    pass


class Setup(BaseModel):
    model: str
    system_instruction: Optional[SystemInstruction] = None
    tools: Optional[List[dict]] = None
    generation_config: Optional[dict] = None
    output_audio_transcription: Optional[AudioTranscriptionConfig] = None
    realtime_input_config: Optional[RealtimeInputConfig] = None


class Config(BaseModel):
    setup: Setup


#
# Server events
#


class SetupComplete(BaseModel):
    pass


class InlineData(BaseModel):
    mimeType: str
    data: str


class Part(BaseModel):
    inlineData: Optional[InlineData] = None
    text: Optional[str] = None


class ModelTurn(BaseModel):
    parts: List[Part]


class ServerContentInterrupted(BaseModel):
    interrupted: bool


class ServerContentTurnComplete(BaseModel):
    turnComplete: bool


class BidiGenerateContentTranscription(BaseModel):
    text: str


class ServerContent(BaseModel):
    modelTurn: Optional[ModelTurn] = None
    interrupted: Optional[bool] = None
    turnComplete: Optional[bool] = None
    outputTranscription: Optional[BidiGenerateContentTranscription] = None


class FunctionCall(BaseModel):
    id: str
    name: str
    args: dict


class ToolCall(BaseModel):
    functionCalls: List[FunctionCall]


class ServerEvent(BaseModel):
    setupComplete: Optional[SetupComplete] = None
    serverContent: Optional[ServerContent] = None
    toolCall: Optional[ToolCall] = None


def parse_server_event(str):
    try:
        evt = json.loads(str)
        return ServerEvent.model_validate(evt)
    except Exception as e:
        print(f"Error parsing server event: {e}")
        return None
