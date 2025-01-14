#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
#

import base64
import io
import json
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


class RealtimeInput(BaseModel):
    mediaChunks: List[MediaChunk]


class ClientContent(BaseModel):
    turns: Optional[List[Turn]] = None
    turnComplete: bool = False


class AudioInputMessage(BaseModel):
    realtimeInput: RealtimeInput

    @classmethod
    def from_raw_audio(cls, raw_audio: bytes, sample_rate=16000) -> "AudioInputMessage":
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


class Setup(BaseModel):
    model: str
    system_instruction: Optional[SystemInstruction] = None
    tools: Optional[List[dict]] = None
    generation_config: Optional[dict] = None


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


class ServerContent(BaseModel):
    modelTurn: Optional[ModelTurn] = None
    interrupted: Optional[bool] = None
    turnComplete: Optional[bool] = None


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
