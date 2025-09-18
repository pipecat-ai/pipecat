#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json
from pipecat.frames.frames import (
    AudioRawFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    StartFrame,
    TransportMessageFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType
from typing import Optional

class ACSFrameSerializer(FrameSerializer):
    """Serializer for Azure Communication Services (ACS) Media Streams WebSocket protocol.

    Handles conversion between Pipecat frames and ACS WebSocket audio protocol.
    Supports bi-directional audio (PCM 16kHz mono, base64-encoded) and control messages.
    """

    def __init__(
        self,
        sample_rate: Optional[int] = 16000,
        channels: Optional[int] = 1,
    ):
        self.sample_rate: Optional[int] = sample_rate
        self.channels: Optional[int] = channels

    @property
    def type(self) -> FrameSerializerType:
        return FrameSerializerType.TEXT

    async def setup(self, frame: StartFrame):
        if hasattr(frame, "audio_in_sample_rate") and frame.audio_in_sample_rate:
            self.sample_rate = frame.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serializes a Pipecat frame to ACS WebSocket format."""
        if isinstance(frame, AudioRawFrame):
            audio_b64 = base64.b64encode(frame.audio).decode("utf-8")
            message = {
                "Kind": "AudioData",
                "AudioData": {
                    "Data": audio_b64,
                },
            }
            return json.dumps(message)
        elif isinstance(frame, EndFrame):
            return json.dumps({"kind": "StopAudio", "AudioData":None,"StopAudio": {}})
        elif isinstance(frame, TransportMessageFrame):
            return json.dumps({"kind": "Control", **frame.message})
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserializes ACS WebSocket data to Pipecat frames."""
        message = json.loads(data)
        
        kind = message.get("kind")
        if kind == "AudioData":
            audio_b64 = message.get("audioData", {}).get("data")
            if audio_b64:
                audio_data = base64.b64decode(audio_b64)
                return InputAudioRawFrame(
                    audio=audio_data,
                    num_channels=self.channels or 1,
                    sample_rate=self.sample_rate or 16000,
                )
        elif kind == "AudioMetadata":
            meta = message.get("audioMetadata", {})
            self.sample_rate = meta.get("sampleRate", self.sample_rate)
            self.channels = meta.get("channels", self.channels)
            return None
        elif kind == "StopAudio":
            return EndFrame()
        elif kind == "Control":
            return TransportMessageFrame(message=message)
        return None
