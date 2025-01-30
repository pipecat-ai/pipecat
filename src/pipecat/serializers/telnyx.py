#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json

from pydantic import BaseModel

from pipecat.audio.utils import alaw_to_pcm, pcm_to_alaw, pcm_to_ulaw, ulaw_to_pcm
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    KeypadEntry,
    StartInterruptionFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType


class TelnyxFrameSerializer(FrameSerializer):
    class InputParams(BaseModel):
        telnyx_sample_rate: int = 8000
        sample_rate: int = 16000
        inbound_encoding: str = "PCMU"
        outbound_encoding: str = "PCMU"

    def __init__(
        self,
        stream_id: str,
        outbound_encoding: str,
        inbound_encoding: str,
        params: InputParams = InputParams(),
    ):
        self._stream_id = stream_id
        params.outbound_encoding = outbound_encoding
        params.inbound_encoding = inbound_encoding
        self._params = params

    @property
    def type(self) -> FrameSerializerType:
        return FrameSerializerType.TEXT

    def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, AudioRawFrame):
            data = frame.audio

            if self._params.inbound_encoding == "PCMU":
                serialized_data = pcm_to_ulaw(
                    data, frame.sample_rate, self._params.telnyx_sample_rate
                )
            elif self._params.inbound_encoding == "PCMA":
                serialized_data = pcm_to_alaw(
                    data, frame.sample_rate, self._params.telnyx_sample_rate
                )
            else:
                raise ValueError(f"Unsupported encoding: {self._params.inbound_encoding}")

            payload = base64.b64encode(serialized_data).decode("utf-8")
            answer = {
                "event": "media",
                "media": {"payload": payload},
            }

            return json.dumps(answer)

        if isinstance(frame, StartInterruptionFrame):
            answer = {"event": "clear"}
            return json.dumps(answer)

    def deserialize(self, data: str | bytes) -> Frame | None:
        message = json.loads(data)

        if message["event"] == "media":
            payload_base64 = message["media"]["payload"]
            payload = base64.b64decode(payload_base64)

            if self._params.outbound_encoding == "PCMU":
                deserialized_data = ulaw_to_pcm(
                    payload, self._params.telnyx_sample_rate, self._params.sample_rate
                )
            elif self._params.outbound_encoding == "PCMA":
                deserialized_data = alaw_to_pcm(
                    payload, self._params.telnyx_sample_rate, self._params.sample_rate
                )
            else:
                raise ValueError(f"Unsupported encoding: {self._params.outbound_encoding}")

            audio_frame = InputAudioRawFrame(
                audio=deserialized_data, num_channels=1, sample_rate=self._params.sample_rate
            )
            return audio_frame
        elif message["event"] == "dtmf":
            digit = message.get("dtmf", {}).get("digit")

            try:
                return InputDTMFFrame(KeypadEntry(digit))
            except ValueError as e:
                # Handle case where string doesn't match any enum value
                return None
        else:
            return None
