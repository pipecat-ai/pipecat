#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json

from pydantic import BaseModel

from pipecat.frames.frames import AudioRawFrame, Frame, StartInterruptionFrame, MarkFrame
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.utils.audio import change_audio_frame_rate, ulaw_to_pcm, pcm_to_ulaw
from loguru import logger


class TwilioFrameSerializer(FrameSerializer):
    class InputParams(BaseModel):
        twilio_sample_rate: int = 8000
        sample_rate: int = 16000

    SERIALIZABLE_TYPES = {
        AudioRawFrame: "audio",
    }

    def __init__(self, stream_sid: str, params: InputParams = InputParams()):
        self._stream_sid = stream_sid
        self._params = params

    def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, AudioRawFrame):
            data = frame.audio
            if frame.encoding == "mulaw":
                if frame.sample_rate != self._params.twilio_sample_rate:
                    serialized_data = change_audio_frame_rate(
                        data, frame.sample_rate, self._params.twilio_sample_rate
                    )
                else:
                    serialized_data = data
            elif frame.encoding == "pcm":
                serialized_data = pcm_to_ulaw(
                    data, frame.sample_rate, self._params.twilio_sample_rate)
            else:
                raise ValueError(f"Unsupported encoding: {frame.encoding}")

            payload = base64.b64encode(serialized_data).decode("utf-8")
            answer = {
                "event": "media",
                "streamSid": self._stream_sid,
                "media": {
                    "payload": payload
                }
            }

            return json.dumps(answer)

        if isinstance(frame, StartInterruptionFrame):
            answer = {"event": "clear", "streamSid": self._stream_sid}
            return json.dumps(answer)

        if isinstance(frame, MarkFrame) and frame.type == "request":
            answer = {
                "event": "mark",
                "streamSid": self._stream_sid,
                "mark": {"name": frame.passed_name},
                # "sequenceNumber": frame.seq_number,
            }
            logger.info(f"Sending the mark frame with data: {answer}")
            return json.dumps(answer)

    def deserialize(self, data: str | bytes) -> Frame | None:
        message = json.loads(data)

        if message["event"] == "mark":
            mark_frame = MarkFrame(
                seq_number=message["sequenceNumber"],
                type="response",
                passed_name=message["mark"]["name"],
            )
            return mark_frame

        elif message["event"] == "media":
            payload_base64 = message["media"]["payload"]
            payload = base64.b64decode(payload_base64)

            deserialized_data = ulaw_to_pcm(
                payload,
                self._params.twilio_sample_rate,
                self._params.sample_rate)
            audio_frame = AudioRawFrame(
                audio=deserialized_data,
                num_channels=1,
                sample_rate=self._params.sample_rate,
                encoding="pcm"
            )
            return audio_frame
        else:
            return None



