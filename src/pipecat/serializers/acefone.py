#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Acefone Media Streams serializer for Pipecat."""

import base64
import json
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.audio.utils import (
    create_stream_resampler,
    pcm_to_ulaw,
    ulaw_to_pcm,
)
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InterruptionFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer


class AcefoneFrameSerializer(FrameSerializer):
    """Serializer for Acefone or Smartflo Media Streams WebSocket protocol.

    This serializer handles converting between Pipecat frames and Acefone's WebSocket
    media streams protocol. It supports audio conversion, DTMF events, and automatic
    call termination.

    Note: Ref docs for events:
        https://docs.acefone.in/docs/bi-directional-audio-streaming-integration-document
        https://docs.smartflo.tatatelebusiness.com/docs/bi-directional-audio-streaming-integration-document
    """

    class InputParams(FrameSerializer.InputParams):
        """Configuration parameters for AcefoneFrameSerializer.

        Parameters:
            acefone_sample_rate: Sample rate used by Acefone, defaults to 8000 Hz.
            sample_rate: Optional override for pipeline input sample rate.
            auto_hang_up: Whether to automatically terminate call on EndFrame.
            ignore_rtvi_messages: Inherited from base FrameSerializer, defaults to True.
        """

        acefone_sample_rate: int = 8000
        sample_rate: int = 0
        stream_sid: Optional[str] = None
        call_sid: Optional[str] = None
        auto_hang_up: bool = False

    def __init__(self, params: InputParams = InputParams()):
        """Initialize the AcefoneFrameSerializer.

        Args:
            stream_sid: The Acefone or Smartflo Media Stream SID.
            call_sid: The associated Acefone or Smartflo Call SID (optional, but required for auto hang-up).
            params: Configuration parameters.
        """
        self._params = params or AcefoneFrameSerializer.InputParams()
        self._stream_sid = self._params.stream_sid
        self._call_sid = self._params.call_sid
        self._auto_hang_up = self._params.auto_hang_up
        self.outbound_chunk = 1

        self._acefone_sample_rate = self._params.acefone_sample_rate
        self._sample_rate = params.sample_rate

        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()
        self._hangup_attempted = False

    async def setup(self, frame: StartFrame):
        """Sets up the serializer with pipeline configuration.

        Args:
            frame: The StartFrame containing pipeline configuration.
        """
        self._sample_rate = self._params.sample_rate or frame.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serializes a Pipecat frame to Acefone WebSocket format.

        Handles conversion of various frame types to Acefone WebSocket messages.

        Args:
            frame: The Pipecat frame to serialize.

        Returns:
            Serialized data as string or bytes, or None if the frame isn't handled.
        """
        if (
            self._params.auto_hang_up
            and not self._hangup_attempted
            and isinstance(frame, (EndFrame, CancelFrame))
        ):
            ## Hangup the call using the websocket end event
            self._hangup_attempted = True
            answer = {"event": "end", "streamSid": self._stream_sid, "call_sid": self._call_sid}
            return json.dumps(answer)
        if isinstance(frame, InterruptionFrame):
            answer = {"event": "clear", "streamSid": self._stream_sid}
            return json.dumps(answer)
        elif isinstance(frame, AudioRawFrame):
            data = frame.audio

            # Output: Convert PCM at frame's rate to 8kHz μ-law for Acefone
            serialized_data = await pcm_to_ulaw(
                data, frame.sample_rate, self._acefone_sample_rate, self._output_resampler
            )

            if len(serialized_data) == 0:
                # Ignoring in case we don't have audio
                return None

            payload = base64.b64encode(serialized_data).decode("utf-8")
            answer: dict[str, Any] = {
                "event": "media",
                "streamSid": self._stream_sid,
                "media": {"payload": payload},
            }

            return json.dumps(answer)
        elif isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame):
                return None
            return json.dumps(frame.message)

        # Return None for unhandled frames
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserializes Acefone WebSocket data to Pipecat frames.

        Handles conversion of Acefone media events to appropriate Pipecat frames.

        Args:
            data: The raw WebSocket data from Acefone.

        Returns:
            A Pipecat frame corresponding to the Acefone event, or None if unhandled.
        """
        message = json.loads(data)

        if message["event"] == "media":
            payload_base64 = message["media"]["payload"]
            payload = base64.b64decode(payload_base64)

            # Input: Convert Acefone's 8kHz μ-law to PCM at pipeline input rate
            deserialized_data = await ulaw_to_pcm(
                payload, self._acefone_sample_rate, self._sample_rate, self._input_resampler
            )
            if len(deserialized_data) == 0:
                # Ignoring in case we don't have audio
                return None

            audio_frame = InputAudioRawFrame(
                audio=deserialized_data, num_channels=1, sample_rate=self._sample_rate
            )
            return audio_frame
        elif message["event"] == "dtmf":
            digit = message.get("dtmf", {}).get("digit")

            try:
                return InputDTMFFrame(KeypadEntry(digit))
            except ValueError:
                # Handle case where string doesn't match any enum value
                logger.info(f"Invalid DTMF digit: {digit}")
                return None
        elif message["event"] == "start" and not self._stream_sid:
            self._stream_sid = message["streamSid"]

        return None
