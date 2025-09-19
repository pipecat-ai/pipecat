#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Jambonz WebSocket protocol serializer for Pipecat."""

import json
from typing import Optional
from aiohttp import web, WSMsgType
from pydantic import BaseModel
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    KeypadEntry,
    StartFrame,
    StartInterruptionFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType


class JambonzFrameSerializer(FrameSerializer):
    """Serializer for Jambonz WebSocket protocol.

    This serializer handles converting between Pipecat frames and Jambonz's WebSocket
    protocol. It supports audio conversion, DTMF events, and automatic
    call termination. It only supports OpenAI and Elevenlabs TTS at the moment.
    """

    class InputParams(BaseModel):
        """Configuration parameters for JambonzFrameSerializer.

        Parameters:
            audio_in_sample_rate: Optional override for the sample rate of the audio we get from Jambonz.
            audio_out_sample_rate: Optional override for the sample rate of the audio we send to Jambonz.
            stt_sample_rate: Optional override for the sample rate of the audio we send to the STT service.
            auto_hang_up: Whether to automatically terminate call on EndFrame.
        """

        audio_in_sample_rate: Optional[int] = None
        audio_out_sample_rate: Optional[int] = None
        stt_sample_rate: Optional[int] = None
        auto_hang_up: bool = True

    def __init__(
        self,
        params: InputParams,
    ):
        """Initialize the JambonzFrameSerializer.

        Args:
            params: Configuration parameters.
        """
        self._params = params
        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()
        self._hangup_attempted = False

    @property
    def type(self) -> FrameSerializerType:
        """Gets the serializer type.

        Returns:
            The serializer type, either TEXT or BINARY.
        """
        return FrameSerializerType.BINARY

    async def setup(self, frame: StartFrame):
        """Sets up the serializer with pipeline configuration.

        Args:
            frame: The StartFrame containing pipeline configuration.
        """
        pass

    async def serialize(
        self, frame: Frame, sample_rate: int = 16000
    ) -> str | bytes | None:
        """Serializes a Pipecat frame to Jambonz WebSocket format.

        Handles conversion of various frame types to Jambonz WebSocket messages.
        For EndFrames, initiates call termination if auto_hang_up is enabled.

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
            return {"type": "disconnect"}
        # stop the previous playAudio from running
        elif isinstance(frame, StartInterruptionFrame):
            return {"type": "killAudio"}
        elif isinstance(frame, AudioRawFrame):
            data = frame.audio
            serialized_data = await self._output_resampler.resample(
                data, frame.sample_rate, self._params.audio_out_sample_rate
            )
            if serialized_data is None or len(serialized_data) == 0:
                return None
            return serialized_data
        elif isinstance(frame, (TransportMessageFrame, TransportMessageUrgentFrame)):
            return json.dumps(frame.message)

        # Return None for unhandled frames
        return None

    async def deserialize(self, msg: web.WebSocketResponse) -> Frame | None:
        """Deserializes Jambonz WebSocket data to Pipecat frames.

        Handles conversion of Jambonz media events to appropriate Pipecat frames.
        Audio comes as binary frames, DTMF and other events come as JSON text frames.

        Args:
            data: The raw WebSocket data from Jambonz.

        Returns:
            A Pipecat frame corresponding to the Jambonz event, or None if unhandled.
        """
        # Handle binary audio frames
        if msg.type == WSMsgType.BINARY:
            # Audio comes as raw binary PCM data
            # Input: Resample from Jambonz sample rate to pipeline input rate
            deserialized_data = await self._input_resampler.resample(
                msg.data,
                self._params.audio_in_sample_rate,
                self._params.stt_sample_rate or self._params.audio_in_sample_rate,
            )
            if deserialized_data is None or len(deserialized_data) == 0:
                # Ignoring in case we don't have audio
                return None

            audio_frame = InputAudioRawFrame(
                audio=deserialized_data,
                num_channels=1,
                sample_rate=self._params.stt_sample_rate
                or self._params.audio_in_sample_rate,
            )
            return audio_frame

        # Handle JSON text frames (DTMF, commands, etc.)
        elif msg.type == WSMsgType.TEXT:
            message = json.loads(msg.data)
            if message["event"] == "dtmf":
                digit = message["dtmf"]
                return InputDTMFFrame(KeypadEntry(digit))
            # Handle other JSON events if needed (initial metadata, commands, etc.)
            else:
                return None
        return None
