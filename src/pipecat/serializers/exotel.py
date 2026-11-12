#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Exotel Media Streams serializer for Pipecat."""

import base64
import json
from typing import Any

from loguru import logger
from pydantic import model_validator

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InterruptionFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.utils.deprecation import warn_deprecated


class ExotelFrameSerializer(FrameSerializer):
    """Serializer for Exotel Media Streams WebSocket protocol.

    This serializer handles converting between Pipecat frames and Exotel's WebSocket
    media streams protocol. It supports audio conversion and DTMF events.

    Note: Ref docs for events:
        https://support.exotel.com/support/solutions/articles/3000108630-working-with-the-stream-and-voicebot-applet
    """

    class InputParams(FrameSerializer.InputParams):
        """Configuration parameters for ExotelFrameSerializer.

        Parameters:
            audio_in_sample_rate: Sample rate of the audio Exotel sends, defaults to 8000 Hz.
            audio_out_sample_rate: Sample rate of the audio sent to Exotel, defaults to 8000 Hz.
            exotel_sample_rate: Sample rate used by Exotel.

                .. deprecated:: 1.16.0
                    Use ``audio_in_sample_rate`` and ``audio_out_sample_rate`` instead.
                    Will be removed in 2.0.0.

            sample_rate: Has no effect: incoming audio is converted to the pipeline's input rate.

                .. deprecated:: 1.16.0
                    Use :attr:`~pipecat.pipeline.worker.PipelineParams.audio_in_sample_rate`
                    instead. Will be removed in 2.0.0.

            ignore_rtvi_messages: Inherited from base FrameSerializer, defaults to True.
        """

        audio_in_sample_rate: int = 8000
        audio_out_sample_rate: int = 8000
        exotel_sample_rate: int | None = None
        sample_rate: int | None = None

        @model_validator(mode="before")
        @classmethod
        def _carry_deprecated_rates(cls, data: Any) -> Any:
            if not isinstance(data, dict):
                return data
            if data.get("exotel_sample_rate") is not None:
                warn_deprecated(
                    "`ExotelFrameSerializer.InputParams.exotel_sample_rate` is deprecated since "
                    "1.16.0 and will be removed in 2.0.0. Use `audio_in_sample_rate` and "
                    "`audio_out_sample_rate` instead.",
                    stacklevel=4,
                )
                data.setdefault("audio_in_sample_rate", data["exotel_sample_rate"])
                data.setdefault("audio_out_sample_rate", data["exotel_sample_rate"])
            if data.get("sample_rate") is not None:
                warn_deprecated(
                    "`ExotelFrameSerializer.InputParams.sample_rate` is deprecated since 1.16.0 "
                    "and will be removed in 2.0.0. Use `PipelineParams.audio_in_sample_rate` "
                    "instead.",
                    stacklevel=4,
                )
            return data

    def __init__(
        self, stream_sid: str, call_sid: str | None = None, params: InputParams | None = None
    ):
        """Initialize the ExotelFrameSerializer.

        Args:
            stream_sid: The Exotel Media Stream SID.
            call_sid: The associated Exotel Call SID (optional, not used in this implementation).
            params: Configuration parameters.
        """
        params = params or ExotelFrameSerializer.InputParams()
        super().__init__(params)
        self._params: ExotelFrameSerializer.InputParams = params

        self._stream_sid = stream_sid
        self._call_sid = call_sid

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serializes a Pipecat frame to Exotel WebSocket format.

        Handles conversion of various frame types to Exotel WebSocket messages.

        Args:
            frame: The Pipecat frame to serialize.

        Returns:
            Serialized data as string or bytes, or None if the frame isn't handled.
        """
        if isinstance(frame, InterruptionFrame):
            await self._reset_audio_out()
            answer = {"event": "clear", "stream_sid": self._stream_sid}
            return json.dumps(answer)
        elif isinstance(frame, AudioRawFrame):
            data = await self._resample_audio_out(frame)
            if not data:
                return None

            payload = base64.b64encode(data).decode("ascii")

            answer = {
                "event": "media",
                "stream_sid": self._stream_sid,
                "media": {"payload": payload},
            }

            return json.dumps(answer)
        elif isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame):
                return None
            return json.dumps(frame.message)

        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserializes Exotel WebSocket data to Pipecat frames.

        Handles conversion of Exotel media events to appropriate Pipecat frames.

        Args:
            data: The raw WebSocket data from Exotel.

        Returns:
            A Pipecat frame corresponding to the Exotel event, or None if unhandled.
        """
        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON message: {data}")
            return None

        if message.get("event") == "media":
            payload_base64 = message.get("media", {}).get("payload")
            if not payload_base64:
                return None

            audio = await self._resample_audio_in(base64.b64decode(payload_base64))
            if not audio:
                return None

            return InputAudioRawFrame(
                audio=audio,
                num_channels=1,  # Assuming mono audio from Exotel
                sample_rate=self._sample_rate,
            )
        elif message.get("event") == "dtmf":
            digit = message.get("dtmf", {}).get("digit")

            try:
                return InputDTMFFrame(KeypadEntry(digit))
            except ValueError:
                # Handle case where string doesn't match any enum value
                logger.info(f"Invalid DTMF digit: {digit}")
                return None

        return None
