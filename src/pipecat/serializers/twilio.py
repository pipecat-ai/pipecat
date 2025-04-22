#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import base64
import json
import os
from typing import Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import create_default_resampler, pcm_to_ulaw, ulaw_to_pcm
from pipecat.frames.frames import (
    AudioRawFrame,
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


class TwilioFrameSerializer(FrameSerializer):
    """Serializer for Twilio Media Streams WebSocket protocol.

    This serializer handles converting between Pipecat frames and Twilio's WebSocket
    media streams protocol. It supports audio conversion, DTMF events, and automatic
    call termination.

    Attributes:
        _stream_sid: The Twilio Media Stream SID.
        _call_sid: The associated Twilio Call SID.
        _params: Configuration parameters.
        _twilio_sample_rate: Sample rate used by Twilio (typically 8kHz).
        _sample_rate: Input sample rate for the pipeline.
        _resampler: Audio resampler for format conversion.
    """

    class InputParams(BaseModel):
        """Configuration parameters for TwilioFrameSerializer.

        Attributes:
            twilio_sample_rate: Sample rate used by Twilio, defaults to 8000 Hz.
            sample_rate: Optional override for pipeline input sample rate.
            auto_hang_up: Whether to automatically terminate call on EndFrame.
        """

        twilio_sample_rate: int = 8000
        sample_rate: Optional[int] = None
        auto_hang_up: bool = True

    def __init__(self, stream_sid: str, call_sid: str, params: InputParams = InputParams()):
        """Initialize the TwilioFrameSerializer.

        Args:
            stream_sid: The Twilio Media Stream SID.
            call_sid: The associated Twilio Call SID.
            params: Configuration parameters.
        """
        self._stream_sid = stream_sid
        self._call_sid = call_sid
        self._params = params

        self._twilio_sample_rate = self._params.twilio_sample_rate
        self._sample_rate = 0  # Pipeline input rate

        self._resampler = create_default_resampler()

    @property
    def type(self) -> FrameSerializerType:
        """Gets the serializer type.

        Returns:
            The serializer type, either TEXT or BINARY.
        """
        return FrameSerializerType.TEXT

    async def setup(self, frame: StartFrame):
        """Sets up the serializer with pipeline configuration.

        Args:
            frame: The StartFrame containing pipeline configuration.
        """
        self._sample_rate = self._params.sample_rate or frame.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serializes a Pipecat frame to Twilio WebSocket format.

        Handles conversion of various frame types to Twilio WebSocket messages.
        For EndFrames, initiates call termination if auto_hang_up is enabled.

        Args:
            frame: The Pipecat frame to serialize.

        Returns:
            Serialized data as string or bytes, or None if the frame isn't handled.
        """
        if self._params.auto_hang_up and isinstance(frame, (EndFrame)):
            asyncio.create_task(self._hang_up_call())
            return None
        elif isinstance(frame, StartInterruptionFrame):
            answer = {"event": "clear", "streamSid": self._stream_sid}
            return json.dumps(answer)
        elif isinstance(frame, AudioRawFrame):
            data = frame.audio

            # Output: Convert PCM at frame's rate to 8kHz μ-law for Twilio
            serialized_data = await pcm_to_ulaw(
                data, frame.sample_rate, self._twilio_sample_rate, self._resampler
            )
            payload = base64.b64encode(serialized_data).decode("utf-8")
            answer = {
                "event": "media",
                "streamSid": self._stream_sid,
                "media": {"payload": payload},
            }

            return json.dumps(answer)
        elif isinstance(frame, (TransportMessageFrame, TransportMessageUrgentFrame)):
            return json.dumps(frame.message)

        # Return None for unhandled frames
        return None

    async def _hang_up_call(self):
        """Hang up the Twilio call using the REST API."""
        try:
            from twilio.rest import Client

            account_sid = os.getenv("TWILIO_ACCOUNT_SID")
            auth_token = os.getenv("TWILIO_AUTH_TOKEN")

            if not account_sid or not auth_token:
                logger.warning(
                    "Missing Twilio credentials (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), cannot hang up call"
                )
                return

            # Create Twilio client and hang up the call using the correct CallSid
            client = Client(account_sid, auth_token)
            client.calls(self._call_sid).update(status="completed")

            logger.info(f"Successfully terminated Twilio call {self._call_sid}")
        except ImportError:
            logger.warning("Twilio Python SDK not installed. Install with: pip install twilio")
        except Exception as e:
            logger.exception(f"Failed to hang up Twilio call: {e}")

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserializes Twilio WebSocket data to Pipecat frames.

        Handles conversion of Twilio media events to appropriate Pipecat frames.

        Args:
            data: The raw WebSocket data from Twilio.

        Returns:
            A Pipecat frame corresponding to the Twilio event, or None if unhandled.
        """
        message = json.loads(data)

        if message["event"] == "media":
            payload_base64 = message["media"]["payload"]
            payload = base64.b64decode(payload_base64)

            # Input: Convert Twilio's 8kHz μ-law to PCM at pipeline input rate
            deserialized_data = await ulaw_to_pcm(
                payload, self._twilio_sample_rate, self._sample_rate, self._resampler
            )
            audio_frame = InputAudioRawFrame(
                audio=deserialized_data, num_channels=1, sample_rate=self._sample_rate
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
