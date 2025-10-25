#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Vonage WebSocket protocol serializer for Pipecat."""

import json
from typing import Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InterruptionFrame,
    StartFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType


class VonageFrameSerializer(FrameSerializer):
    """Serializer for Vonage WebSocket protocol.

    This serializer handles converting between Pipecat frames and Vonage's WebSocket
    protocol. Unlike Twilio which uses base64-encoded μ-law in JSON messages,
    Vonage uses raw PCM audio bytes directly.

    Key differences from Twilio:
    - Raw binary PCM audio instead of base64-encoded μ-law
    - 16kHz sample rate by default instead of 8kHz
    - Direct byte streaming instead of JSON-wrapped messages
    - NCCO control instead of TwiML
    """

    class InputParams(BaseModel):
        """Configuration parameters for VonageFrameSerializer.

        Parameters:
            vonage_sample_rate: Sample rate used by Vonage, defaults to 16000 Hz.
            sample_rate: Optional override for pipeline input sample rate.
            auto_hang_up: Whether to automatically terminate call on EndFrame.
        """

        vonage_sample_rate: int = 16000  # Vonage default is 16kHz
        sample_rate: Optional[int] = None
        auto_hang_up: bool = True

    def __init__(
        self,
        call_uuid: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        application_id: Optional[str] = None,
        private_key: Optional[str] = None,
        params: Optional[InputParams] = None,
    ):
        """Initialize the VonageFrameSerializer.

        Args:
            call_uuid: The Vonage Call UUID.
            api_key: Vonage API key (optional, for some operations).
            api_secret: Vonage API secret (optional, for webhook verification).
            application_id: Vonage application ID (required for call control).
            private_key: Private key for JWT generation (required for call control).
            params: Configuration parameters.
        """
        self._call_uuid = call_uuid
        self._api_key = api_key
        self._api_secret = api_secret
        self._application_id = application_id
        self._private_key = private_key
        self._params = params or VonageFrameSerializer.InputParams()

        self._vonage_sample_rate = self._params.vonage_sample_rate
        self._sample_rate = 0  # Pipeline input rate

        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()
        self._hangup_attempted = False

    @property
    def type(self) -> FrameSerializerType:
        """Gets the serializer type.

        Returns:
            The serializer type - BINARY for Vonage (not TEXT like Twilio).
        """
        return FrameSerializerType.BINARY

    async def setup(self, frame: StartFrame):
        """Sets up the serializer with pipeline configuration.

        Args:
            frame: The StartFrame containing pipeline configuration.
        """
        self._sample_rate = self._params.sample_rate or frame.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> bytes | str | None:
        """Serializes a Pipecat frame to Vonage WebSocket format.

        For audio frames, returns raw PCM bytes directly.
        For control frames, may return JSON messages or None.

        Args:
            frame: The Pipecat frame to serialize.

        Returns:
            Serialized data as bytes (audio) or string (control), or None.
        """
        if (
            self._params.auto_hang_up
            and not self._hangup_attempted
            and isinstance(frame, (EndFrame, CancelFrame))
        ):
            self._hangup_attempted = True
            await self._hang_up_call()
            return None
        elif isinstance(frame, InterruptionFrame):
            return None
        elif isinstance(frame, AudioRawFrame):
            data = frame.audio

            if frame.sample_rate != self._vonage_sample_rate:
                logger.error(
                    f"Sample rate mismatch: frame={frame.sample_rate}Hz, vonage={self._vonage_sample_rate}Hz. "
                    f"Check audio_config setup. Expected {self._vonage_sample_rate}Hz throughout."
                )

            return data
        elif isinstance(frame, (TransportMessageFrame, TransportMessageUrgentFrame)):
            return json.dumps(frame.message)

        return None

    async def _hang_up_call(self):
        """Hang up the Vonage call using Vonage's REST API."""
        try:
            import aiohttp
            import jwt
            import time

            if not self._call_uuid or not self._application_id or not self._private_key:
                missing = []
                if not self._call_uuid:
                    missing.append("call_uuid")
                if not self._application_id:
                    missing.append("application_id")
                if not self._private_key:
                    missing.append("private_key")

                logger.warning(
                    f"Cannot hang up Vonage call: missing required parameters: {', '.join(missing)}"
                )
                return

            claims = {
                "application_id": self._application_id,
                "iat": int(time.time()),
                "exp": int(time.time()) + 3600,
                "jti": str(time.time())
            }
            token = jwt.encode(claims, self._private_key, algorithm="RS256")

            endpoint = f"https://api.nexmo.com/v1/calls/{self._call_uuid}"

            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }

            data = {"action": "hangup"}

            async with aiohttp.ClientSession() as session:
                async with session.put(endpoint, headers=headers, json=data) as response:
                    if response.status == 200:
                        logger.info(f"Successfully terminated Vonage call {self._call_uuid}")
                    elif response.status == 404:
                        logger.debug(f"Vonage call {self._call_uuid} was already terminated")
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"Failed to terminate Vonage call {self._call_uuid}: "
                            f"Status {response.status}, Response: {error_text}"
                        )

        except Exception as e:
            logger.exception(f"Failed to hang up Vonage call: {e}")

    async def deserialize(self, data: bytes | str) -> Frame | None:
        """Deserializes Vonage WebSocket data to Pipecat frames.

        Handles raw PCM audio bytes and optional JSON control messages.

        Args:
            data: The raw WebSocket data from Vonage.

        Returns:
            A Pipecat frame corresponding to the data, or None if unhandled.
        """
        if isinstance(data, bytes):
            if self._sample_rate != self._vonage_sample_rate:
                logger.error(
                    f"Sample rate mismatch on input: vonage={self._vonage_sample_rate}Hz, pipeline={self._sample_rate}Hz. "
                    f"Check audio_config setup. Expected {self._vonage_sample_rate}Hz throughout."
                )
            
            audio_data = data

            audio_frame = InputAudioRawFrame(
                audio=audio_data, 
                num_channels=1, 
                sample_rate=self._sample_rate
            )
            return audio_frame
        elif isinstance(data, str):
            try:
                message = json.loads(data)
                
                if message.get("type") == "dtmf":
                    digit = message.get("digit")
                    if digit:
                        try:
                            return InputDTMFFrame(KeypadEntry(digit))
                        except ValueError:
                            return None
                
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON message from Vonage: {data}")
                
        return None