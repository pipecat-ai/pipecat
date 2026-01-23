#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Vonage Audio Connector WebSocket serializer for Pipecat."""

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
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer


class VonageFrameSerializer(FrameSerializer):
    """Serializer for Vonage Video API Audio Connector WebSocket protocol.

    This serializer converts between Pipecat frames and the Vonage Audio Connector
    WebSocket streaming protocol.

    Note:
        Ref docs: https://developer.vonage.com/en/video/guides/audio-connector
    """

    class InputParams(BaseModel):
        """Configuration parameters for VonageFrameSerializer.

        Parameters:
            vonage_sample_rate: Sample rate used by Vonage, defaults to 16000 Hz.
                Common values: 8000, 16000, 24000 Hz.
            sample_rate: Optional override for pipeline input sample rate.
        """

        vonage_sample_rate: int = 16000
        sample_rate: Optional[int] = None

    def __init__(self, call_uuid: str, application_id: Optional[str] = None, private_key: Optional[str] = None, params: Optional[InputParams] = None):
        """Initialize the VonageFrameSerializer.

        Args:
            call_uuid: The Vonage Call UUID.
            application_id: Vonage application ID (required for call control).
            private_key: Private key for JWT generation (required for call control).
            params: Configuration parameters.
        """
        self._call_uuid = call_uuid
        self._application_id = application_id
        self._private_key = private_key
        self._params = params or VonageFrameSerializer.InputParams()

        self._vonage_sample_rate = self._params.vonage_sample_rate
        self._sample_rate = 0  # Pipeline input rate

        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()

    async def setup(self, frame: StartFrame):
        """Sets up the serializer with pipeline configuration.

        Args:
            frame: The StartFrame containing pipeline configuration.
        """
        self._sample_rate = self._params.sample_rate or frame.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serializes a Pipecat frame to Vonage WebSocket format.

        Handles conversion of various frame types to Vonage WebSocket messages.

        Args:
            frame: The Pipecat frame to serialize.

        Returns:
            Serialized data as string (JSON commands) or bytes (audio), or None if the frame isn't handled.
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
            # Clear the audio buffer to stop playback immediately
            answer = {"action": "clear"}
            return json.dumps(answer)
        elif isinstance(frame, AudioRawFrame):
            data = frame.audio

            # Output: Convert PCM at frame's rate to Vonage's sample rate (16-bit linear PCM)
            serialized_data = await self._output_resampler.resample(
                data, frame.sample_rate, self._vonage_sample_rate
            )
            if serialized_data is None or len(serialized_data) == 0:
                # Ignoring in case we don't have audio
                return None

            # Vonage expects raw binary PCM data (not base64 encoded)
            return serialized_data
        elif isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            # Allow sending custom JSON commands (e.g., notify)
            return json.dumps(frame.message)

        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserializes Vonage WebSocket data to Pipecat frames.

        Handles conversion of Vonage events to appropriate Pipecat frames.
        - Binary messages contain audio data (16-bit linear PCM)
        - Text messages contain JSON events (websocket:connected, websocket:cleared, dtmf, etc.)

        Args:
            data: The raw WebSocket data from Vonage.

        Returns:
            A Pipecat frame corresponding to the Vonage event, or None if unhandled.
        """
        # Check if this is binary audio data
        if isinstance(data, bytes):
            # Binary message = audio data (16-bit linear PCM)
            payload = data

            # Input: Convert Vonage's PCM audio to pipeline sample rate
            deserialized_data = await self._input_resampler.resample(
                payload,
                self._vonage_sample_rate,
                self._sample_rate,
            )
            if deserialized_data is None or len(deserialized_data) == 0:
                # Ignoring in case we don't have audio
                return None

            audio_frame = InputAudioRawFrame(
                audio=deserialized_data,
                num_channels=1,  # Vonage uses mono audio
                sample_rate=self._sample_rate,  # Use the configured pipeline input rate
            )
            return audio_frame
        else:
            # Text message = JSON event
            try:
                message = json.loads(data)
                event = message.get("event")

                # Handle different event types
                if event == "websocket:connected":
                    logger.debug(
                        f"Vonage WebSocket connected: content-type={message.get('content-type')}"
                    )
                    return None
                elif event == "websocket:cleared":
                    logger.debug("Vonage audio buffer cleared")
                    return None
                elif event == "websocket:notify":
                    logger.debug(f"Vonage notify event: {message.get('payload')}")
                    return None
                elif event == "websocket:dtmf":
                    # Handle DTMF input
                    # Vonage may send digit in different formats, try both
                    digit = message.get("digit") or message.get("dtmf", {}).get("digit")
                    if digit is None:
                        logger.warning(f"DTMF event received but no digit found: {message}")
                        return None

                    digit = str(digit)
                    logger.debug(f"Received DTMF digit: {digit}")
                    try:
                        return InputDTMFFrame(KeypadEntry(digit))
                    except ValueError:
                        logger.warning(f"Invalid DTMF digit received: {digit}")
                        return None
                else:
                    logger.debug(f"Vonage event: {event}")
                    return None

            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON message from Vonage: {data}")
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