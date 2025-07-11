#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Plivo WebSocket frame serializer for audio streaming."""

import base64
import json
from typing import Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import create_stream_resampler, pcm_to_ulaw, ulaw_to_pcm
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


class PlivoFrameSerializer(FrameSerializer):
    """Serializer for Plivo Audio Streaming WebSocket protocol.

    This serializer handles converting between Pipecat frames and Plivo's WebSocket
    audio streaming protocol. It supports audio conversion, DTMF events, and automatic
    call termination.

    When auto_hang_up is enabled (default), the serializer will automatically terminate
    the Plivo call when an EndFrame or CancelFrame is processed, but requires Plivo
    credentials to be provided.
    """

    class InputParams(BaseModel):
        """Configuration parameters for PlivoFrameSerializer.

        Parameters:
            plivo_sample_rate: Sample rate used by Plivo, defaults to 8000 Hz.
            sample_rate: Optional override for pipeline input sample rate.
            auto_hang_up: Whether to automatically terminate call on EndFrame.
        """

        plivo_sample_rate: int = 8000
        sample_rate: Optional[int] = None
        auto_hang_up: bool = True

    def __init__(
        self,
        stream_id: str,
        call_id: Optional[str] = None,
        auth_id: Optional[str] = None,
        auth_token: Optional[str] = None,
        params: Optional[InputParams] = None,
    ):
        """Initialize the PlivoFrameSerializer.

        Args:
            stream_id: The Plivo Stream ID.
            call_id: The associated Plivo Call ID (optional, but required for auto hang-up).
            auth_id: Plivo auth ID (required for auto hang-up).
            auth_token: Plivo auth token (required for auto hang-up).
            params: Configuration parameters.
        """
        self._stream_id = stream_id
        self._call_id = call_id
        self._auth_id = auth_id
        self._auth_token = auth_token
        self._params = params or PlivoFrameSerializer.InputParams()

        self._plivo_sample_rate = self._params.plivo_sample_rate
        self._sample_rate = 0  # Pipeline input rate

        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()
        self._hangup_attempted = False

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
        """Serializes a Pipecat frame to Plivo WebSocket format.

        Handles conversion of various frame types to Plivo WebSocket messages.
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
            self._hangup_attempted = True
            await self._hang_up_call()
            return None
        elif isinstance(frame, StartInterruptionFrame):
            answer = {"event": "clearAudio", "streamId": self._stream_id}
            return json.dumps(answer)
        elif isinstance(frame, AudioRawFrame):
            data = frame.audio

            # Output: Convert PCM at frame's rate to 8kHz μ-law for Plivo
            serialized_data = await pcm_to_ulaw(
                data, frame.sample_rate, self._plivo_sample_rate, self._output_resampler
            )
            if serialized_data is None or len(serialized_data) == 0:
                # Ignoring in case we don't have audio
                return None

            payload = base64.b64encode(serialized_data).decode("utf-8")
            answer = {
                "event": "playAudio",
                "media": {
                    "contentType": "audio/x-mulaw",
                    "sampleRate": self._plivo_sample_rate,
                    "payload": payload,
                },
                "streamId": self._stream_id,
            }

            return json.dumps(answer)
        elif isinstance(frame, (TransportMessageFrame, TransportMessageUrgentFrame)):
            return json.dumps(frame.message)

        # Return None for unhandled frames
        return None

    async def _hang_up_call(self):
        """Hang up the Plivo call using Plivo's REST API."""
        try:
            import aiohttp

            auth_id = self._auth_id
            auth_token = self._auth_token
            call_id = self._call_id

            if not call_id or not auth_id or not auth_token:
                missing = []
                if not call_id:
                    missing.append("call_id")
                if not auth_id:
                    missing.append("auth_id")
                if not auth_token:
                    missing.append("auth_token")

                logger.warning(
                    f"Cannot hang up Plivo call: missing required parameters: {', '.join(missing)}"
                )
                return

            # Plivo API endpoint for hanging up calls
            endpoint = f"https://api.plivo.com/v1/Account/{auth_id}/Call/{call_id}/"

            # Create basic auth from auth_id and auth_token
            auth = aiohttp.BasicAuth(auth_id, auth_token)

            # Make the DELETE request to hang up the call
            async with aiohttp.ClientSession() as session:
                async with session.delete(endpoint, auth=auth) as response:
                    if response.status == 204:  # Plivo returns 204 for successful hangup
                        logger.debug(f"Successfully terminated Plivo call {call_id}")
                    elif response.status == 404:  # Call already ended
                        logger.debug(f"Plivo call {call_id} already terminated")
                    else:
                        # Get the error details for better debugging
                        error_text = await response.text()
                        logger.error(
                            f"Failed to terminate Plivo call {call_id}: "
                            f"Status {response.status}, Response: {error_text}"
                        )

        except Exception as e:
            logger.exception(f"Failed to hang up Plivo call: {e}")

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserializes Plivo WebSocket data to Pipecat frames.

        Handles conversion of Plivo media events to appropriate Pipecat frames.

        Args:
            data: The raw WebSocket data from Plivo.

        Returns:
            A Pipecat frame corresponding to the Plivo event, or None if unhandled.
        """
        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON message: {data}")
            return None

        if message.get("event") == "media":
            media = message.get("media", {})
            payload_base64 = media.get("payload")

            if not payload_base64:
                return None

            payload = base64.b64decode(payload_base64)

            # Input: Convert Plivo's 8kHz μ-law to PCM at pipeline input rate
            deserialized_data = await ulaw_to_pcm(
                payload, self._plivo_sample_rate, self._sample_rate, self._input_resampler
            )
            if deserialized_data is None or len(deserialized_data) == 0:
                # Ignoring in case we don't have audio
                return None

            audio_frame = InputAudioRawFrame(
                audio=deserialized_data, num_channels=1, sample_rate=self._sample_rate
            )
            return audio_frame
        elif message.get("event") == "dtmf":
            dtmf_data = message.get("dtmf", {})
            digit = dtmf_data.get("digit")
            if digit:
                try:
                    return InputDTMFFrame(KeypadEntry(digit))
                except ValueError:
                    # Handle case where string doesn't match any enum value
                    logger.warning(f"Invalid DTMF digit received: {digit}")
                    return None
        else:
            return None
