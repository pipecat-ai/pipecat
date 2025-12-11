#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Vobiz Media Streams WebSocket protocol serializer for Pipecat."""

import base64
import json
from typing import Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.audio.utils import create_stream_resampler, pcm_to_ulaw, ulaw_to_pcm
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
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType


class VobizFrameSerializer(FrameSerializer):
    """Serializer for Vobiz Media Streams WebSocket protocol.

    This serializer handles converting between Pipecat frames and Vobiz's WebSocket
    media streams protocol. It supports audio conversion, DTMF events, and automatic
    call termination.

    When auto_hang_up is enabled (default), the serializer will automatically terminate
    the Vobiz call when an EndFrame or CancelFrame is processed, but requires Vobiz
    credentials to be provided.
    """

    class InputParams(BaseModel):
        """Configuration parameters for VobizFrameSerializer.

        Parameters:
            vobiz_sample_rate: Sample rate used by Vobiz, defaults to 8000 Hz.
            sample_rate: Optional override for pipeline input sample rate.
            auto_hang_up: Whether to automatically terminate call on EndFrame.
        """

        vobiz_sample_rate: int = 8000
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
        """Initialize the VobizFrameSerializer.

        Args:
            stream_id: The Vobiz Media Stream ID.
            call_id: The associated Vobiz Call ID (optional, but required for auto hang-up).
            auth_id: Vobiz auth ID (required for auto hang-up).
            auth_token: Vobiz auth token (required for auto hang-up).
            params: Configuration parameters.
        """
        self._params = params or VobizFrameSerializer.InputParams()

        # Validate hangup-related parameters if auto_hang_up is enabled
        if self._params.auto_hang_up:
            # Validate required credentials
            missing_credentials = []
            if not call_id:
                missing_credentials.append("call_id")
            if not auth_id:
                missing_credentials.append("auth_id")
            if not auth_token:
                missing_credentials.append("auth_token")

            if missing_credentials:
                raise ValueError(
                    f"auto_hang_up is enabled but missing required parameters: {', '.join(missing_credentials)}"
                )

        self._stream_id = stream_id
        self._call_id = call_id
        self._auth_id = auth_id
        self._auth_token = auth_token

        self._vobiz_sample_rate = self._params.vobiz_sample_rate
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
        """Serializes a Pipecat frame to Vobiz WebSocket format.

        Handles conversion of various frame types to Vobiz WebSocket messages.
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
        elif isinstance(frame, InterruptionFrame):
            answer = {"event": "clearAudio", "streamId": self._stream_id}
            return json.dumps(answer)
        elif isinstance(frame, AudioRawFrame):
            data = frame.audio

            # Output: Convert PCM at frame's rate to 8kHz μ-law for Vobiz
            serialized_data = await pcm_to_ulaw(
                data, frame.sample_rate, self._vobiz_sample_rate, self._output_resampler
            )
            if serialized_data is None or len(serialized_data) == 0:
                # Ignoring in case we don't have audio
                return None

            payload = base64.b64encode(serialized_data).decode("utf-8")
            answer = {
                "event": "playAudio",
                "media": {
                    "contentType": "audio/x-mulaw",
                    "sampleRate": self._vobiz_sample_rate,
                    "payload": payload,
                },
                "streamId": self._stream_id,
            }

            return json.dumps(answer)
        elif isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            return json.dumps(frame.message)

        # Return None for unhandled frames
        return None

    async def _hang_up_call(self):
        """Hang up the Vobiz call using Vobiz's REST API."""
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
                    f"Cannot hang up Vobiz call: missing required parameters: {', '.join(missing)}"
                )
                return

            # Vobiz API endpoint for hanging up calls
            endpoint = f"https://api.vobiz.ai/api/v1/Account/{auth_id}/Call/{call_id}/"

            # Create headers for Vobiz authentication
            headers = {
                "X-Auth-ID": auth_id,
                "X-Auth-Token": auth_token,
            }

            # Make the DELETE request to hang up the call
            async with aiohttp.ClientSession() as session:
                async with session.delete(endpoint, headers=headers) as response:
                    if response.status == 204:
                        logger.info(f"Successfully terminated Vobiz call {call_id}")
                    elif response.status == 404:
                        # Call already ended
                        logger.debug(f"Vobiz call {call_id} already terminated")
                    else:
                        error_text = await response.text()

                        logger.error(
                            f"Vobiz call termination failure details - "
                            f"Call ID: {call_id}, "
                            f"HTTP Status: {response.status}, "
                            f"Request URL: {endpoint}, "
                            f"Auth ID: {auth_id[:8]}*****, "
                            f"Response Body: {error_text}"
                        )

        except Exception as e:
            logger.error(
                f"Exception during Vobiz call termination - "
                f"Call ID: {call_id}, "
                f"Auth ID: {auth_id[:8] if auth_id else 'None'}*****, "
                f"Exception: {type(e).__name__}: {e}"
            )

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserializes Vobiz WebSocket data to Pipecat frames.

        Handles conversion of Vobiz media events to appropriate Pipecat frames.

        Args:
            data: The raw WebSocket data from Vobiz.

        Returns:
            A Pipecat frame corresponding to the Vobiz event, or None if unhandled.
        """
        message = json.loads(data)

        if message["event"] == "media":
            payload_base64 = message["media"]["payload"]
            payload = base64.b64decode(payload_base64)

            # Input: Convert Vobiz's 8kHz μ-law to PCM at pipeline input rate
            deserialized_data = await ulaw_to_pcm(
                payload, self._vobiz_sample_rate, self._sample_rate, self._input_resampler
            )
            if deserialized_data is None or len(deserialized_data) == 0:
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
            except ValueError as e:
                # Handle case where string doesn't match any enum value
                return None
        else:
            return None
