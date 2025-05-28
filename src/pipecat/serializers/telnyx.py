#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json
from typing import Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import (
    alaw_to_pcm,
    create_default_resampler,
    pcm_to_alaw,
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
    KeypadEntry,
    StartFrame,
    StartInterruptionFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType


class TelnyxFrameSerializer(FrameSerializer):
    """Serializer for Telnyx WebSocket protocol.

    This serializer handles converting between Pipecat frames and Telnyx's WebSocket
    media streams protocol. It supports audio conversion, DTMF events, and automatic
    call termination.

    When auto_hang_up is enabled (default), the serializer will automatically terminate
    the Telnyx call when an EndFrame or CancelFrame is processed, but requires Telnyx
    credentials to be provided.

    Attributes:
        _stream_id: The Telnyx Stream ID.
        _call_control_id: The associated Telnyx Call Control ID.
        _api_key: Telnyx API key for API access.
        _params: Configuration parameters.
        _telnyx_sample_rate: Sample rate used by Telnyx (typically 8kHz).
        _sample_rate: Input sample rate for the pipeline.
        _resampler: Audio resampler for format conversion.
        _hangup_attempted: Flag to track if hang-up has been attempted.
    """

    class InputParams(BaseModel):
        """Configuration parameters for TelnyxFrameSerializer.

        Attributes:
            telnyx_sample_rate: Sample rate used by Telnyx, defaults to 8000 Hz.
            sample_rate: Optional override for pipeline input sample rate.
            inbound_encoding: Audio encoding for data sent to Telnyx (e.g., "PCMU").
            outbound_encoding: Audio encoding for data received from Telnyx (e.g., "PCMU").
            auto_hang_up: Whether to automatically terminate call on EndFrame.
        """

        telnyx_sample_rate: int = 8000
        sample_rate: Optional[int] = None
        inbound_encoding: str = "PCMU"
        outbound_encoding: str = "PCMU"
        auto_hang_up: bool = True

    def __init__(
        self,
        stream_id: str,
        outbound_encoding: str,
        inbound_encoding: str,
        call_control_id: Optional[str] = None,
        api_key: Optional[str] = None,
        params: Optional[InputParams] = None,
    ):
        """Initialize the TelnyxFrameSerializer.

        Args:
            stream_id: The Stream ID for Telnyx.
            outbound_encoding: The encoding type for outbound audio (e.g., "PCMU").
            inbound_encoding: The encoding type for inbound audio (e.g., "PCMU").
            call_control_id: The Call Control ID for the Telnyx call (optional, but required for auto hang-up).
            api_key: Your Telnyx API key (required for auto hang-up).
            params: Configuration parameters.
        """
        self._stream_id = stream_id
        self._call_control_id = call_control_id
        self._api_key = api_key
        self._params = params or TelnyxFrameSerializer.InputParams()
        self._params.outbound_encoding = outbound_encoding
        self._params.inbound_encoding = inbound_encoding

        self._telnyx_sample_rate = self._params.telnyx_sample_rate
        self._sample_rate = 0  # Pipeline input rate

        self._resampler = create_default_resampler()
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
        """Serializes a Pipecat frame to Telnyx WebSocket format.

        Handles conversion of various frame types to Telnyx WebSocket messages.
        For EndFrames and CancelFrames, initiates call termination if auto_hang_up is enabled.

        Args:
            frame: The Pipecat frame to serialize.

        Returns:
            Serialized data as string or bytes, or None if the frame isn't handled.

        Raises:
            ValueError: If an unsupported encoding is specified.
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
            answer = {"event": "clear"}
            return json.dumps(answer)
        elif isinstance(frame, AudioRawFrame):
            data = frame.audio

            # Output: Convert PCM at frame's rate to 8kHz encoded for Telnyx
            if self._params.inbound_encoding == "PCMU":
                serialized_data = await pcm_to_ulaw(
                    data, frame.sample_rate, self._telnyx_sample_rate, self._resampler
                )
            elif self._params.inbound_encoding == "PCMA":
                serialized_data = await pcm_to_alaw(
                    data, frame.sample_rate, self._telnyx_sample_rate, self._resampler
                )
            else:
                raise ValueError(f"Unsupported encoding: {self._params.inbound_encoding}")

            payload = base64.b64encode(serialized_data).decode("utf-8")
            answer = {
                "event": "media",
                "media": {"payload": payload},
            }

            return json.dumps(answer)

        # Return None for unhandled frames
        return None

    async def _hang_up_call(self):
        """Hang up the Telnyx call using Telnyx's REST API."""
        try:
            call_control_id = self._call_control_id
            api_key = self._api_key

            if not call_control_id or not api_key:
                logger.warning(
                    "Cannot hang up Telnyx call: call_control_id and api_key must be provided"
                )
                return

            # Telnyx API endpoint for hanging up a call
            endpoint = f"https://api.telnyx.com/v2/calls/{call_control_id}/actions/hangup"

            # Set headers with API key
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

            # Make the POST request to hang up the call
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, headers=headers) as response:
                    if response.status == 200:
                        logger.info(f"Successfully terminated Telnyx call {call_control_id}")
                    else:
                        # Get the error details for better debugging
                        error_text = await response.text()
                        logger.error(
                            f"Failed to terminate Telnyx call {call_control_id}: "
                            f"Status {response.status}, Response: {error_text}"
                        )

        except Exception as e:
            logger.exception(f"Failed to hang up Telnyx call: {e}")

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserializes Telnyx WebSocket data to Pipecat frames.

        Handles conversion of Telnyx media events to appropriate Pipecat frames,
        including audio data and DTMF keypresses.

        Args:
            data: The raw WebSocket data from Telnyx.

        Returns:
            A Pipecat frame corresponding to the Telnyx event, or None if unhandled.

        Raises:
            ValueError: If an unsupported encoding is specified.
        """
        message = json.loads(data)

        if message["event"] == "media":
            payload_base64 = message["media"]["payload"]
            payload = base64.b64decode(payload_base64)

            # Input: Convert Telnyx's 8kHz encoded audio to PCM at pipeline input rate
            if self._params.outbound_encoding == "PCMU":
                deserialized_data = await ulaw_to_pcm(
                    payload,
                    self._telnyx_sample_rate,
                    self._sample_rate,
                    self._resampler,
                )
            elif self._params.outbound_encoding == "PCMA":
                deserialized_data = await alaw_to_pcm(
                    payload,
                    self._telnyx_sample_rate,
                    self._sample_rate,
                    self._resampler,
                )
            else:
                raise ValueError(f"Unsupported encoding: {self._params.outbound_encoding}")

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
