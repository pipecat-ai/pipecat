#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json
from typing import Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import create_default_resampler, pcm_to_ulaw, ulaw_to_pcm
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


class TwilioFrameSerializer(FrameSerializer):
    """Serializer for Twilio Media Streams WebSocket protocol.

    This serializer handles converting between Pipecat frames and Twilio's WebSocket
    media streams protocol. It supports audio conversion, DTMF events, and automatic
    call termination.

    When auto_hang_up is enabled (default), the serializer will automatically terminate
    the Twilio call when an EndFrame or CancelFrame is processed, but requires Twilio
    credentials to be provided.

    Attributes:
        _stream_sid: The Twilio Media Stream SID.
        _call_sid: The associated Twilio Call SID.
        _account_sid: Twilio account SID for API access.
        _auth_token: Twilio authentication token for API access.
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

    def __init__(
        self,
        stream_sid: str,
        call_sid: Optional[str] = None,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None,
        params: Optional[InputParams] = None,
    ):
        """Initialize the TwilioFrameSerializer.

        Args:
            stream_sid: The Twilio Media Stream SID.
            call_sid: The associated Twilio Call SID (optional, but required for auto hang-up).
            account_sid: Twilio account SID (required for auto hang-up).
            auth_token: Twilio auth token (required for auto hang-up).
            params: Configuration parameters.
        """
        self._stream_sid = stream_sid
        self._call_sid = call_sid
        self._account_sid = account_sid
        self._auth_token = auth_token
        self._params = params or TwilioFrameSerializer.InputParams()

        self._twilio_sample_rate = self._params.twilio_sample_rate
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
        """Serializes a Pipecat frame to Twilio WebSocket format.

        Handles conversion of various frame types to Twilio WebSocket messages.
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
        """Hang up the Twilio call using Twilio's REST API."""
        try:
            import aiohttp

            account_sid = self._account_sid
            auth_token = self._auth_token
            call_sid = self._call_sid

            if not call_sid or not account_sid or not auth_token:
                missing = []
                if not call_sid:
                    missing.append("call_sid")
                if not account_sid:
                    missing.append("account_sid")
                if not auth_token:
                    missing.append("auth_token")

                logger.warning(
                    f"Cannot hang up Twilio call: missing required parameters: {', '.join(missing)}"
                )
                return

            # Twilio API endpoint for updating calls
            endpoint = (
                f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Calls/{call_sid}.json"
            )

            # Create basic auth from account_sid and auth_token
            auth = aiohttp.BasicAuth(account_sid, auth_token)

            # Parameters to set the call status to "completed" (hang up)
            params = {"Status": "completed"}

            # Make the POST request to update the call
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, auth=auth, data=params) as response:
                    if response.status == 200:
                        logger.info(f"Successfully terminated Twilio call {call_sid}")
                    else:
                        # Get the error details for better debugging
                        error_text = await response.text()
                        logger.error(
                            f"Failed to terminate Twilio call {call_sid}: "
                            f"Status {response.status}, Response: {error_text}"
                        )

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
