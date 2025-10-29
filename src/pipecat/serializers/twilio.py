#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Twilio Media Streams WebSocket protocol serializer for Pipecat."""

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


class TwilioFrameSerializer(FrameSerializer):
    """Serializer for Twilio Media Streams WebSocket protocol.

    This serializer handles converting between Pipecat frames and Twilio's WebSocket
    media streams protocol. It supports audio conversion, DTMF events, and automatic
    call termination.

    When auto_hang_up is enabled (default), the serializer will automatically terminate
    the Twilio call when an EndFrame or CancelFrame is processed, but requires Twilio
    credentials to be provided.
    """

    class InputParams(BaseModel):
        """Configuration parameters for TwilioFrameSerializer.

        Parameters:
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
        region: Optional[str] = None,
        edge: Optional[str] = None,
        params: Optional[InputParams] = None,
    ):
        """Initialize the TwilioFrameSerializer.

        Args:
            stream_sid: The Twilio Media Stream SID.
            call_sid: The associated Twilio Call SID (optional, but required for auto hang-up).
            account_sid: Twilio account SID (required for auto hang-up).
            auth_token: Twilio auth token (required for auto hang-up).
            region: Twilio region (e.g., "au1", "ie1"). Must be specified with edge.
            edge: Twilio edge location (e.g., "sydney", "dublin"). Must be specified with region.
            params: Configuration parameters.
        """
        self._params = params or TwilioFrameSerializer.InputParams()

        # Validate hangup-related parameters if auto_hang_up is enabled
        if self._params.auto_hang_up:
            # Validate required credentials
            missing_credentials = []
            if not call_sid:
                missing_credentials.append("call_sid")
            if not account_sid:
                missing_credentials.append("account_sid")
            if not auth_token:
                missing_credentials.append("auth_token")

            if missing_credentials:
                raise ValueError(
                    f"auto_hang_up is enabled but missing required parameters: {', '.join(missing_credentials)}"
                )

            # Validate region and edge are both provided if either is specified
            if (region and not edge) or (edge and not region):
                raise ValueError(
                    "Both edge and region parameters are required if one is set. "
                    f"Twilio's FQDN format requires both: api.{{edge}}.{{region}}.twilio.com. "
                    f"Got: region='{region}', edge='{edge}'"
                )

        self._stream_sid = stream_sid
        self._call_sid = call_sid
        self._account_sid = account_sid
        self._auth_token = auth_token
        self._region = region
        self._edge = edge

        self._twilio_sample_rate = self._params.twilio_sample_rate
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
        elif isinstance(frame, InterruptionFrame):
            answer = {"event": "clear", "streamSid": self._stream_sid}
            return json.dumps(answer)
        elif isinstance(frame, AudioRawFrame):
            data = frame.audio

            # Output: Convert PCM at frame's rate to 8kHz μ-law for Twilio
            serialized_data = await pcm_to_ulaw(
                data, frame.sample_rate, self._twilio_sample_rate, self._output_resampler
            )
            if serialized_data is None or len(serialized_data) == 0:
                # Ignoring in case we don't have audio
                return None

            payload = base64.b64encode(serialized_data).decode("utf-8")
            answer = {
                "event": "media",
                "streamSid": self._stream_sid,
                "media": {"payload": payload},
            }

            return json.dumps(answer)
        elif isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
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
            region = self._region
            edge = self._edge

            region_prefix = f"{region}." if region else ""
            edge_prefix = f"{edge}." if edge else ""

            # Twilio API endpoint for updating calls
            endpoint = f"https://api.{edge_prefix}{region_prefix}twilio.com/2010-04-01/Accounts/{account_sid}/Calls/{call_sid}.json"

            # Create basic auth from account_sid and auth_token
            auth = aiohttp.BasicAuth(account_sid, auth_token)

            # Parameters to set the call status to "completed" (hang up)
            params = {"Status": "completed"}

            # Make the POST request to update the call
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, auth=auth, data=params) as response:
                    if response.status == 200:
                        logger.info(f"Successfully terminated Twilio call {call_sid}")
                    elif response.status == 404:
                        # Handle the case where the call has already ended
                        # Error code 20404: "The requested resource was not found"
                        # Source: https://www.twilio.com/docs/errors/20404
                        try:
                            error_data = await response.json()
                            if error_data.get("code") == 20404:
                                logger.debug(f"Twilio call {call_sid} was already terminated")
                                return
                        except:
                            pass  # Fall through to log the raw error

                        # Log other 404 errors
                        error_text = await response.text()
                        logger.error(
                            f"Failed to terminate Twilio call {call_sid}: "
                            f"Status {response.status}, Response: {error_text}"
                        )
                    else:
                        # Log other errors
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
                payload, self._twilio_sample_rate, self._sample_rate, self._input_resampler
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
