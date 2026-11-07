#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Plivo WebSocket frame serializer for audio streaming."""

import audioop
import base64
import json
from typing import Any, cast

from loguru import logger
from pydantic import model_validator

from pipecat.audio.dtmf.types import KeypadEntry
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
)
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.utils.deprecation import warn_deprecated


class PlivoFrameSerializer(FrameSerializer):
    """Serializer for Plivo Audio Streaming WebSocket protocol.

    This serializer handles converting between Pipecat frames and Plivo's WebSocket
    audio streaming protocol. It supports audio conversion, DTMF events, and automatic
    call termination.

    When auto_hang_up is enabled (default), the serializer will automatically terminate
    the Plivo call when an EndFrame or CancelFrame is processed, but requires Plivo
    credentials to be provided.
    """

    class InputParams(FrameSerializer.InputParams):
        """Configuration parameters for PlivoFrameSerializer.

        Parameters:
            audio_in_sample_rate: Sample rate of the audio Plivo sends, defaults to 8000 Hz.
            audio_out_sample_rate: Sample rate of the audio sent to Plivo, defaults to 8000 Hz.
            plivo_sample_rate: Sample rate used by Plivo.

                .. deprecated:: 1.16.0
                    Use ``audio_in_sample_rate`` and ``audio_out_sample_rate`` instead.
                    Will be removed in 2.0.0.

            sample_rate: Has no effect: incoming audio is converted to the pipeline's input rate.

                .. deprecated:: 1.16.0
                    Use :attr:`~pipecat.pipeline.worker.PipelineParams.audio_in_sample_rate`
                    instead. Will be removed in 2.0.0.

            auto_hang_up: Whether to automatically terminate call on EndFrame.
            ignore_rtvi_messages: Inherited from base FrameSerializer, defaults to True.
        """

        audio_in_sample_rate: int = 8000
        audio_out_sample_rate: int = 8000
        plivo_sample_rate: int | None = None
        sample_rate: int | None = None
        auto_hang_up: bool = True

        @model_validator(mode="before")
        @classmethod
        def _carry_deprecated_rates(cls, data: Any) -> Any:
            if not isinstance(data, dict):
                return data
            if data.get("plivo_sample_rate") is not None:
                warn_deprecated(
                    "`PlivoFrameSerializer.InputParams.plivo_sample_rate` is deprecated since "
                    "1.16.0 and will be removed in 2.0.0. Use `audio_in_sample_rate` and "
                    "`audio_out_sample_rate` instead.",
                    stacklevel=4,
                )
                data.setdefault("audio_in_sample_rate", data["plivo_sample_rate"])
                data.setdefault("audio_out_sample_rate", data["plivo_sample_rate"])
            if data.get("sample_rate") is not None:
                warn_deprecated(
                    "`PlivoFrameSerializer.InputParams.sample_rate` is deprecated since 1.16.0 "
                    "and will be removed in 2.0.0. Use `PipelineParams.audio_in_sample_rate` "
                    "instead.",
                    stacklevel=4,
                )
            return data

    def __init__(
        self,
        stream_id: str,
        call_id: str | None = None,
        auth_id: str | None = None,
        auth_token: str | None = None,
        params: InputParams | None = None,
    ):
        """Initialize the PlivoFrameSerializer.

        Args:
            stream_id: The Plivo Stream ID.
            call_id: The associated Plivo Call ID (optional, but required for auto hang-up).
            auth_id: Plivo auth ID (required for auto hang-up).
            auth_token: Plivo auth token (required for auto hang-up).
            params: Configuration parameters.
        """
        params = params or PlivoFrameSerializer.InputParams()
        super().__init__(params)
        self._params: PlivoFrameSerializer.InputParams = params

        # Validate hangup-related parameters if auto_hang_up is enabled
        if self._params.auto_hang_up:
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

        self._hangup_attempted = False

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
        elif isinstance(frame, InterruptionFrame):
            await self._reset_audio_out()
            answer = {"event": "clearAudio", "streamId": self._stream_id}
            return json.dumps(answer)
        elif isinstance(frame, AudioRawFrame):
            data = await self._resample_audio_out(frame)
            if not data:
                return None

            payload = base64.b64encode(audioop.lin2ulaw(data, 2)).decode("utf-8")
            answer = {
                "event": "playAudio",
                "media": {
                    "contentType": "audio/x-mulaw",
                    "sampleRate": self._params.audio_out_sample_rate,
                    "payload": payload,
                },
                "streamId": self._stream_id,
            }

            return json.dumps(answer)
        elif isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame):
                return None
            return json.dumps(frame.message)

        # Return None for unhandled frames
        return None

    async def _hang_up_call(self):
        """Hang up the Plivo call using Plivo's REST API."""
        try:
            import aiohttp

            # __init__ guarantees these are non-None whenever auto_hang_up is True,
            # which is the only path that reaches this method.
            auth_id = cast(str, self._auth_id)
            auth_token = cast(str, self._auth_token)
            call_id = cast(str, self._call_id)

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
            logger.error(f"Failed to hang up Plivo call: {e}")

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

            audio = await self._resample_audio_in(audioop.ulaw2lin(payload, 2))
            if not audio:
                return None

            return InputAudioRawFrame(audio=audio, num_channels=1, sample_rate=self._sample_rate)
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
