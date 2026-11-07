#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Telnyx WebSocket frame serializer for Pipecat."""

import audioop
import base64
import json
from typing import Any, cast

import aiohttp
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


class TelnyxFrameSerializer(FrameSerializer):
    """Serializer for Telnyx WebSocket protocol.

    This serializer handles converting between Pipecat frames and Telnyx's WebSocket
    media streams protocol. It supports audio conversion, DTMF events, and automatic
    call termination.

    When auto_hang_up is enabled (default), the serializer will automatically terminate
    the Telnyx call when an EndFrame or CancelFrame is processed, but requires Telnyx
    credentials to be provided.
    """

    class InputParams(FrameSerializer.InputParams):
        """Configuration parameters for TelnyxFrameSerializer.

        Parameters:
            audio_in_sample_rate: Sample rate of the audio Telnyx sends, defaults to 8000 Hz.
            audio_out_sample_rate: Sample rate of the audio sent to Telnyx, defaults to 8000 Hz.
            telnyx_sample_rate: Sample rate used by Telnyx.

                .. deprecated:: 1.16.0
                    Use ``audio_in_sample_rate`` and ``audio_out_sample_rate`` instead.
                    Will be removed in 2.0.0.

            sample_rate: Has no effect: incoming audio is converted to the pipeline's input rate.

                .. deprecated:: 1.16.0
                    Use :attr:`~pipecat.pipeline.worker.PipelineParams.audio_in_sample_rate`
                    instead. Will be removed in 2.0.0.

            inbound_encoding: Audio encoding for data sent to Telnyx (e.g., "PCMU").
            outbound_encoding: Audio encoding for data received from Telnyx (e.g., "PCMU").
            auto_hang_up: Whether to automatically terminate call on EndFrame.
        """

        audio_in_sample_rate: int = 8000
        audio_out_sample_rate: int = 8000
        telnyx_sample_rate: int | None = None
        sample_rate: int | None = None
        inbound_encoding: str = "PCMU"
        outbound_encoding: str = "PCMU"
        auto_hang_up: bool = True

        @model_validator(mode="before")
        @classmethod
        def _carry_deprecated_rates(cls, data: Any) -> Any:
            if not isinstance(data, dict):
                return data
            if data.get("telnyx_sample_rate") is not None:
                warn_deprecated(
                    "`TelnyxFrameSerializer.InputParams.telnyx_sample_rate` is deprecated since "
                    "1.16.0 and will be removed in 2.0.0. Use `audio_in_sample_rate` and "
                    "`audio_out_sample_rate` instead.",
                    stacklevel=4,
                )
                data.setdefault("audio_in_sample_rate", data["telnyx_sample_rate"])
                data.setdefault("audio_out_sample_rate", data["telnyx_sample_rate"])
            if data.get("sample_rate") is not None:
                warn_deprecated(
                    "`TelnyxFrameSerializer.InputParams.sample_rate` is deprecated since 1.16.0 "
                    "and will be removed in 2.0.0. Use `PipelineParams.audio_in_sample_rate` "
                    "instead.",
                    stacklevel=4,
                )
            return data

    def __init__(
        self,
        stream_id: str,
        outbound_encoding: str,
        inbound_encoding: str,
        call_control_id: str | None = None,
        api_key: str | None = None,
        params: InputParams | None = None,
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
        params = params or TelnyxFrameSerializer.InputParams()
        super().__init__(params)
        self._params: TelnyxFrameSerializer.InputParams = params

        # Validate hangup-related parameters if auto_hang_up is enabled
        if self._params.auto_hang_up:
            missing_credentials = []
            if not call_control_id:
                missing_credentials.append("call_control_id")
            if not api_key:
                missing_credentials.append("api_key")

            if missing_credentials:
                raise ValueError(
                    f"auto_hang_up is enabled but missing required parameters: {', '.join(missing_credentials)}"
                )

        self._stream_id = stream_id
        self._call_control_id = call_control_id
        self._api_key = api_key
        self._params.outbound_encoding = outbound_encoding
        self._params.inbound_encoding = inbound_encoding

        self._hangup_attempted = False

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
        elif isinstance(frame, InterruptionFrame):
            await self._reset_audio_out()
            answer = {"event": "clear"}
            return json.dumps(answer)
        elif isinstance(frame, AudioRawFrame):
            data = await self._resample_audio_out(frame)
            if not data:
                return None

            if self._params.inbound_encoding == "PCMU":
                serialized_data = audioop.lin2ulaw(data, 2)
            elif self._params.inbound_encoding == "PCMA":
                serialized_data = audioop.lin2alaw(data, 2)
            else:
                raise ValueError(f"Unsupported encoding: {self._params.inbound_encoding}")

            payload = base64.b64encode(serialized_data).decode("utf-8")
            answer = {
                "event": "media",
                "media": {"payload": payload},
            }

            return json.dumps(answer)
        elif isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame):
                return None
            return json.dumps(frame.message)

        # Return None for unhandled frames
        return None

    async def _hang_up_call(self):
        """Hang up the Telnyx call using Telnyx's REST API."""
        try:
            # __init__ guarantees these are non-None whenever auto_hang_up is True,
            # which is the only path that reaches this method.
            call_control_id = cast(str, self._call_control_id)
            api_key = cast(str, self._api_key)

            # Telnyx API endpoint for hanging up a call
            endpoint = f"https://api.telnyx.com/v2/calls/{call_control_id}/actions/hangup"

            # Set headers with API key
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

            # Make the POST request to hang up the call
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, headers=headers) as response:
                    if response.status == 200:
                        logger.info(f"Successfully terminated Telnyx call {call_control_id}")
                    elif response.status == 422:
                        # Handle the case where the call has already ended
                        # Error code 90018: "Call has already ended"
                        # Source: https://developers.telnyx.com/api/errors/90018
                        try:
                            error_data = await response.json()
                            if any(
                                error.get("code") == "90018"
                                for error in error_data.get("errors", [])
                            ):
                                logger.debug(
                                    f"Telnyx call {call_control_id} was already terminated"
                                )
                                return
                        except Exception:
                            pass  # Fall through to log the raw error

                        # Log other 422 errors
                        error_text = await response.text()
                        logger.error(
                            f"Failed to terminate Telnyx call {call_control_id}: "
                            f"Status {response.status}, Response: {error_text}"
                        )
                    else:
                        # Log other errors
                        error_text = await response.text()
                        logger.error(
                            f"Failed to terminate Telnyx call {call_control_id}: "
                            f"Status {response.status}, Response: {error_text}"
                        )

        except Exception as e:
            logger.error(f"Failed to hang up Telnyx call: {e}")

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
        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON message: {data}")
            return None

        if message.get("event") == "media":
            payload_base64 = message.get("media", {}).get("payload")
            if not payload_base64:
                return None
            payload = base64.b64decode(payload_base64)

            if self._params.outbound_encoding == "PCMU":
                deserialized_data = audioop.ulaw2lin(payload, 2)
            elif self._params.outbound_encoding == "PCMA":
                deserialized_data = audioop.alaw2lin(payload, 2)
            else:
                raise ValueError(f"Unsupported encoding: {self._params.outbound_encoding}")

            audio = await self._resample_audio_in(deserialized_data)
            if not audio:
                return None

            return InputAudioRawFrame(audio=audio, num_channels=1, sample_rate=self._sample_rate)
        elif message.get("event") == "dtmf":
            digit = message.get("dtmf", {}).get("digit")

            try:
                return InputDTMFFrame(KeypadEntry(digit))
            except ValueError:
                # Handle case where string doesn't match any enum value
                return None
        else:
            return None
