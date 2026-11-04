#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Vobiz WebSocket frame serializer for audio streaming."""

import base64
import json
from typing import cast

from loguru import logger

from pipecat.audio.utils import create_stream_resampler, pcm_to_ulaw, ulaw_to_pcm
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.serializers.base_serializer import FrameSerializer

ENCODING_MULAW = "audio/x-mulaw"
ENCODING_L16 = "audio/x-l16"

# The (encoding, sample rate) pairs Vobiz supports in each direction:
# https://www.vobiz.ai/docs/xml/stream/audio-formats
INBOUND_FORMATS = frozenset({(ENCODING_MULAW, 8000), (ENCODING_L16, 8000), (ENCODING_L16, 16000)})
OUTBOUND_FORMATS = INBOUND_FORMATS | {(ENCODING_L16, 24000)}


class VobizFrameSerializer(FrameSerializer):
    """Serializer for Vobiz Audio Streaming WebSocket protocol.

    This serializer handles converting between Pipecat frames and Vobiz's WebSocket
    audio streaming protocol. It supports audio conversion and automatic call termination.

    When auto_hang_up is enabled (default), the serializer will automatically terminate
    the Vobiz call when an EndFrame or CancelFrame is processed, but requires Vobiz
    credentials to be provided.
    """

    class InputParams(FrameSerializer.InputParams):
        """Configuration parameters for VobizFrameSerializer.

        Parameters:
            vobiz_sample_rate: Inbound sample rate, defaults to 8000 Hz. Must match
                ``start.mediaFormat.sampleRate``: 8000 for μ-law, 8000 or 16000 for L16.
            encoding: Inbound encoding, ``audio/x-mulaw`` (default) or ``audio/x-l16``
                (little-endian). Must match ``start.mediaFormat.encoding``.
            output_sample_rate: Outbound sample rate, defaults to the inbound rate.
                L16 also supports 24000 outbound.
            output_encoding: Outbound encoding, defaults to the inbound encoding.
            sample_rate: Optional override for pipeline input sample rate.
            auto_hang_up: Whether to automatically terminate call on EndFrame.
            ignore_rtvi_messages: Inherited from base FrameSerializer, defaults to True.
        """

        vobiz_sample_rate: int = 8000
        encoding: str = ENCODING_MULAW
        output_sample_rate: int | None = None
        output_encoding: str | None = None
        sample_rate: int | None = None
        auto_hang_up: bool = True

    def __init__(
        self,
        stream_id: str,
        call_id: str | None = None,
        auth_id: str | None = None,
        auth_token: str | None = None,
        params: InputParams | None = None,
    ):
        """Initialize the VobizFrameSerializer.

        Args:
            stream_id: The Vobiz Stream ID.
            call_id: The associated Vobiz Call ID (optional, but required for auto hang-up).
            auth_id: Vobiz auth ID (required for auto hang-up).
            auth_token: Vobiz auth token (required for auto hang-up).
            params: Configuration parameters.
        """
        params = params or VobizFrameSerializer.InputParams()
        super().__init__(params)
        self._params: VobizFrameSerializer.InputParams = params

        self._vobiz_sample_rate = self._params.vobiz_sample_rate
        self._encoding = self._params.encoding
        self._output_sample_rate = self._params.output_sample_rate or self._vobiz_sample_rate
        self._output_encoding = self._params.output_encoding or self._encoding

        if (self._encoding, self._vobiz_sample_rate) not in INBOUND_FORMATS:
            raise ValueError(
                f"Unsupported Vobiz inbound format: {self._encoding} at "
                f"{self._vobiz_sample_rate} Hz"
            )
        if (self._output_encoding, self._output_sample_rate) not in OUTBOUND_FORMATS:
            raise ValueError(
                f"Unsupported Vobiz outbound format: {self._output_encoding} at "
                f"{self._output_sample_rate} Hz"
            )

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

        self._sample_rate = 0  # Pipeline input rate

        self._input_resampler = create_stream_resampler(
            clear_after_secs=self._params.resampler_clear_after_secs
        )
        self._output_resampler = create_stream_resampler(
            clear_after_secs=self._params.resampler_clear_after_secs
        )
        self._hangup_attempted = False

    async def setup(self, setup: FrameProcessorSetup):
        """Sets up the serializer with pipeline configuration.

        Args:
            setup: Configuration object containing setup parameters.
        """
        self._sample_rate = self._params.sample_rate or setup.audio_in_sample_rate

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

            # Output: Convert PCM at frame's rate to Vobiz's outbound format
            if self._output_encoding == ENCODING_MULAW:
                serialized_data = await pcm_to_ulaw(
                    data, frame.sample_rate, self._output_sample_rate, self._output_resampler
                )
            else:
                serialized_data = await self._output_resampler.resample(
                    data, frame.sample_rate, self._output_sample_rate
                )
            if serialized_data is None or len(serialized_data) == 0:
                # Ignoring in case we don't have audio
                return None

            payload = base64.b64encode(serialized_data).decode("utf-8")
            answer = {
                "event": "playAudio",
                "media": {
                    "contentType": self._output_encoding,
                    "sampleRate": self._output_sample_rate,
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
        """Hang up the Vobiz call using Vobiz's REST API."""
        try:
            import aiohttp

            # __init__ guarantees these are non-None whenever auto_hang_up is True,
            # which is the only path that reaches this method.
            auth_id = cast(str, self._auth_id)
            auth_token = cast(str, self._auth_token)
            call_id = cast(str, self._call_id)

            # Vobiz API endpoint for hanging up calls
            endpoint = f"https://api.vobiz.ai/api/v1/Account/{auth_id}/Call/{call_id}/"

            # Vobiz authenticates with X-Auth-ID and X-Auth-Token headers
            headers = {"X-Auth-ID": auth_id, "X-Auth-Token": auth_token}

            # Make the DELETE request to hang up the call
            async with aiohttp.ClientSession() as session:
                async with session.delete(endpoint, headers=headers) as response:
                    if response.status == 204:  # Vobiz returns 204 for successful hangup
                        logger.debug(f"Successfully terminated Vobiz call {call_id}")
                    elif response.status == 404:  # Call already ended
                        logger.debug(f"Vobiz call {call_id} already terminated")
                    else:
                        # Get the error details for better debugging
                        error_text = await response.text()
                        logger.error(
                            f"Failed to terminate Vobiz call {call_id}: "
                            f"Status {response.status}, Response: {error_text}"
                        )

        except Exception as e:
            logger.error(f"Failed to hang up Vobiz call: {e}")

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserializes Vobiz WebSocket data to Pipecat frames.

        Handles conversion of Vobiz media events to appropriate Pipecat frames.

        Args:
            data: The raw WebSocket data from Vobiz.

        Returns:
            A Pipecat frame corresponding to the Vobiz event, or None if unhandled.
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

            # Input: Convert Vobiz's inbound format to PCM at pipeline input rate
            if self._encoding == ENCODING_MULAW:
                deserialized_data = await ulaw_to_pcm(
                    payload, self._vobiz_sample_rate, self._sample_rate, self._input_resampler
                )
            else:
                deserialized_data = await self._input_resampler.resample(
                    payload, self._vobiz_sample_rate, self._sample_rate
                )
            if deserialized_data is None or len(deserialized_data) == 0:
                # Ignoring in case we don't have audio
                return None

            audio_frame = InputAudioRawFrame(
                audio=deserialized_data, num_channels=1, sample_rate=self._sample_rate
            )
            return audio_frame
        else:
            return None
