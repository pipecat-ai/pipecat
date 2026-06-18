#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Vonage Audio Connector WebSocket serializer for Pipecat."""

import base64
import json
from enum import StrEnum
from typing import Any, Literal

from loguru import logger
from pydantic import Field

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InterruptionFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer


class VonageAudioTransport(StrEnum):
    """Audio transport formats supported by Vonage Audio Connector."""

    BINARY = "binary"
    JSON = "json"


class VonageFrameSerializer(FrameSerializer):
    """Serializer for Vonage Video API Audio Connector WebSocket protocol.

    This serializer converts between Pipecat frames and the Vonage Audio Connector
    WebSocket streaming protocol.

    Note:
        Ref docs: https://developer.vonage.com/en/video/guides/audio-connector
    """

    class InputParams(FrameSerializer.InputParams):
        """Configuration parameters for VonageFrameSerializer.

        Parameters:
            vonage_sample_rate: Sample rate used by Vonage, defaults to 16000 Hz.
                Common values: 8000, 16000, 24000 Hz.
            sample_rate: Optional override for pipeline input sample rate.
            audio_transport: Audio transport selected in the Vonage Audio Connector
                configuration. ``binary`` sends raw PCM bytes. ``json`` sends JSON
                text messages with base64-encoded PCM in the configured audio field.
            encoding: Encoding used by the JSON audio transport. Currently only
                ``base64`` is supported.
            audio_field: Outbound JSON field name for base64-encoded audio.
            receive_audio_field: Optional inbound JSON field name for base64-encoded
                audio. Defaults to ``audio_field``.
            static_fields: Additional fields to include in every outbound JSON audio
                message.
            ignore_rtvi_messages: Inherited from base FrameSerializer, defaults to True.
        """

        vonage_sample_rate: int = 16000
        sample_rate: int | None = None
        audio_transport: VonageAudioTransport = VonageAudioTransport.BINARY
        encoding: Literal["base64"] = "base64"
        audio_field: str = "audio"
        receive_audio_field: str | None = None
        static_fields: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, params: InputParams | None = None):
        """Initialize the VonageFrameSerializer.

        Args:
            params: Configuration parameters.
        """
        params = params or VonageFrameSerializer.InputParams()
        super().__init__(params)
        self._params: VonageFrameSerializer.InputParams = params

        self._vonage_sample_rate = self._params.vonage_sample_rate
        self._sample_rate = 0  # Pipeline input rate
        self._receive_audio_field = self._params.receive_audio_field or self._params.audio_field

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
            Serialized data as string (JSON commands) or bytes (audio), or None if the
            frame isn't handled.
        """
        if isinstance(frame, InterruptionFrame):
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

            if self._params.audio_transport == VonageAudioTransport.JSON:
                return self._serialize_json_audio(serialized_data)

            # By default, Vonage expects raw binary PCM data (not base64 encoded).
            return serialized_data
        elif isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame):
                return None
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
            return await self._deserialize_audio(data)
        else:
            # Text message = JSON event
            try:
                message = json.loads(data)
                audio_payload = message.get(self._receive_audio_field)
                if audio_payload is not None:
                    return await self._deserialize_json_audio(audio_payload)

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

    def _serialize_json_audio(self, data: bytes) -> str:
        payload = base64.b64encode(data).decode("ascii")
        message = {
            **self._params.static_fields,
            self._params.audio_field: payload,
        }
        return json.dumps(message, separators=(",", ":"))

    async def _deserialize_json_audio(self, audio_payload: Any) -> Frame | None:
        if not isinstance(audio_payload, str):
            logger.warning(
                f"Vonage JSON audio field '{self._receive_audio_field}' is not a string"
            )
            return None

        try:
            payload = base64.b64decode(audio_payload)
        except Exception as e:
            logger.warning(f"Failed to decode Vonage base64 audio payload: {e}")
            return None

        return await self._deserialize_audio(payload)

    async def _deserialize_audio(self, payload: bytes) -> Frame | None:
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
