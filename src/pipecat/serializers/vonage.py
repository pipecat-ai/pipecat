#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Vonage Audio Connector WebSocket serializer for Pipecat."""

import json
from typing import Any

from loguru import logger
from pydantic import model_validator

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InterruptionFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.utils.deprecation import warn_deprecated


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
            audio_in_sample_rate: Sample rate of the audio Vonage sends, defaults to
                16000 Hz. Common values: 8000, 16000, 24000 Hz.
            audio_out_sample_rate: Sample rate of the audio sent to Vonage, defaults to
                16000 Hz.
            vonage_sample_rate: Sample rate used by Vonage.

                .. deprecated:: 1.16.0
                    Use ``audio_in_sample_rate`` and ``audio_out_sample_rate`` instead.
                    Will be removed in 2.0.0.

            sample_rate: Has no effect: incoming audio is converted to the pipeline's input rate.

                .. deprecated:: 1.16.0
                    Use :attr:`~pipecat.pipeline.worker.PipelineParams.audio_in_sample_rate`
                    instead. Will be removed in 2.0.0.

            ignore_rtvi_messages: Inherited from base FrameSerializer, defaults to True.
        """

        audio_in_sample_rate: int = 16000
        audio_out_sample_rate: int = 16000
        vonage_sample_rate: int | None = None
        sample_rate: int | None = None

        @model_validator(mode="before")
        @classmethod
        def _carry_deprecated_rates(cls, data: Any) -> Any:
            if not isinstance(data, dict):
                return data
            if data.get("vonage_sample_rate") is not None:
                warn_deprecated(
                    "`VonageFrameSerializer.InputParams.vonage_sample_rate` is deprecated since "
                    "1.16.0 and will be removed in 2.0.0. Use `audio_in_sample_rate` and "
                    "`audio_out_sample_rate` instead.",
                    stacklevel=4,
                )
                data.setdefault("audio_in_sample_rate", data["vonage_sample_rate"])
                data.setdefault("audio_out_sample_rate", data["vonage_sample_rate"])
            if data.get("sample_rate") is not None:
                warn_deprecated(
                    "`VonageFrameSerializer.InputParams.sample_rate` is deprecated since 1.16.0 "
                    "and will be removed in 2.0.0. Use `PipelineParams.audio_in_sample_rate` "
                    "instead.",
                    stacklevel=4,
                )
            return data

    def __init__(self, params: InputParams | None = None):
        """Initialize the VonageFrameSerializer.

        Args:
            params: Configuration parameters.
        """
        params = params or VonageFrameSerializer.InputParams()
        super().__init__(params)
        self._params: VonageFrameSerializer.InputParams = params

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serializes a Pipecat frame to Vonage WebSocket format.

        Handles conversion of various frame types to Vonage WebSocket messages.

        Args:
            frame: The Pipecat frame to serialize.

        Returns:
            Serialized data as string (JSON commands) or bytes (audio), or None if the frame isn't handled.
        """
        if isinstance(frame, InterruptionFrame):
            await self._reset_audio_out()
            # Clear the audio buffer to stop playback immediately
            answer = {"action": "clear"}
            return json.dumps(answer)
        elif isinstance(frame, AudioRawFrame):
            data = await self._resample_audio_out(frame)
            if not data:
                return None

            # Vonage expects raw binary PCM data (not base64 encoded)
            return data
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
            audio = await self._resample_audio_in(data)
            if not audio:
                return None

            return InputAudioRawFrame(
                audio=audio,
                num_channels=1,  # Vonage uses mono audio
                sample_rate=self._sample_rate,
            )
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
