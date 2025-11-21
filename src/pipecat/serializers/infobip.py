"""Infobip Media Streams WebSocket protocol serializer for Pipecat."""

import json
from typing import Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    KeypadEntry,
    OutputAudioRawFrame,
    StartFrame,
    StartInterruptionFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
    TTSAudioRawFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType


class InfobipFrameSerializer(FrameSerializer):
    """Serializer for Infobip WebSocket Media Streaming protocol.

    This serializer handles converting between Pipecat frames and Infobip's WebSocket
    media streams protocol. It supports Linear PCM 16-bit audio audio conversion, DTMF events, and automatic
    call termination.

    When auto_hang_up is enabled (default), the serializer will automatically terminate
    the Infobip call when an EndFrame or CancelFrame is processed, but requires Infobip
    credentials to be provided.
    """

    class InputParams(BaseModel):
        """Configuration parameters for InfobipFrameSerializer.

        Parameters:
            infobip_sample_rate: Sample rate used by Infobip (8000 or 16000 Hz).
            sample_rate: Optional override for pipeline input sample rate.
            call_id: The call ID for the Infobip call (optional, but required for auto hang-up).
            api_key: Your Infobip API key (required for auto hang-up).
            base_url: The Infobip base URL (default: api.infobip.com).
            auto_hang_up: Whether to automatically terminate call on EndFrame.
        """

        infobip_sample_rate: int = 16000
        sample_rate: Optional[int] = None
        call_id: Optional[str] = None
        api_key: Optional[str] = None
        base_url: str = "api.infobip.com"
        auto_hang_up: bool = True

    def __init__(self, params: Optional[InputParams] = None):
        """Initialize the InfobipFrameSerializer.

        Args:
            params: Optional configuration parameters for the serializer.
        """
        self._params = params or InfobipFrameSerializer.InputParams()
        self._sample_rate = 0
        self._infobip_sample_rate = self._params.infobip_sample_rate
        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()
        self._hangup_attempted = False

    @property
    def type(self) -> FrameSerializerType:
        """Gets the serializer type.

        Returns:
            The serializer type, BINARY for Infobip audio frames.
        """
        return FrameSerializerType.BINARY

    async def setup(self, frame: StartFrame):
        """Sets up the serializer with pipeline configuration.

        Args:
            frame: The StartFrame containing pipeline configuration.
        """
        self._sample_rate = self._params.sample_rate or frame.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serializes a Pipecat frame to Infobip WebSocket format.

        Handles conversion of various frame types to Infobip WebSocket messages.

        Args:
            frame: The Pipecat frame to serialize.

        Returns:
            Serialized data as string or bytes, or None if the frame isn't handled.
        """
        if isinstance(frame, StartInterruptionFrame):
            return None
        elif isinstance(frame, (OutputAudioRawFrame, TTSAudioRawFrame)):
            data = frame.audio
            resampled_data = await self._input_resampler(
                data, frame.sample_rate, self._infobip_sample_rate
            )
            if not resampled_data:
                return None
            return resampled_data
        elif isinstance(frame, EndFrame):
            if (
                self._params.auto_hang_up
                and not self._hangup_attempted
                and self._params.call_id
                and self._params.api_key
            ):
                self._hangup_attempted = True
                await self._hang_up_call()
            logger.info("EndFrame received - call ending")
            return None

        elif isinstance(frame, (TransportMessageFrame, TransportMessageUrgentFrame)):
            if isinstance(frame.message, dict):
                result = json.dumps(frame.message)
                return result
            return frame.message
        return None

    async def _hang_up_call(self):
        """Hang up the Infobip call using Infobip's REST API."""
        try:
            call_id = self._params.call_id
            api_key = self._params.api_key
            base_url = self._params.base_url

            if not call_id or not api_key:
                logger.warning(
                    "Cannot hang up Infobip call: call_id and api_key must be provided"
                )
                return

            if call_id == "unknown":
                logger.warning("No valid call_id for remote hangup; local end only.")
                return

            # Infobip API endpoint for hanging up a call
            endpoint = f"https://{base_url}/calls/1/calls/{call_id}/hangup"

            # Set headers with API key
            headers = {
                "Authorization": api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

            # Make the POST request to hang up the call
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, headers=headers, json={}) as response:
                    response_text = await response.text()
                    logger.info(
                        f"Hangup response call_id={call_id} status={response.status} body={response_text}"
                    )

        except Exception as e:
            logger.error(f"Hangup request failed: {e}")

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserializes Infobip WebSocket data to Pipecat frames.

        Handles conversion of Infobip media events to appropriate Pipecat frames.

        Args:
            data: The raw WebSocket data from Infobip.

        Returns:
            A Pipecat frame corresponding to the Infobip event, or None if unhandled.
        """
        if isinstance(data, str):
            try:
                message = json.loads(data)
                event_type = message.get("event")
                if event_type == "websocket:dtmf":
                    digit = message.get("digit")
                    try:
                        result = InputDTMFFrame(KeypadEntry(digit))
                        return result
                    except ValueError:
                        return None
                elif event_type == "websocket:connected":
                    return None
            except json.JSONDecodeError as e:
                logger.error(f"deserialize: JSONDecodeError {e}")
                return None
        elif isinstance(data, bytes):
            if not data:
                return None
            resampled_data = await self._output_resampler(
                data, self._sample_rate, self._infobip_sample_rate
            )
            if resampled_data is not None and len(resampled_data) > 0:
                audio_frame = InputAudioRawFrame(
                    audio=resampled_data,
                    num_channels=1,
                    sample_rate=self._sample_rate,
                )
                return audio_frame
            return None
        return None
