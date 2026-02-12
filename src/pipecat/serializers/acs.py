#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Azure Communication Services serializer for Pipecat."""

import base64
import json
from typing import Optional

from loguru import logger

try:
    from azure.communication.callautomation.aio import CallAutomationClient
    from azure.core.exceptions import ResourceNotFoundError
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use ACS, you need to `pip install pipecat-ai[acs-serializer]`.")
    raise ImportError(f"Missing module: {e}.") from e

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.audio.utils import create_stream_resampler
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
from pipecat.serializers.base_serializer import FrameSerializer


class ACSFrameSerializer(FrameSerializer):
    """Serializer for ACS Media Bidirectional WebSocket protocol.

    This serializer handles conversion between Pipecat frames and Azure Communication
    Services' WebSocket media streams protocol. It supports audio conversion, DTMF
    events, and interruptions. The call is automatically terminated when an EndFrame
    or CancelFrame is processed.

    Attributes:
        NUM_CHANNELS (int): Mono audio expected from ACS.
    """

    NUM_CHANNELS = 1

    class InputParams(FrameSerializer.InputParams):
        """Configuration parameters for ACSFrameSerializer.

        Parameters:
            acs_sample_rate: Sample rate used by ACS (16000 or 24000). Defaults to 24000.
            sample_rate: Optional override for pipeline input sample rate.
            ignore_rtvi_messages: Inherited from base FrameSerializer, defaults to True.
        """

        acs_sample_rate: int = 24000
        sample_rate: Optional[int] = None

    def __init__(
        self,
        call_automation_client: CallAutomationClient,
        call_connection_id: str,
        params: Optional[InputParams] = None,
    ):
        """Initialize the ACSFrameSerializer.

        Args:
            call_automation_client: Azure client for call control.
            call_connection_id: ID of the call used for automatic hang-up.
            params: Configuration parameters.
        """
        super().__init__(params or ACSFrameSerializer.InputParams())
        self._acs_sample_rate = self._params.acs_sample_rate
        self._sample_rate = 0  # Pipeline input rate, set by setup method

        self._call_automation_client = call_automation_client
        self._call_connection_id = call_connection_id
        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()

    async def setup(self, frame: StartFrame):
        """Sets up the serializer with pipeline configuration.

        Args:
            frame: The StartFrame containing pipeline configuration.
        """
        self._sample_rate = self._params.sample_rate or frame.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serializes a Pipecat frame to ACS WebSocket format.

        Handles conversion of various frame types to ACS WebSocket messages.
        For EndFrame and CancelFrame, initiates call termination.

        Args:
            frame: The Pipecat frame to serialize.

        Returns:
            Serialized data as string or bytes, or None if the frame isn't handled.
        """
        if isinstance(frame, (EndFrame, CancelFrame)):
            logger.info(f"Hanging up call (call_connection_id={self._call_connection_id})")
            await self._hang_up_call()
            return None

        elif isinstance(frame, AudioRawFrame):
            data = frame.audio
            # Output: ACS expects PCM audio, but we need to resample to match requested sample_rate
            serialized_data = await self._output_resampler.resample(
                data, in_rate=frame.sample_rate, out_rate=self._acs_sample_rate
            )
            if not serialized_data:
                # Ignoring in case we don't have audio
                return None
            payload = base64.b64encode(serialized_data).decode("ascii")
            answer = {"Kind": "AudioData", "AudioData": {"data": payload}, "StopAudio": None}
            return json.dumps(answer)

        elif isinstance(frame, InterruptionFrame):
            answer = {"Kind": "StopAudio", "AudioData": None, "StopAudio": {}}
            return json.dumps(answer)

        elif isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame):
                return None
            return json.dumps(frame.message)

        return None

    async def _hang_up_call(self):
        """Terminates the active Azure Call Automation connection."""
        if self._call_connection_id is None:
            logger.warning("Could not hang up call. No call connection ID was provided.")
            return

        try:
            call_connection = self._call_automation_client.get_call_connection(
                self._call_connection_id
            )
            await call_connection.hang_up(is_for_everyone=True)
            logger.info(f"Hung up call with connection ID: {self._call_connection_id}")
        except ResourceNotFoundError:
            logger.warning(f"Call {self._call_connection_id} was already disconnected")

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserializes ACS WebSocket data to Pipecat frames.

        Handles conversion of ACS media events to appropriate Pipecat frames.

        Args:
            data: The raw WebSocket data from ACS.

        Returns:
            A Pipecat frame corresponding to the ACS event, or None if unhandled.
        """
        obj = json.loads(data)
        kind = obj.get("kind")

        if kind == "DtmfData":
            digit = obj["dtmfData"]["data"]
            try:
                return InputDTMFFrame(KeypadEntry(digit))
            except ValueError:
                # Handle case where string doesn't match any enum value
                logger.info(f"Invalid DTMF digit received: {digit}")
                return None

        if kind == "AudioData":
            payload_base64 = obj["audioData"]["data"]
            payload = base64.b64decode(payload_base64)

            deserialized_data = await self._input_resampler.resample(
                payload,
                in_rate=self._acs_sample_rate,
                out_rate=self._sample_rate,
            )

            if not deserialized_data:
                # Ignoring in case we don't have audio
                return None

            return InputAudioRawFrame(
                audio=deserialized_data,
                num_channels=self.NUM_CHANNELS,
                sample_rate=self._sample_rate,
            )

        return None
