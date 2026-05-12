#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Telnyx WebSocket frame serializer for Pipecat."""

import base64
import json
from typing import TYPE_CHECKING

from loguru import logger

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.audio.utils import (
    alaw_to_pcm,
    create_stream_resampler,
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
    InterruptionFrame,
    StartFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.utils.enums import EndTaskReason

if TYPE_CHECKING:
    from pipecat.serializers.call_strategies import HangupStrategy, TransferStrategy


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
            telnyx_sample_rate: Sample rate used by Telnyx, defaults to 8000 Hz.
            sample_rate: Optional override for pipeline input sample rate.
            inbound_encoding: Audio encoding for data sent to Telnyx (e.g., "PCMU").
            outbound_encoding: Audio encoding for data received from Telnyx (e.g., "PCMU").
            auto_hang_up: Whether to automatically terminate call on EndFrame.
        """

        telnyx_sample_rate: int = 8000
        sample_rate: int | None = None
        inbound_encoding: str = "PCMU"
        outbound_encoding: str = "PCMU"
        auto_hang_up: bool = True

    def __init__(
        self,
        stream_id: str,
        outbound_encoding: str,
        inbound_encoding: str,
        call_control_id: str | None = None,
        api_key: str | None = None,
        transfer_strategy: "TransferStrategy | None" = None,
        hangup_strategy: "HangupStrategy | None" = None,
        params: InputParams | None = None,
    ):
        """Initialize the TelnyxFrameSerializer.

        Args:
            stream_id: The Stream ID for Telnyx.
            outbound_encoding: The encoding type for outbound audio (e.g., "PCMU").
            inbound_encoding: The encoding type for inbound audio (e.g., "PCMU").
            call_control_id: The Call Control ID for the Telnyx call (optional, but required for auto hang-up).
            api_key: Your Telnyx API key (required for auto hang-up).
            transfer_strategy: Strategy for handling call transfers. Invoked on
                EndFrame/CancelFrame whose reason is EndTaskReason.TRANSFER_CALL.
            hangup_strategy: Strategy for handling call hangups. Required when
                auto_hang_up is True to actually terminate the call.
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
        self._transfer_strategy = transfer_strategy
        self._hangup_strategy = hangup_strategy
        self._params.outbound_encoding = outbound_encoding
        self._params.inbound_encoding = inbound_encoding

        self._telnyx_sample_rate = self._params.telnyx_sample_rate
        self._sample_rate = 0  # Pipeline input rate

        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()
        self._hangup_attempted = False
        self._transfer_attempted = False

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
        if isinstance(frame, (EndFrame, CancelFrame)):
            frame_reason = getattr(frame, "reason", None)
            if frame_reason == EndTaskReason.TRANSFER_CALL.value and not self._transfer_attempted:
                self._transfer_attempted = True
                if self._transfer_strategy:
                    context = {
                        "call_control_id": self._call_control_id,
                        "api_key": self._api_key,
                    }
                    success = await self._transfer_strategy.execute_transfer(context)
                    if not success:
                        logger.error(f"Transfer strategy failed for call {self._call_control_id}")
                else:
                    logger.warning(
                        f"No transfer strategy configured for call {self._call_control_id}"
                    )
                return None
            elif self._params.auto_hang_up and not self._hangup_attempted:
                self._hangup_attempted = True
                if self._hangup_strategy:
                    context = {
                        "call_control_id": self._call_control_id,
                        "api_key": self._api_key,
                    }
                    success = await self._hangup_strategy.execute_hangup(context)
                    if not success:
                        logger.error(f"Hangup strategy failed for call {self._call_control_id}")
                else:
                    logger.warning(
                        f"No hangup strategy configured for call {self._call_control_id}"
                    )
                return None
        elif isinstance(frame, InterruptionFrame):
            answer = {"event": "clear"}
            return json.dumps(answer)
        elif isinstance(frame, AudioRawFrame):
            data = frame.audio

            # Output: Convert PCM at frame's rate to 8kHz encoded for Telnyx
            if self._params.inbound_encoding == "PCMU":
                serialized_data = await pcm_to_ulaw(
                    data, frame.sample_rate, self._telnyx_sample_rate, self._output_resampler
                )
            elif self._params.inbound_encoding == "PCMA":
                serialized_data = await pcm_to_alaw(
                    data, frame.sample_rate, self._telnyx_sample_rate, self._output_resampler
                )
            else:
                raise ValueError(f"Unsupported encoding: {self._params.inbound_encoding}")

            if serialized_data is None or len(serialized_data) == 0:
                # Ignoring in case we don't have audio
                return None

            payload = base64.b64encode(serialized_data).decode("utf-8")
            answer = {
                "event": "media",
                "media": {"payload": payload},
            }

            return json.dumps(answer)

        # Return None for unhandled frames
        return None

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
                    self._input_resampler,
                )
            elif self._params.outbound_encoding == "PCMA":
                deserialized_data = await alaw_to_pcm(
                    payload,
                    self._telnyx_sample_rate,
                    self._sample_rate,
                    self._input_resampler,
                )
            else:
                raise ValueError(f"Unsupported encoding: {self._params.outbound_encoding}")

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
