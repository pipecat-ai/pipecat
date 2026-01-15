#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Talkdesk Media Streams WebSocket protocol serializer for Pipecat."""

import base64
import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.audio.utils import create_stream_resampler, pcm_to_ulaw, ulaw_to_pcm
from pipecat.frames.frames import (
    AudioRawFrame,
    ControlFrame,
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InterruptionFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
    UninterruptibleFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType


class TalkdeskControlAction(Enum):
    """Control actions for Talkdesk call management.

    Parameters:
        HANGUP: End the call normally.
        ERROR: End the call due to an error.
        ESCALATE: Transfer the call to a human agent.
    """

    HANGUP = "ok"
    ERROR = "error"
    ESCALATE = "escalate"


@dataclass
class TalkdeskControlFrame(ControlFrame, UninterruptibleFrame):
    """Control frame for Talkdesk-specific call management.

    Used to signal call termination or escalation to a human agent.
    This is an uninterruptible frame to ensure the stop message is always
    delivered to Talkdesk.

    Parameters:
        action: The control action to perform (hangup or escalate).
        ring_group: Optional ring group for escalation (defaults to "agents").
    """

    action: TalkdeskControlAction = TalkdeskControlAction.HANGUP
    ring_group: Optional[str] = None


class TalkdeskFrameSerializer(FrameSerializer):
    """Serializer for Talkdesk Media Streams WebSocket protocol.

    This serializer handles converting between Pipecat frames and Talkdesk's WebSocket
    media streams protocol. It supports audio conversion, DTMF events, and call control
    (hangup/escalation).
    """

    class InputParams(BaseModel):
        """Configuration parameters for TalkdeskFrameSerializer.

        Parameters:
            talkdesk_sample_rate: Sample rate used by Talkdesk, defaults to 8000 Hz.
            sample_rate: Optional override for pipeline input sample rate.
        """

        talkdesk_sample_rate: int = 8000
        sample_rate: Optional[int] = None

    def __init__(
        self,
        stream_sid: str,
        params: Optional[InputParams] = None,
    ):
        """Initialize the TalkdeskFrameSerializer.

        Args:
            stream_sid: The Talkdesk Media Stream SID.
            params: Configuration parameters.
        """
        self._params = params or TalkdeskFrameSerializer.InputParams()
        self._stream_sid = stream_sid
        self._talkdesk_sample_rate = self._params.talkdesk_sample_rate
        self._sample_rate = 0  # Pipeline input rate

        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()
        self._stop_sent = False

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
        """Serializes a Pipecat frame to Talkdesk WebSocket format.

        Handles conversion of various frame types to Talkdesk WebSocket messages.

        Args:
            frame: The Pipecat frame to serialize.

        Returns:
            Serialized data as string or bytes, or None if the frame isn't handled.
        """
        if isinstance(frame, TalkdeskControlFrame):
            if self._stop_sent:
                return None
            self._stop_sent = True

            logger.info(f"Sending stop event to Talkdesk with command {frame.action.value}")

            return json.dumps(
                {
                    "event": "stop",
                    "streamSid": self._stream_sid,
                    "stop": {"command": frame.action.value, "ringGroup": frame.ring_group},
                }
            )
        elif isinstance(frame, InterruptionFrame):
            answer = {"event": "clear", "streamSid": self._stream_sid}
            return json.dumps(answer)
        elif isinstance(frame, AudioRawFrame):
            data = frame.audio

            # Output: Convert PCM at frame's rate to 8kHz μ-law for Talkdesk
            serialized_data = await pcm_to_ulaw(
                data, frame.sample_rate, self._talkdesk_sample_rate, self._output_resampler
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

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserializes Talkdesk WebSocket data to Pipecat frames.

        Handles conversion of Talkdesk media events to appropriate Pipecat frames.

        Args:
            data: The raw WebSocket data from Talkdesk.

        Returns:
            A Pipecat frame corresponding to the Talkdesk event, or None if unhandled.
        """
        message = json.loads(data)
        event = message.get("event")

        if event == "media":
            payload_base64 = message.get("media", {}).get("payload")
            payload = base64.b64decode(payload_base64)

            # Input: Convert Talkdesk's 8kHz μ-law to PCM at pipeline input rate
            deserialized_data = await ulaw_to_pcm(
                payload, self._talkdesk_sample_rate, self._sample_rate, self._input_resampler
            )
            if deserialized_data is None or len(deserialized_data) == 0:
                # Ignoring in case we don't have audio
                return None

            audio_frame = InputAudioRawFrame(
                audio=deserialized_data, num_channels=1, sample_rate=self._sample_rate
            )
            return audio_frame
        elif event == "dtmf":
            digit = message.get("dtmf", {}).get("digit")

            try:
                return InputDTMFFrame(KeypadEntry(digit))
            except ValueError:
                # Handle case where string doesn't match any enum value
                return None
        elif event == "stop":
            # Talkdesk has closed the stream from their side (caller hung up)
            logger.debug("Received stop event from Talkdesk")
            # Mark that stop has occurred to prevent sending duplicate stop
            self._stop_sent = True
            return None

        return None
