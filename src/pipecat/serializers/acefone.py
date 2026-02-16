#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Acefone/Smartflo Media Streams WebSocket protocol serializer for Pipecat."""

import base64
import json
from enum import StrEnum
from typing import Any

from loguru import logger

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.audio.utils import (
    create_stream_resampler,
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
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer


class AcefoneMediaFormat(StrEnum):
    """Wire audio format negotiated with Acefone/Smartflo.

    Parameters:
        ULAW: 8kHz G.711 μ-law. The default and most common telephony format.
        PCM: 16-bit signed linear PCM (``slin16``).
    """

    ULAW = "ulaw"
    PCM = "slin16"


class AcefoneFrameSerializer(FrameSerializer):
    """Serializer for the Acefone/Smartflo Media Streams WebSocket protocol.

    Converts between Pipecat frames and Acefone's (and the compatible Smartflo)
    bidirectional audio streaming protocol. It handles audio conversion, DTMF
    keypad events, interruptions, and optional automatic call termination.

    The wire audio format is configurable via `InputParams.media_format`
    (`AcefoneMediaFormat.ULAW` by default, or `AcefoneMediaFormat.PCM` for 16-bit
    linear PCM). Regardless of the wire format, audio is normalized to PCM
    `InputAudioRawFrame` for the pipeline.

    Unlike REST-based providers, auto hang-up is performed by sending an ``end``
    event over the same WebSocket connection. When ``auto_hang_up`` is enabled,
    the serializer emits this event on the first `EndFrame` or `CancelFrame`.

    Note:
        Reference docs for the wire protocol and events:

        - https://docs.acefone.in/docs/bi-directional-audio-streaming-integration-document
        - https://docs.smartflo.tatatelebusiness.com/docs/bi-directional-audio-streaming-integration-document
    """

    class InputParams(FrameSerializer.InputParams):
        """Configuration parameters for `AcefoneFrameSerializer`.

        Parameters:
            media_format: Wire audio format exchanged with Acefone/Smartflo.
                `AcefoneMediaFormat.ULAW` (the default) uses 8kHz μ-law;
                `AcefoneMediaFormat.PCM` uses 16-bit linear PCM at
                ``acefone_sample_rate``.
            acefone_sample_rate: Sample rate used by Acefone/Smartflo, in Hz.
                Defaults to 8000.
            sample_rate: Optional override for the pipeline input sample rate. When
                0 (the default), the rate is taken from the `StartFrame` during setup.
            stream_sid: The Acefone/Smartflo Media Stream SID. When not provided, it
                is captured from the incoming ``start`` event.
            call_sid: The associated Acefone/Smartflo Call SID. Required for auto
                hang-up.
            auto_hang_up: Whether to terminate the call (via a WebSocket ``end``
                event) when an `EndFrame` or `CancelFrame` is processed. Defaults
                to False.
            ignore_rtvi_messages: Inherited from `FrameSerializer.InputParams`,
                defaults to True.
        """

        media_format: AcefoneMediaFormat = AcefoneMediaFormat.ULAW
        acefone_sample_rate: int = 8000
        sample_rate: int = 0
        stream_sid: str | None = None
        call_sid: str | None = None
        auto_hang_up: bool = False

    def __init__(self, params: InputParams | None = None):
        """Initialize the `AcefoneFrameSerializer`.

        Args:
            params: Configuration parameters. `stream_sid` and `call_sid` are read
                from here; `call_sid` is required when `auto_hang_up` is enabled.
        """
        params = params or AcefoneFrameSerializer.InputParams()
        super().__init__(params)
        self._params: AcefoneFrameSerializer.InputParams = params

        self._stream_sid = self._params.stream_sid
        self._call_sid = self._params.call_sid
        self._auto_hang_up = self._params.auto_hang_up

        self._acefone_sample_rate = self._params.acefone_sample_rate
        self._sample_rate = 0  # Pipeline input rate; resolved in setup().

        self._input_resampler = create_stream_resampler(
            clear_after_secs=self._params.resampler_clear_after_secs
        )
        self._output_resampler = create_stream_resampler(
            clear_after_secs=self._params.resampler_clear_after_secs
        )
        self._hangup_attempted = False

    async def setup(self, frame: StartFrame):
        """Set up the serializer with pipeline configuration.

        Args:
            frame: The `StartFrame` containing pipeline configuration.
        """
        self._sample_rate = self._params.sample_rate or frame.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serialize a Pipecat frame to an Acefone WebSocket message.

        Handles conversion of the frame types relevant to the media stream:
        `AudioRawFrame` becomes a ``media`` event, `InterruptionFrame` becomes a
        ``clear`` event, and — when `auto_hang_up` is enabled — `EndFrame` and
        `CancelFrame` become an ``end`` event that terminates the call.

        Args:
            frame: The Pipecat frame to serialize.

        Returns:
            The serialized message as a JSON string, or None if the frame isn't
            handled.
        """
        if (
            self._params.auto_hang_up
            and not self._hangup_attempted
            and isinstance(frame, (EndFrame, CancelFrame))
        ):
            # Hang up the call using the WebSocket "end" event.
            self._hangup_attempted = True
            answer = {"event": "end", "streamSid": self._stream_sid, "call_sid": self._call_sid}
            return json.dumps(answer)
        elif isinstance(frame, InterruptionFrame):
            answer = {"event": "clear", "streamSid": self._stream_sid}
            return json.dumps(answer)
        elif isinstance(frame, AudioRawFrame):
            data = frame.audio

            # Output: convert the pipeline's PCM to Acefone's wire format at
            # acefone_sample_rate — either resampled PCM or 8kHz μ-law.
            if self._params.media_format == AcefoneMediaFormat.PCM:
                serialized_data = await self._output_resampler.resample(
                    data, frame.sample_rate, self._acefone_sample_rate
                )
            else:
                serialized_data = await pcm_to_ulaw(
                    data, frame.sample_rate, self._acefone_sample_rate, self._output_resampler
                )
            if serialized_data is None or len(serialized_data) == 0:
                # Ignoring in case we don't have audio.
                return None

            payload = base64.b64encode(serialized_data).decode("utf-8")
            answer: dict[str, Any] = {
                "event": "media",
                "streamSid": self._stream_sid,
                "media": {"payload": payload},
            }

            return json.dumps(answer)
        elif isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame):
                return None
            return json.dumps(frame.message)

        # Return None for unhandled frames.
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserialize Acefone WebSocket data into a Pipecat frame.

        Handles ``media`` events (audio), ``dtmf`` events (keypad input), and the
        ``start`` event (captures the stream SID).

        Args:
            data: The raw WebSocket message from Acefone.

        Returns:
            The corresponding Pipecat frame, or None if the event isn't handled.
        """
        message = json.loads(data)

        if message["event"] == "media":
            payload_base64 = message["media"]["payload"]
            payload = base64.b64decode(payload_base64)

            # Input: convert Acefone's wire audio to PCM at the pipeline input
            # rate — resample when it's already PCM, else decode 8kHz μ-law.
            if self._params.media_format == AcefoneMediaFormat.PCM:
                deserialized_data = await self._input_resampler.resample(
                    payload, self._acefone_sample_rate, self._sample_rate
                )
            else:
                deserialized_data = await ulaw_to_pcm(
                    payload, self._acefone_sample_rate, self._sample_rate, self._input_resampler
                )
            if deserialized_data is None or len(deserialized_data) == 0:
                # Ignoring in case we don't have audio.
                return None

            audio_frame = InputAudioRawFrame(
                audio=deserialized_data, num_channels=1, sample_rate=self._sample_rate
            )
            return audio_frame
        elif message["event"] == "dtmf":
            digit = message.get("dtmf", {}).get("digit")

            try:
                return InputDTMFFrame(KeypadEntry(digit))
            except ValueError:
                # The digit doesn't match any known keypad entry.
                logger.info(f"Invalid DTMF digit: {digit}")
                return None
        elif message["event"] == "start" and not self._stream_sid:
            self._stream_sid = message["streamSid"]

        return None
