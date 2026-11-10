#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Wire format of the Deepgram Voice Agent API.

The Voice Agent API (``/v1/agent/converse``) carries raw linear16 PCM audio as
binary WebSocket messages and everything else as JSON text messages. This
serializer handles the audio itself and passes the JSON through as transport
messages, which :class:`~pipecat.extensions.deepgram_voice_agent.processor.DeepgramVoiceAgentProcessor`
interprets.
"""

import json

from loguru import logger

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer


class DeepgramVoiceAgentSerializer(FrameSerializer):
    """Serializer for the Deepgram Voice Agent API wire format.

    Binary messages are mono linear16 PCM at the rates the client chose in its
    ``Settings`` message, resampled to and from the pipeline's rates. Text
    messages become :class:`~pipecat.frames.frames.InputTransportMessageFrame`
    with the parsed JSON object as ``message``, or the raw text when it isn't
    a JSON object. Outgoing transport messages are sent as JSON text.
    """

    def __init__(
        self,
        *,
        client_in_sample_rate: int,
        client_out_sample_rate: int,
        pipeline_in_sample_rate: int,
        **kwargs,
    ):
        """Initialize the serializer.

        Args:
            client_in_sample_rate: Sample rate of the audio the client sends.
            client_out_sample_rate: Sample rate the client wants audio in.
            pipeline_in_sample_rate: Sample rate of the pipeline's input audio.
            **kwargs: Additional arguments passed to :class:`FrameSerializer`.
        """
        super().__init__(**kwargs)
        self._client_in_sample_rate = client_in_sample_rate
        self._client_out_sample_rate = client_out_sample_rate
        self._pipeline_in_sample_rate = pipeline_in_sample_rate
        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()
        # A client may split a 16-bit sample across two messages.
        self._input_remainder = b""

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serialize outgoing audio and transport messages.

        Args:
            frame: The frame to serialize.

        Returns:
            PCM bytes for audio, JSON text for transport messages, otherwise None.
        """
        if isinstance(frame, OutputAudioRawFrame):
            audio = frame.audio
            if frame.sample_rate != self._client_out_sample_rate:
                audio = await self._output_resampler.resample(
                    audio, frame.sample_rate, self._client_out_sample_rate
                )
            return audio or None
        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame):
                return None
            return json.dumps(frame.message)
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserialize client audio and JSON messages.

        Args:
            data: The binary or text WebSocket message.

        Returns:
            An audio frame, a transport message frame, or None for empty audio.
        """
        if isinstance(data, (bytes, bytearray)):
            audio = self._input_remainder + bytes(data)
            split = len(audio) - len(audio) % 2
            audio, self._input_remainder = audio[:split], audio[split:]
            if not audio:
                return None
            if self._client_in_sample_rate != self._pipeline_in_sample_rate:
                audio = await self._input_resampler.resample(
                    audio, self._client_in_sample_rate, self._pipeline_in_sample_rate
                )
            return InputAudioRawFrame(
                audio=audio, sample_rate=self._pipeline_in_sample_rate, num_channels=1
            )

        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            logger.debug(f"{self} received a text message that isn't JSON: {data!r}")
            message = data
        return InputTransportMessageFrame(message=message)
