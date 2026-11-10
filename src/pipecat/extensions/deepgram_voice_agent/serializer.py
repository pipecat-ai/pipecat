#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Wire format of the Deepgram Voice Agent API.

The Voice Agent API (``/v1/agent/converse``) carries audio as binary WebSocket
messages, in the encodings the client chooses in ``Settings``, and everything
else as JSON text messages. This serializer converts the audio to and from the
pipeline's PCM and passes the JSON through as transport messages, which
:class:`~pipecat.extensions.deepgram_voice_agent.processor.DeepgramVoiceAgentProcessor`
interprets.
"""

import json
import struct

from loguru import logger

from pipecat.audio.utils import (
    alaw_to_pcm,
    create_stream_resampler,
    pcm_to_alaw,
    pcm_to_ulaw,
    ulaw_to_pcm,
)
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer

INPUT_ENCODINGS = {"linear16", "mulaw", "alaw"}
OUTPUT_ENCODINGS = {"linear16", "mulaw", "alaw"}


def _wav_header(sample_rate: int, encoding: str) -> bytes:
    """A WAV header for a stream of unknown length (sizes set to the maximum)."""
    format_code, bits = {"linear16": (1, 16), "alaw": (6, 8), "mulaw": (7, 8)}[encoding]
    block_align = bits // 8
    unknown = 0xFFFFFFFF
    return (
        b"RIFF"
        + struct.pack("<I", unknown)
        + b"WAVEfmt "
        + struct.pack(
            "<IHHIIHH",
            16,
            format_code,
            1,
            sample_rate,
            sample_rate * block_align,
            block_align,
            bits,
        )
        + b"data"
        + struct.pack("<I", unknown)
    )


class DeepgramVoiceAgentSerializer(FrameSerializer):
    """Serializer for the Deepgram Voice Agent API wire format.

    Binary messages are mono audio in the encodings and rates the client chose
    in its ``Settings`` message: ``linear16``, ``mulaw`` or ``alaw``, in and
    out. Output may use a ``wav`` container, whose header goes out ahead of the first audio.
    Text messages become :class:`~pipecat.frames.frames.InputTransportMessageFrame`
    with the parsed JSON object as ``message``, or the raw text when it isn't
    a JSON object. Outgoing transport messages are sent as JSON text.
    """

    def __init__(
        self,
        *,
        client_in_sample_rate: int,
        client_out_sample_rate: int,
        pipeline_in_sample_rate: int,
        input_encoding: str = "linear16",
        output_encoding: str = "linear16",
        output_container: str = "none",
        **kwargs,
    ):
        """Initialize the serializer.

        Args:
            client_in_sample_rate: Sample rate of the audio the client sends.
            client_out_sample_rate: Sample rate the client wants audio in.
            pipeline_in_sample_rate: Sample rate of the pipeline's input audio.
            input_encoding: Encoding of the audio the client sends.
            output_encoding: Encoding the client wants audio in.
            output_container: ``none``, or ``wav`` for a WAV header first.
            **kwargs: Additional arguments passed to :class:`FrameSerializer`.

        Raises:
            ValueError: If an encoding or container isn't supported.
        """
        super().__init__(**kwargs)
        if input_encoding not in INPUT_ENCODINGS:
            raise ValueError(f"Unsupported audio input encoding: {input_encoding}")
        if output_encoding not in OUTPUT_ENCODINGS:
            raise ValueError(f"Unsupported audio output encoding: {output_encoding}")
        if output_container not in ("none", "wav"):
            raise ValueError(f"Unsupported audio output container: {output_container}")
        self._client_in_sample_rate = client_in_sample_rate
        self._client_out_sample_rate = client_out_sample_rate
        self._pipeline_in_sample_rate = pipeline_in_sample_rate
        self._input_encoding = input_encoding
        self._output_encoding = output_encoding
        self._output_container = output_container
        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()
        # A client may split a 16-bit sample across two messages.
        self._input_remainder = b""
        self._header_sent = False

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serialize outgoing audio and transport messages.

        Args:
            frame: The frame to serialize.

        Returns:
            Encoded audio bytes, JSON text for transport messages, otherwise None.
        """
        if isinstance(frame, OutputAudioRawFrame):
            audio = await self._encode(frame)
            if not audio:
                return None
            if self._output_container == "wav" and not self._header_sent:
                self._header_sent = True
                audio = _wav_header(self._client_out_sample_rate, self._output_encoding) + audio
            return audio
        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame):
                return None
            return json.dumps(frame.message)
        return None

    async def _encode(self, frame: OutputAudioRawFrame) -> bytes:
        rate = self._client_out_sample_rate
        if self._output_encoding == "mulaw":
            return await pcm_to_ulaw(frame.audio, frame.sample_rate, rate, self._output_resampler)
        if self._output_encoding == "alaw":
            return await pcm_to_alaw(frame.audio, frame.sample_rate, rate, self._output_resampler)
        audio = frame.audio
        if frame.sample_rate != rate:
            audio = await self._output_resampler.resample(audio, frame.sample_rate, rate)
        return audio

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserialize client audio and JSON messages.

        Args:
            data: The binary or text WebSocket message.

        Returns:
            An audio frame, a transport message frame, or None for empty audio.
        """
        if isinstance(data, (bytes, bytearray)):
            audio = await self._decode(bytes(data))
            if not audio:
                return None
            return InputAudioRawFrame(
                audio=audio, sample_rate=self._pipeline_in_sample_rate, num_channels=1
            )

        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            logger.debug(f"{self} received a text message that isn't JSON: {data!r}")
            message = data
        return InputTransportMessageFrame(message=message)

    async def _decode(self, data: bytes) -> bytes:
        rate, pipeline_rate = self._client_in_sample_rate, self._pipeline_in_sample_rate
        if self._input_encoding == "mulaw":
            return await ulaw_to_pcm(data, rate, pipeline_rate, self._input_resampler)
        if self._input_encoding == "alaw":
            return await alaw_to_pcm(data, rate, pipeline_rate, self._input_resampler)
        audio = self._input_remainder + data
        split = len(audio) - len(audio) % 2
        audio, self._input_remainder = audio[:split], audio[split:]
        if audio and rate != pipeline_rate:
            audio = await self._input_resampler.resample(audio, rate, pipeline_rate)
        return audio
