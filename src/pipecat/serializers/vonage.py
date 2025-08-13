# SPDX-License-Identifier: BSD-2-Clause
"""Vonage WebSocket serializer (WAV+pydub resample, fixed-size chunking)."""

from __future__ import annotations

import io
import json
import wave
from typing import List, Optional, Union

from loguru import logger
from pydantic import BaseModel
from pydub import AudioSegment

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    StartInterruptionFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType

# ---- Audio/timing constants --------------------------------------------------

AUDIO_TARGET_RATE_HZ: int = 16_000  # 16 kHz target
AUDIO_CHANNELS_MONO: int = 1  # mono
PCM16_SAMPLE_WIDTH_BYTES: int = 2  # 16-bit PCM
SLEEP_INTERVAL: float = 0.01

BYTES_PER_SAMPLE_MONO: int = AUDIO_CHANNELS_MONO * PCM16_SAMPLE_WIDTH_BYTES
BYTES_PER_CHUNK: int = int(AUDIO_TARGET_RATE_HZ * SLEEP_INTERVAL) * BYTES_PER_SAMPLE_MONO


class VonageFrameSerializer(FrameSerializer):
    """Produces 16 kHz mono PCM chunks; resamples using WAV+pydub path."""

    class InputParams(BaseModel):
        """Configuration options for the Vonage frame serializer.

        Controls whether to send a clear-audio event and whether
        to auto-hang-up on End/Cancel frames.
        """

        auto_hang_up: bool = True
        send_clear_audio_event: bool = True

    def __init__(self, params: Optional[InputParams] = None) -> None:
        """Initialize the VonageFrameSerializer.

        Args:
            params: Optional configuration parameters for serialization.
        """
        self._params: VonageFrameSerializer.InputParams = (
            params or VonageFrameSerializer.InputParams()
        )
        self._sample_rate_hz: int = AUDIO_TARGET_RATE_HZ
        self._in_resampler = create_stream_resampler()
        self._out_resampler = create_stream_resampler()

        # Transport reads this for pacing (one sleep per chunk).
        self.sleep_interval: float = SLEEP_INTERVAL

        # Serializer-side audio format assumptions for pydub path:
        self._channels: int = AUDIO_CHANNELS_MONO
        self._sample_width_bytes: int = PCM16_SAMPLE_WIDTH_BYTES

    @property
    def type(self) -> FrameSerializerType:
        """Return the serializer type (binary frames)."""
        return FrameSerializerType.BINARY

    async def setup(self, frame: StartFrame) -> None:
        """Prepare the serializer for a new session.

        Sets the sample rate and sleep interval for chunk pacing.
        """
        self._sample_rate_hz = AUDIO_TARGET_RATE_HZ
        self.sleep_interval = SLEEP_INTERVAL

    # --- helpers --------------------------------------------------------------

    @staticmethod
    def _resample_audio_with_pydub(
        data: bytes,
        src_rate_hz: int,
        num_channels: int,
        sample_width_bytes: int,
        target_rate_hz: int,
    ) -> bytes:
        """Resample via WAV header + pydub.

        NOTE: This assumes `data` contains a WAV header. If your pipeline disables
        WAV headers, switch to a raw-PCM resampler instead.
        """
        with wave.open(io.BytesIO(data), "rb") as wf:
            num_frames = wf.getnframes()
            pcm_data = wf.readframes(num_frames)

        segment = AudioSegment.from_raw(
            io.BytesIO(pcm_data),
            sample_width=sample_width_bytes,
            frame_rate=src_rate_hz,
            channels=num_channels,
        )
        resampled = (
            segment.set_channels(num_channels)
            .set_sample_width(sample_width_bytes)
            .set_frame_rate(target_rate_hz)
        )
        return resampled.raw_data

    @staticmethod
    def _split_into_chunks(audio16: bytes) -> List[bytes]:
        return [audio16[i : i + BYTES_PER_CHUNK] for i in range(0, len(audio16), BYTES_PER_CHUNK)]

    # --- API ------------------------------------------------------------------

    async def serialize(self, frame: Frame) -> Optional[Union[str, bytes, list[bytes]]]:
        """Convert a Frame into one or more serialized payloads.

        Args:
            frame: The frame to serialize.

        Returns:
            The serialized data as a string, bytes, or list of bytes.
        """
        if self._params.auto_hang_up and isinstance(frame, (EndFrame, CancelFrame)):
            logger.debug(
                "VonageFrameSerializer: End/Cancel observed (auto-hang-up not implemented)."
            )
            return None

        if isinstance(frame, StartInterruptionFrame) and self._params.send_clear_audio_event:
            return json.dumps({"event": "clearAudio"})

        if isinstance(frame, OutputAudioRawFrame):
            audio16 = self._resample_audio_with_pydub(
                data=frame.audio,
                src_rate_hz=frame.sample_rate,
                num_channels=self._channels,
                sample_width_bytes=self._sample_width_bytes,
                target_rate_hz=self._sample_rate_hz,
            )
            return self._split_into_chunks(audio16)

        logger.debug(f"VonageFrameSerializer: ignoring frame type {type(frame).__name__}.")
        return None

    async def deserialize(self, data: Union[str, bytes]) -> Optional[Frame]:
        """Convert serialized input data into a Frame.

        Args:
            data: The raw audio or frame payload.

        Returns:
            The corresponding Frame instance, or None if parsing fails.
        """
        if isinstance(data, (bytes, bytearray)):
            audio = await self._in_resampler.resample(
                bytes(data), self._sample_rate_hz, self._sample_rate_hz
            )
            return InputAudioRawFrame(
                audio=audio,
                num_channels=AUDIO_CHANNELS_MONO,
                sample_rate=self._sample_rate_hz,
            )

        logger.info("VonageFrameSerializer: ignoring non-binary inbound data.")
        return None
