#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Mock TTS service for testing XMLFunctionTagFilter integration."""

import asyncio
from typing import AsyncGenerator, Optional

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.services.tts_service import TTSService


class MockTTSService(TTSService):
    """Mock TTS service that generates predictable audio frames for testing."""

    def __init__(
        self,
        *,
        mock_audio_data: Optional[bytes] = None,
        chunk_size: int = 1024,
        frame_delay: float = 0.01,
        **kwargs
    ):
        """Initialize mock TTS service.

        Args:
            mock_audio_data: Bytes to use as fake audio
            chunk_size: Size of each audio frame chunk
            frame_delay: Delay between audio frames for realistic timing
            **kwargs: Additional args
        """
        super().__init__(**kwargs)

        self._mock_audio_data = mock_audio_data or self.create_mock_audio(1000)
        self._chunk_size = chunk_size
        self._frame_delay = frame_delay
        self.received_texts = []

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate mock audio frames for given text.

        Args:
            text: The text to convert (after filtering)

        Yields:
            Frames simulating real TTS service behavior
        """
        self.received_texts.append(text)

        yield TTSStartedFrame()

        if text.strip():
            for i in range(0, len(self._mock_audio_data), self._chunk_size):
                chunk = self._mock_audio_data[i : i + self._chunk_size]
                if chunk:
                    audio_frame = TTSAudioRawFrame(
                        audio=chunk, sample_rate=24000, num_channels=1
                    )
                    yield audio_frame
                    if self._frame_delay > 0:
                        await asyncio.sleep(self._frame_delay)

        yield TTSStoppedFrame()
        yield TTSTextFrame(text=text)

    @staticmethod
    def create_mock_audio(duration_ms: int, sample_rate: int = 24000) -> bytes:
        """Helper to create mock audio data of specific duration.

        Args:
            duration_ms: Duration in milliseconds
            sample_rate: Audio sample rate

        Returns:
            Bytes representing mock audio data
        """
        samples = int(duration_ms * sample_rate / 1000)

        audio_data = bytearray()
        for i in range(samples):
            # Simple pattern that creates audio-like data
            value = int(32767 * 0.3 * (i % 100) / 100)
            audio_data.extend(value.to_bytes(2, byteorder="little", signed=True))

        return bytes(audio_data)


class PredictableMockTTSService(MockTTSService):
    """Mock TTS that generates predictable audio based on input text."""

    def __init__(self, **kwargs):
        """Initialize PredictableMockTTSService with deterministic audio generation."""
        super().__init__(**kwargs)

    def get_audio_for_text(self, text: str) -> bytes:
        """Get deterministic audio bytes for given text input."""
        return self.create_deterministic_audio_from_text(text)

    @staticmethod
    def create_deterministic_audio_from_text(text: str) -> bytes:
        """Create deterministic audio bytes from text content."""
        # use text hash + length
        text_hash = hash(text) % 1000
        text_len = len(text)

        audio_data = bytearray()
        for i in range(text_len * 100):  # 100 bytes per character
            # Pattern based on text hash and position
            value = (text_hash + i) % 256
            audio_data.append(value)

        return bytes(audio_data)

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate specific audio frames for the filtered text."""

        self.received_texts.append(text)
        audio_data = self.get_audio_for_text(text)

        yield TTSStartedFrame()

        if text.strip():
            # Split audio into chunks and yield audio frames
            chunk_size = 1024
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                if chunk:
                    yield TTSAudioRawFrame(
                        audio=chunk, sample_rate=24000, num_channels=1
                    )

        yield TTSStoppedFrame()
        yield TTSTextFrame(text=text)