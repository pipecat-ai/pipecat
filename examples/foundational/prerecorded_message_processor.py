#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Prerecorded message processor for playing audio instead of TTS."""

import wave

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class PrerecordedMessageProcessor(FrameProcessor):
    """Processor that intercepts specific LLM text and plays prerecorded audio.

    This processor checks incoming LLMTextFrame instances for a specific text pattern.
    When the pattern "Your pre-recorded message" is detected, it replaces the text
    with a prerecorded audio message by pushing LLMFullResponseStartFrame, the audio
    data as OutputAudioRawFrame, and LLMFullResponseEndFrame. Other frames pass through
    unchanged.

    Parameters:
        audio_file_path: Path to the WAV file containing the prerecorded message.

    Example::

        processor = PrerecordedMessageProcessor(
            audio_file_path="path/to/message.wav"
        )

        pipeline = Pipeline([
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            processor,  # Insert before TTS
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ])
    """

    def __init__(
        self,
        *,
        audio_file_path: str,
        **kwargs,
    ):
        """Initialize the prerecorded message processor.

        Args:
            audio_file_path: Path to the WAV file containing the prerecorded message.
            **kwargs: Additional arguments passed to FrameProcessor.
        """
        super().__init__(**kwargs)
        self._audio_file_path = audio_file_path
        self._audio_data = None
        self._sample_rate = None
        self._num_channels = None
        self._load_audio()

    def _load_audio(self) -> None:
        """Load the prerecorded audio file into memory."""
        try:
            with wave.open(self._audio_file_path, "rb") as wav_file:
                self._sample_rate = wav_file.getframerate()
                self._num_channels = wav_file.getnchannels()
                self._audio_data = wav_file.readframes(wav_file.getnframes())
        except Exception as e:
            raise ValueError(f"Failed to load audio file {self._audio_file_path}: {e}")

        # Ensure audio was loaded successfully
        if self._audio_data is None or self._sample_rate is None or self._num_channels is None:
            raise ValueError(f"Failed to load audio data from {self._audio_file_path}")

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process incoming frames and replace specific text with prerecorded audio.

        Args:
            frame: The frame to process.
            direction: Direction of the frame flow.
        """
        await super().process_frame(frame, direction)

        # Check if this is an LLMTextFrame with our trigger text
        if isinstance(frame, LLMTextFrame) and frame.text == "Your pre-recorded message":
            # Ensure audio data is loaded (should always be true after __init__)
            if self._audio_data is None or self._sample_rate is None or self._num_channels is None:
                raise RuntimeError("Audio data not loaded")

            # Push the prerecorded message sequence
            await self.push_frame(LLMFullResponseStartFrame(), direction)
            await self.push_frame(frame, direction)  # Keep the text frame for context

            # Push the prerecorded audio
            audio_frame = OutputAudioRawFrame(
                audio=self._audio_data,
                sample_rate=self._sample_rate,
                num_channels=self._num_channels,
            )
            await self.push_frame(audio_frame, direction)

            await self.push_frame(LLMFullResponseEndFrame(), direction)
        else:
            # Pass through all other frames unchanged
            await self.push_frame(frame, direction)
