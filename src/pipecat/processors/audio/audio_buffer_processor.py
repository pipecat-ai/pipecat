#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import time
from typing import Optional

from pipecat.audio.utils import create_default_resampler, interleave_stereo_audio, mix_audio
from pipecat.frames.frames import (
    AudioRawFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class AudioBufferProcessor(FrameProcessor):
    """Processes and buffers audio frames from both input (user) and output (bot) sources.

    This processor manages audio buffering and synchronization, providing both merged and
    track-specific audio access through event handlers. It supports various audio configurations
    including sample rate conversion and mono/stereo output.

    Events:
        on_audio_data: Triggered when buffer_size is reached, providing merged audio
        on_track_audio_data: Triggered when buffer_size is reached, providing separate tracks
        on_user_turn_audio_data: Triggered when user turn has ended, providing that user turn's audio
        on_bot_turn_audio_data: Triggered when bot turn has ended, providing that bot turn's audio

    Args:
        sample_rate (Optional[int]): Desired output sample rate. If None, uses source rate
        num_channels (int): Number of channels (1 for mono, 2 for stereo). Defaults to 1
        buffer_size (int): Size of buffer before triggering events. 0 for no buffering
        enable_turn_audio (bool): Whether turn audio event handlers should be triggered

    Audio handling:
        - Mono output (num_channels=1): User and bot audio are mixed
        - Stereo output (num_channels=2): User audio on left, bot audio on right
        - Automatic resampling of incoming audio to match desired sample_rate
        - Silence insertion for non-continuous audio streams
        - Buffer synchronization between user and bot audio
    """

    def __init__(
        self,
        *,
        sample_rate: Optional[int] = None,
        num_channels: int = 1,
        buffer_size: int = 0,
        user_continuous_stream: Optional[bool] = None,
        enable_turn_audio: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._init_sample_rate = sample_rate
        self._sample_rate = 0
        self._audio_buffer_size_1s = 0
        self._num_channels = num_channels
        self._buffer_size = buffer_size
        self._enable_turn_audio = enable_turn_audio

        if user_continuous_stream is not None:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "Parameter `user_continuous_stream` is deprecated.",
                    DeprecationWarning,
                )

        self._user_audio_buffer = bytearray()
        self._bot_audio_buffer = bytearray()

        self._user_speaking = False
        self._bot_speaking = False
        self._user_turn_audio_buffer = bytearray()
        self._bot_turn_audio_buffer = bytearray()

        # Intermittent (non continous user stream variables)
        self._last_user_frame_at = 0
        self._last_bot_frame_at = 0

        self._recording = False

        self._resampler = create_default_resampler()

        self._register_event_handler("on_audio_data")
        self._register_event_handler("on_track_audio_data")
        self._register_event_handler("on_user_turn_audio_data")
        self._register_event_handler("on_bot_turn_audio_data")

    @property
    def sample_rate(self) -> int:
        """Current sample rate of the audio processor.

        Returns:
            int: The sample rate in Hz
        """
        return self._sample_rate

    @property
    def num_channels(self) -> int:
        """Number of channels in the audio output.

        Returns:
            int: Number of channels (1 for mono, 2 for stereo)
        """
        return self._num_channels

    def has_audio(self) -> bool:
        """Check if both user and bot audio buffers contain data.

        Returns:
            bool: True if both buffers contain audio data
        """
        return self._buffer_has_audio(self._user_audio_buffer) and self._buffer_has_audio(
            self._bot_audio_buffer
        )

    def merge_audio_buffers(self) -> bytes:
        """Merge user and bot audio buffers into a single audio stream.

        For mono output, audio is mixed. For stereo output, user audio is placed
        on the left channel and bot audio on the right channel.

        Returns:
            bytes: Mixed audio data
        """
        if self._num_channels == 1:
            return mix_audio(bytes(self._user_audio_buffer), bytes(self._bot_audio_buffer))
        elif self._num_channels == 2:
            return interleave_stereo_audio(
                bytes(self._user_audio_buffer), bytes(self._bot_audio_buffer)
            )
        else:
            return b""

    async def start_recording(self):
        """Start recording audio from both user and bot.

        Initializes recording state and resets audio buffers.
        """
        self._recording = True
        self._reset_recording()

    async def stop_recording(self):
        """Stop recording and trigger final audio data handlers.

        Calls audio handlers with any remaining buffered audio before stopping.
        """
        await self._call_on_audio_data_handler()
        self._recording = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming audio frames and manage audio buffers."""
        await super().process_frame(frame, direction)

        # Update output sample rate if necessary.
        if isinstance(frame, StartFrame):
            self._update_sample_rate(frame)

        if self._recording:
            await self._process_recording(frame)
            if self._enable_turn_audio:
                await self._process_turn_recording(frame)

        if isinstance(frame, (CancelFrame, EndFrame)):
            await self.stop_recording()

        await self.push_frame(frame, direction)

    def _update_sample_rate(self, frame: StartFrame):
        self._sample_rate = self._init_sample_rate or frame.audio_out_sample_rate
        self._audio_buffer_size_1s = self._sample_rate * 2

    async def _process_recording(self, frame: Frame):
        if isinstance(frame, InputAudioRawFrame):
            # Add silence if we need to.
            silence = self._compute_silence(self._last_user_frame_at)
            self._user_audio_buffer.extend(silence)
            # Add user audio.
            resampled = await self._resample_audio(frame)
            self._user_audio_buffer.extend(resampled)
            # Save time of frame so we can compute silence.
            self._last_user_frame_at = time.time()
        elif self._recording and isinstance(frame, OutputAudioRawFrame):
            # Add silence if we need to.
            silence = self._compute_silence(self._last_bot_frame_at)
            self._bot_audio_buffer.extend(silence)
            # Add bot audio.
            resampled = await self._resample_audio(frame)
            self._bot_audio_buffer.extend(resampled)
            # Save time of frame so we can compute silence.
            self._last_bot_frame_at = time.time()

        if self._buffer_size > 0 and len(self._user_audio_buffer) > self._buffer_size:
            await self._call_on_audio_data_handler()

    async def _process_turn_recording(self, frame: Frame):
        if isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking = True
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._call_event_handler(
                "on_user_turn_audio_data", self._user_turn_audio_buffer, self.sample_rate, 1
            )
            self._user_speaking = False
            self._user_turn_audio_buffer = bytearray()
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._call_event_handler(
                "on_bot_turn_audio_data", self._bot_turn_audio_buffer, self.sample_rate, 1
            )
            self._bot_speaking = False
            self._bot_turn_audio_buffer = bytearray()

        if isinstance(frame, InputAudioRawFrame):
            resampled = await self._resample_audio(frame)
            self._user_turn_audio_buffer += resampled
            # In the case of the user, we need to keep a short buffer of audio
            # since VAD notification of when the user starts speaking comes
            # later.
            if (
                not self._user_speaking
                and len(self._user_turn_audio_buffer) > self._audio_buffer_size_1s
            ):
                discarded = len(self._user_turn_audio_buffer) - self._audio_buffer_size_1s
                self._user_turn_audio_buffer = self._user_turn_audio_buffer[discarded:]
        elif self._bot_speaking and isinstance(frame, OutputAudioRawFrame):
            resampled = await self._resample_audio(frame)
            self._bot_turn_audio_buffer += resampled

    async def _call_on_audio_data_handler(self):
        if not self.has_audio() or not self._recording:
            return

        # Call original handler with merged audio
        merged_audio = self.merge_audio_buffers()
        await self._call_event_handler(
            "on_audio_data", merged_audio, self._sample_rate, self._num_channels
        )

        # Call new handler with separate tracks
        await self._call_event_handler(
            "on_track_audio_data",
            bytes(self._user_audio_buffer),
            bytes(self._bot_audio_buffer),
            self._sample_rate,
            self._num_channels,
        )

        self._reset_audio_buffers()

    def _buffer_has_audio(self, buffer: bytearray) -> bool:
        return buffer is not None and len(buffer) > 0

    def _reset_recording(self):
        self._reset_audio_buffers()
        self._last_user_frame_at = time.time()
        self._last_bot_frame_at = time.time()

    def _reset_audio_buffers(self):
        self._user_audio_buffer = bytearray()
        self._bot_audio_buffer = bytearray()
        self._user_turn_audio_buffer = bytearray()
        self._bot_turn_audio_buffer = bytearray()

    async def _resample_audio(self, frame: AudioRawFrame) -> bytes:
        return await self._resampler.resample(frame.audio, frame.sample_rate, self._sample_rate)

    def _compute_silence(self, from_time: float) -> bytes:
        quiet_time = time.time() - from_time
        # We should get audio frames very frequently. We introduce silence only
        # if there's a big enough gap of 1s.
        if from_time == 0 or quiet_time < 1.0:
            return b""
        num_bytes = int(quiet_time * self._sample_rate) * 2
        silence = b"\x00" * num_bytes
        return silence
