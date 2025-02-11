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
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class AudioBufferProcessor(FrameProcessor):
    """This processor buffers audio raw frames (input and output). The mixed
    audio can be obtained by calling `get_audio()` (if `buffer_size` is 0) or by
    registering an "on_audio_data" event handler. The event handler will be
    called every time `buffer_size` is reached.

    You can provide the desired output `sample_rate` and incoming audio frames
    will resampled to match it. Also, you can provide the number of channels, 1
    for mono and 2 for stereo. With mono audio user and bot audio will be mixed,
    in the case of stereo the left channel will be used for the user's audio and
    the right channel for the bot.

    Most of the time, user audio will be a continuous stream but it's possible
    that in some cases only the spoken audio is sent. To accomodate for those
    cases make sure to set `user_continuous_stream` accordingly.

    """

    def __init__(
        self,
        *,
        sample_rate: Optional[int] = None,
        num_channels: int = 1,
        buffer_size: int = 0,
        user_continuous_stream: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._init_sample_rate = sample_rate
        self._sample_rate = 0
        self._num_channels = num_channels
        self._buffer_size = buffer_size
        self._user_continuous_stream = user_continuous_stream

        self._user_audio_buffer = bytearray()
        self._bot_audio_buffer = bytearray()

        # Intermittent (non continous user stream variables)
        self._last_user_frame_at = 0
        self._last_bot_frame_at = 0

        self._recording = False

        self._resampler = create_default_resampler()

        self._register_event_handler("on_audio_data")

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def num_channels(self) -> int:
        return self._num_channels

    def has_audio(self) -> bool:
        return self._buffer_has_audio(self._user_audio_buffer) and self._buffer_has_audio(
            self._bot_audio_buffer
        )

    def merge_audio_buffers(self) -> bytes:
        if self._num_channels == 1:
            return mix_audio(bytes(self._user_audio_buffer), bytes(self._bot_audio_buffer))
        elif self._num_channels == 2:
            return interleave_stereo_audio(
                bytes(self._user_audio_buffer), bytes(self._bot_audio_buffer)
            )
        else:
            return b""

    async def start_recording(self):
        self._recording = True
        self._reset_recording()

    async def stop_recording(self):
        await self._call_on_audio_data_handler()
        self._recording = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Update output sample rate if necessary.
        if isinstance(frame, StartFrame):
            self._update_sample_rate(frame)

        if self._recording:
            if self._user_continuous_stream:
                await self._handle_continuous_stream(frame)
            else:
                await self._handle_intermittent_stream(frame)

        if self._buffer_size > 0 and len(self._user_audio_buffer) > self._buffer_size:
            await self._call_on_audio_data_handler()

        if isinstance(frame, (CancelFrame, EndFrame)):
            await self.stop_recording()

        await self.push_frame(frame, direction)

    def _update_sample_rate(self, frame: StartFrame):
        self._sample_rate = self._init_sample_rate or frame.audio_out_sample_rate

    async def _handle_continuous_stream(self, frame: Frame):
        if isinstance(frame, InputAudioRawFrame):
            # Add user audio.
            resampled = await self._resample_audio(frame)
            self._user_audio_buffer.extend(resampled)
            # Sync the bot's buffer to the user's buffer by adding silence if needed
            if len(self._user_audio_buffer) > len(self._bot_audio_buffer):
                silence_size = len(self._user_audio_buffer) - len(self._bot_audio_buffer)
                silence = b"\x00" * silence_size
                self._bot_audio_buffer.extend(silence)
        elif self._recording and isinstance(frame, OutputAudioRawFrame):
            # Add bot audio.
            resampled = await self._resample_audio(frame)
            self._bot_audio_buffer.extend(resampled)

    async def _handle_intermittent_stream(self, frame: Frame):
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

    async def _call_on_audio_data_handler(self):
        if not self.has_audio() or not self._recording:
            return

        merged_audio = self.merge_audio_buffers()
        await self._call_event_handler(
            "on_audio_data", merged_audio, self._sample_rate, self._num_channels
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
