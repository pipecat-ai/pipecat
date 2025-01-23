#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from pipecat.audio.utils import interleave_stereo_audio, mix_audio, resample_audio
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
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

    """

    def __init__(
        self, *, sample_rate: int = 24000, num_channels: int = 1, buffer_size: int = 0, **kwargs
    ):
        super().__init__(**kwargs)
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._buffer_size = buffer_size

        self._user_audio_buffer = bytearray()
        self._bot_audio_buffer = bytearray()

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

    def reset_audio_buffers(self):
        self._user_audio_buffer = bytearray()
        self._bot_audio_buffer = bytearray()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Include all audio from the user.
        if isinstance(frame, InputAudioRawFrame):
            resampled = resample_audio(frame.audio, frame.sample_rate, self._sample_rate)
            self._user_audio_buffer.extend(resampled)
            # Sync the bot's buffer to the user's buffer by adding silence if needed
            if len(self._user_audio_buffer) > len(self._bot_audio_buffer):
                silence = b"\x00" * len(resampled)
                self._bot_audio_buffer.extend(silence)
        # If the bot is speaking, include all audio from the bot.
        elif isinstance(frame, OutputAudioRawFrame):
            resampled = resample_audio(frame.audio, frame.sample_rate, self._sample_rate)
            self._bot_audio_buffer.extend(resampled)

        if self._buffer_size > 0 and len(self._user_audio_buffer) > self._buffer_size:
            await self._call_on_audio_data_handler()

        if isinstance(frame, EndFrame):
            await self._call_on_audio_data_handler()

        await self.push_frame(frame, direction)

    async def _call_on_audio_data_handler(self):
        if not self.has_audio():
            return

        merged_audio = self.merge_audio_buffers()
        await self._call_event_handler(
            "on_audio_data", merged_audio, self._sample_rate, self._num_channels
        )
        self.reset_audio_buffers()

    def _buffer_has_audio(self, buffer: bytearray) -> bool:
        return buffer is not None and len(buffer) > 0
