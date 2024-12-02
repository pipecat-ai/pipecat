#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import wave
from io import BytesIO

from pipecat.audio.utils import interleave_stereo_audio, mix_audio, resample_audio
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class AudioBufferProcessor(FrameProcessor):
    """This processor buffers audio raw frames (input and output) that can later
    be obtained as an in-memory WAV. You can provide the desired output
    `sample_rate` and incoming audio frames will resampled to match it. Also,
    you can provide the number of channels, 1 for mono and 2 for stereo. With
    mono audio user and bot audio will be mixed, in the case of stereo the left
    channel will be used for the user's audio and the right channel for the bot.

    """

    def __init__(self, *, sample_rate: int = 24000, num_channels: int = 1, **kwargs):
        super().__init__(**kwargs)
        self._sample_rate = sample_rate
        self._num_channels = num_channels

        self._user_audio_buffer = bytearray()
        self._bot_audio_buffer = bytearray()

    def _buffer_has_audio(self, buffer: bytearray) -> bool:
        return buffer is not None and len(buffer) > 0

    def has_audio(self) -> bool:
        return self._buffer_has_audio(self._user_audio_buffer) and self._buffer_has_audio(
            self._bot_audio_buffer
        )

    def reset_audio_buffer(self):
        self._user_audio_buffer = bytearray()
        self._bot_audio_buffer = bytearray()

    def merge_audio_buffers(self) -> bytes:
        if self._num_channels == 1:
            return self._merge_mono()
        elif self._num_channels == 2:
            return self._merge_stereo()
        else:
            return b""

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
        if isinstance(frame, OutputAudioRawFrame):
            resampled = resample_audio(frame.audio, frame.sample_rate, self._sample_rate)
            self._bot_audio_buffer.extend(resampled)

    def _merge_mono(self) -> bytes:
        with BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self._sample_rate)
                mixed = mix_audio(bytes(self._user_audio_buffer), bytes(self._bot_audio_buffer))
                wf.writeframes(mixed)
            return buffer.getvalue()

    def _merge_stereo(self) -> bytes:
        with BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(self._sample_rate)
                stereo = interleave_stereo_audio(
                    bytes(self._user_audio_buffer), bytes(self._bot_audio_buffer)
                )
                wf.writeframes(stereo)
            return buffer.getvalue()
