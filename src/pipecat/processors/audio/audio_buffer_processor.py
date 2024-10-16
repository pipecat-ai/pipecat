#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import wave
from io import BytesIO

from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class AudioBufferProcessor(FrameProcessor):
    def __init__(self, **kwargs):
        """
        Initialize the AudioBufferProcessor.

        This constructor sets up the initial state for audio processing:
        - audio_buffer: A bytearray to store incoming audio data.
        - num_channels: The number of audio channels (initialized as None).
        - sample_rate: The sample rate of the audio (initialized as None).

        The num_channels and sample_rate are set to None initially and will be
        populated when the first audio frame is processed.
        """
        super().__init__(**kwargs)
        self._user_audio_buffer = bytearray()
        self._assistant_audio_buffer = bytearray()
        self._num_channels = None
        self._sample_rate = None

    def _buffer_has_audio(self, buffer: bytearray):
        return buffer is not None and len(buffer) > 0

    def has_audio(self):
        return (
            self._buffer_has_audio(self._user_audio_buffer)
            and self._buffer_has_audio(self._assistant_audio_buffer)
            and self._sample_rate is not None
        )

    def reset_audio_buffer(self):
        self._user_audio_buffer = bytearray()
        self._assistant_audio_buffer = bytearray()

    def merge_audio_buffers(self):
        with BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(self._sample_rate)
                # Interleave the two audio streams
                max_length = max(len(self._user_audio_buffer), len(self._assistant_audio_buffer))
                interleaved = bytearray(max_length * 2)

                for i in range(0, max_length, 2):
                    if i < len(self._user_audio_buffer):
                        interleaved[i * 2] = self._user_audio_buffer[i]
                        interleaved[i * 2 + 1] = self._user_audio_buffer[i + 1]
                    else:
                        interleaved[i * 2] = 0
                        interleaved[i * 2 + 1] = 0

                    if i < len(self._assistant_audio_buffer):
                        interleaved[i * 2 + 2] = self._assistant_audio_buffer[i]
                        interleaved[i * 2 + 3] = self._assistant_audio_buffer[i + 1]
                    else:
                        interleaved[i * 2 + 2] = 0
                        interleaved[i * 2 + 3] = 0

                wf.writeframes(interleaved)
            return buffer.getvalue()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, AudioRawFrame) and self._sample_rate is None:
            self._sample_rate = frame.sample_rate

        # include all audio from the user
        if isinstance(frame, InputAudioRawFrame):
            self._user_audio_buffer.extend(frame.audio)
            # Sync the assistant's buffer to the user's buffer by adding silence if needed
            if len(self._user_audio_buffer) > len(self._assistant_audio_buffer):
                silence_length = len(self._user_audio_buffer) - len(self._assistant_audio_buffer)
                silence = b"\x00" * silence_length
                self._assistant_audio_buffer.extend(silence)

        # if the assistant is speaking, include all audio from the assistant,
        if isinstance(frame, OutputAudioRawFrame):
            self._assistant_audio_buffer.extend(frame.audio)

        # do not push the user's audio frame, doing so will result in echo
        if not isinstance(frame, InputAudioRawFrame):
            await self.push_frame(frame, direction)
