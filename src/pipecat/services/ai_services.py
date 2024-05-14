#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import array
import io
import math
import wave

from abc import abstractmethod
from typing import AsyncGenerator, BinaryIO

from pipecat.frames.frames import (
    AudioRawFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    TextFrame,
    VisionImageRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class AIService(FrameProcessor):
    def __init__(self):
        super().__init__()

    async def process_generator(self, generator: AsyncGenerator[Frame, None]):
        async for f in generator:
            if isinstance(f, ErrorFrame):
                await self.push_error(f)
            else:
                await self.push_frame(f)


class LLMService(AIService):
    """This class is a no-op but serves as a base class for LLM services."""

    def __init__(self):
        super().__init__()


class TTSService(AIService):
    def __init__(self, aggregate_sentences: bool = True):
        super().__init__()
        self._aggregate_sentences: bool = aggregate_sentences
        self._current_sentence: str = ""

    # Converts the text to audio.
    @abstractmethod
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        pass

    async def say(self, text: str):
        await self.process_frame(TextFrame(text=text), FrameDirection.DOWNSTREAM)

    async def _process_text_frame(self, frame: TextFrame):
        text: str | None = None
        if not self._aggregate_sentences:
            text = frame.text
        else:
            self._current_sentence += frame.text
            if self._current_sentence.strip().endswith((".", "?", "!")):
                text = self._current_sentence
                self._current_sentence = ""

        if text:
            await self.process_generator(self.run_tts(text))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, TextFrame):
            await self._process_text_frame(frame)
        elif isinstance(frame, EndFrame):
            if self._current_sentence:
                await self.process_generator(self.run_tts(self._current_sentence))
            await self.push_frame(frame)
        else:
            await self.push_frame(frame, direction)


class STTService(AIService):
    """STTService is a base class for speech-to-text services."""

    def __init__(self,
                 min_rms: int = 400,
                 max_silence_frames: int = 3,
                 sample_rate: int = 16000,
                 num_channels: int = 1):
        super().__init__()
        self._min_rms = min_rms
        self._max_silence_frames = max_silence_frames
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._current_silence_frames = 0
        (self._content, self._wave) = self._new_wave()

    @abstractmethod
    async def run_stt(self, audio: BinaryIO) -> AsyncGenerator[Frame, None]:
        """Returns transcript as a string"""
        pass

    def _new_wave(self):
        content = io.BufferedRandom(io.BytesIO())
        ww = wave.open(content, "wb")
        ww.setsampwidth(2)
        ww.setnchannels(self._num_channels)
        ww.setframerate(self._sample_rate)
        return (content, ww)

    def _get_volume(self, audio: bytes) -> float:
        # https://docs.python.org/3/library/array.html
        audio_array = array.array('h', audio)
        squares = [sample**2 for sample in audio_array]
        mean = sum(squares) / len(audio_array)
        rms = math.sqrt(mean)
        return rms

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Processes a frame of audio data, either buffering or transcribing it."""
        if not isinstance(frame, AudioRawFrame):
            await self.push_frame(frame, direction)
            return

        audio = frame.audio

        # Try to filter out empty background noise
        # (Very rudimentary approach, can be improved)
        rms = self._get_volume(audio)
        if rms >= self._min_rms:
            # If volume is high enough, write new data to wave file
            self._wave.writeframes(audio)

        # If buffer is not empty and we detect a 3-frame pause in speech,
        # transcribe the audio gathered so far.
        if self._content.tell() > 0 and self._current_silence_frames > self._max_silence_frames:
            self._current_silence_frames = 0
            self._wave.close()
            self._content.seek(0)
            await self.process_generator(self.run_stt(self._content))
            (self._content, self._wave) = self._new_wave()
        # If we get this far, this is a frame of silence
        self._current_silence_frames += 1


class ImageGenService(AIService):

    def __init__(self):
        super().__init__()

    # Renders the image. Returns an Image object.
    @abstractmethod
    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, TextFrame):
            await self.process_generator(self.run_image_gen(frame.text))
        else:
            await self.push_frame(frame, direction)


class VisionService(AIService):
    """VisionService is a base class for vision services."""

    def __init__(self):
        super().__init__()
        self._describe_text = None

    @abstractmethod
    async def run_vision(self, frame: VisionImageRawFrame) -> AsyncGenerator[Frame, None]:
        pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, VisionImageRawFrame):
            await self.process_generator(self.run_vision(frame))
        else:
            await self.push_frame(frame, direction)
