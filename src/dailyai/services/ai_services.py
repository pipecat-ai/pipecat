import array
import asyncio
import io
import logging
import math
import wave

from dailyai.queue_frame import (
    AudioQueueFrame,
    EndStreamQueueFrame,
    ImageQueueFrame,
    LLMMessagesQueueFrame,
    QueueFrame,
    TextQueueFrame,
)

from abc import abstractmethod
from typing import AsyncGenerator, AsyncIterable, BinaryIO, Iterable
from dataclasses import dataclass


class AIService:

    def __init__(self):
        self.logger = logging.getLogger("dailyai")

    def stop(self):
        pass

    async def run_to_queue(self, queue: asyncio.Queue, frames, add_end_of_stream=False) -> None:
        async for frame in self.run(frames):
            await queue.put(frame)

        if add_end_of_stream:
            await queue.put(EndStreamQueueFrame())

    async def run(
        self,
        frames: Iterable[QueueFrame]
        | AsyncIterable[QueueFrame]
        | asyncio.Queue[QueueFrame],
    ) -> AsyncGenerator[QueueFrame, None]:
        try:
            if isinstance(frames, AsyncIterable):
                async for frame in frames:
                    async for output_frame in self.process_frame(frame):
                        yield output_frame
            elif isinstance(frames, Iterable):
                for frame in frames:
                    async for output_frame in self.process_frame(frame):
                        yield output_frame
            elif isinstance(frames, asyncio.Queue):
                while True:
                    frame = await frames.get()
                    async for output_frame in self.process_frame(frame):
                        yield output_frame
                    if isinstance(frame, EndStreamQueueFrame):
                        break
            else:
                raise Exception("Frames must be an iterable or async iterable")

            async for output_frame in self.finalize():
                yield output_frame
        except Exception as e:
            self.logger.error("Exception occurred while running AI service", e)
            raise e

    @abstractmethod
    async def process_frame(self, frame:QueueFrame) -> AsyncGenerator[QueueFrame, None]:
        # This is a trick for the interpreter (and linter) to know that this is a generator.
        if False:
            yield QueueFrame()

    @abstractmethod
    async def finalize(self) -> AsyncGenerator[QueueFrame, None]:
        # This is a trick for the interpreter (and linter) to know that this is a generator.
        if False:
            yield QueueFrame()

class LLMService(AIService):
    @abstractmethod
    async def run_llm_async(self, messages) -> AsyncGenerator[str, None]:
        yield ""

    @abstractmethod
    async def run_llm(self, messages) -> str:
        pass

    async def process_frame(self, frame: QueueFrame) -> AsyncGenerator[QueueFrame, None]:
        if isinstance(frame, LLMMessagesQueueFrame):
            async for text_chunk in self.run_llm_async(frame.messages):
                yield TextQueueFrame(text_chunk)


class TTSService(AIService):
    def __init__(self, aggregate_sentences=True):
        super().__init__()
        self.aggregate_sentences: bool = aggregate_sentences
        self.current_sentence: str = ""

    # Some TTS services require a specific sample rate. We default to 16k
    def get_mic_sample_rate(self):
        return 16000

    # Converts the text to audio. Yields a list of audio frames that can
    # be sent to the microphone device
    @abstractmethod
    async def run_tts(self, text) -> AsyncGenerator[bytes, None]:
        # yield empty bytes here, so linting can infer what this method does
        yield bytes()

    async def process_frame(self, frame: QueueFrame) -> AsyncGenerator[QueueFrame, None]:
        if not isinstance(frame, TextQueueFrame):
            yield frame
            return

        text: str | None = None
        if not self.aggregate_sentences:
            text = frame.text
        else:
            self.current_sentence += frame.text
            if self.current_sentence.endswith((".", "?", "!")):
                text = self.current_sentence
                self.current_sentence = ""

        if text:
            async for audio_chunk in self.run_tts(text):
                yield AudioQueueFrame(audio_chunk)

    async def finalize(self):
        if self.current_sentence:
            async for audio_chunk in self.run_tts(self.current_sentence):
                yield AudioQueueFrame(audio_chunk)

    # Convenience function to send the audio for a sentence to the given queue
    async def say(self, sentence, queue: asyncio.Queue):
        await self.run_to_queue(queue, [TextQueueFrame(sentence)])


class ImageGenService(AIService):
    def __init__(self, image_size, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size

    # Renders the image. Returns an Image object.
    @abstractmethod
    async def run_image_gen(self, sentence:str) -> tuple[str, bytes]:
        pass

    async def process_frame(self, frame: QueueFrame) -> AsyncGenerator[QueueFrame, None]:
        if not isinstance(frame, TextQueueFrame):
            return

        (url, image_data) = await self.run_image_gen(frame.text)
        yield ImageQueueFrame(url, image_data)

class STTService(AIService):
    """STTService is a base class for speech-to-text services."""
    _content: io.BufferedRandom
    _wave: wave.Wave_write
    _silence_frames: int

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._new_wave()
        self._silence_frames = 0

    def _new_wave(self):
        """Creates a new wave object and content buffer."""
        self._content = io.BufferedRandom(io.BytesIO())
        ww = wave.open(self._content, "wb")
        ww.setnchannels(1)
        ww.setsampwidth(2)
        ww.setframerate(16000)
        self._wave = ww

    @abstractmethod
    def run_stt(self, audio: BinaryIO) -> str:
        """Returns transcript as a string"""
        pass

    async def process_frame(self, frame: QueueFrame) -> AsyncGenerator[QueueFrame, None]:
        """Processes a frame of audio data, either buffering or transcribing it."""
        if not isinstance(frame, AudioQueueFrame):
            return
        
        data = frame.data
        # Try to filter out empty background noise
        # (Very rudimentary approach, can be improved)
        volume = self._get_volume(data)
        if volume > 400:
            # If volume is high enough, write new data to wave file
            self._wave.writeframesraw(data)

        # If buffer is not empty and we detect a 3-frame pause in speech,
        # transcribe the audio gathered so far.
        if self._content.tell() > 0 and self._silence_frames > 3:
            self._silence_frames = 0
            self._wave.close()
            self._content.seek(0)
            text = self.run_stt(self._content)
            self._new_wave()
            yield TextQueueFrame(text)
        # If we get this far, this is a frame of silence
        self._silence_frames += 1

    def _get_volume(self, audio: bytes) -> float:
        # https://docs.python.org/3/library/array.html
        audio_array = array.array('h', audio) 
        squares = [sample**2 for sample in audio_array]
        mean = sum(squares) / len(audio_array)
        rms = math.sqrt(mean)
        return rms

@dataclass
class AIServiceConfig:
    tts: TTSService
    image: ImageGenService
    llm: LLMService
    stt: STTService
