import asyncio
import logging
import re

from httpx import request

from dailyai.queue_frame import QueueFrame, FrameType

from abc import abstractmethod
from typing import AsyncGenerator, AsyncIterable, Iterable
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
            await queue.put(QueueFrame(FrameType.END_STREAM, None))

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
                    if frame.frame_type == FrameType.END_STREAM:
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
            yield QueueFrame(FrameType.NOOP, None)

    @abstractmethod
    async def finalize(self) -> AsyncGenerator[QueueFrame, None]:
        # This is a trick for the interpreter (and linter) to know that this is a generator.
        if False:
            yield QueueFrame(FrameType.NOOP, None)

class LLMService(AIService):
    def allowed_input_frame_types(self) -> set[FrameType]:
        return set([FrameType.LLM_MESSAGE])

    def allowed_output_frame_types(self) -> set[FrameType]:
        return set([FrameType.TEXT])

    @abstractmethod
    async def run_llm_async(self, messages) -> AsyncGenerator[str, None]:
        yield ""

    @abstractmethod
    async def run_llm(self, messages) -> str:
        pass

    async def process_frame(self, frame: QueueFrame) -> AsyncGenerator[QueueFrame, None]:
        if frame.frame_type == FrameType.LLM_MESSAGE:
            if type(frame.frame_data) != list:
                raise Exception("LLM service requires a dict for the data field")

            messages: list[dict[str, str]] = frame.frame_data
            async for text_chunk in self.run_llm_async(messages):
                yield QueueFrame(FrameType.TEXT, text_chunk)


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
        if frame.frame_type != FrameType.TEXT or type(frame.frame_data) != str:
            raise Exception(f"TTS service requires a string for the data field, got {frame.frame_type} and frame_data type {type(frame.frame_data)}")

        text: str | None = None
        if not self.aggregate_sentences:
            text = frame.frame_data
        else:
            self.current_sentence += frame.frame_data
            if self.current_sentence.endswith((".", "?", "!")):
                text = self.current_sentence
                self.current_sentence = ""

        if text:
            async for audio_chunk in self.run_tts(text):
                yield QueueFrame(FrameType.AUDIO, audio_chunk)

    async def finalize(self):
        if self.current_sentence:
            async for audio_chunk in self.run_tts(self.current_sentence):
                yield QueueFrame(FrameType.AUDIO, audio_chunk)

    # Convenience function to send the audio for a sentence to the given queue
    async def say(self, sentence, queue: asyncio.Queue):
        await self.run_to_queue(queue, [QueueFrame(FrameType.TEXT, sentence)])


class ImageGenService(AIService):
    def __init__(self, image_size, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size

    # Renders the image. Returns an Image object.
    @abstractmethod
    async def run_image_gen(self, sentence) -> tuple[str, bytes]:
        pass

    async def process_frame(self, frame: QueueFrame) -> AsyncGenerator[QueueFrame, None]:
        if type(frame.frame_data) != str:
            raise Exception("Image service requires a string for the data field")

        (_, image_data) = await self.run_image_gen(frame.frame_data)
        yield QueueFrame(FrameType.IMAGE, image_data)


@dataclass
class AIServiceConfig:
    tts: TTSService
    image: ImageGenService
    llm: LLMService
