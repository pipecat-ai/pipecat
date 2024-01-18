import asyncio
import logging
import re

from httpx import request

from dailyai.queue_frame import QueueFrame, FrameType

from abc import abstractmethod
from typing import AsyncGenerator, Iterable
from dataclasses import dataclass
from typing import AsyncGenerator

from collections.abc import Iterable, AsyncIterable

class AIService:

    def __init__(self):
        self.logger = logging.getLogger("dailyai")

    def stop(self):
        pass

    def allowed_input_frame_types(self) -> set[FrameType]:
        return set()

    def possible_output_frame_types(self) -> set[FrameType]:
        return set()

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
        requested_frame_types: set[FrameType] | None=None,
    ) -> AsyncGenerator[QueueFrame, None]:
        if requested_frame_types and self.possible_output_frame_types().intersection(requested_frame_types) == set():
            raise Exception(f"Requested frame types {requested_frame_types} are not supported by this service.")

        if not requested_frame_types:
            requested_frame_types = self.possible_output_frame_types()

        if isinstance(frames, AsyncIterable):
            async for frame in frames:
                async for output_frame in self.process_frame(requested_frame_types, frame):
                    yield output_frame
        elif isinstance(frames, Iterable):
            for frame in frames:
                async for output_frame in self.process_frame(requested_frame_types, frame):
                    yield output_frame
        elif isinstance(frames, asyncio.Queue):
            while True:
                frame = await frames.get()
                async for output_frame in self.process_frame(requested_frame_types, frame):
                    yield output_frame
                if frame.frame_type == FrameType.END_STREAM:
                    break
        else:
            raise Exception("Frames must be an iterable or async iterable")

    @abstractmethod
    async def process_frame(self, requested_frame_types:set[FrameType], frame:QueueFrame) -> AsyncGenerator[QueueFrame, None]:
        # Yield something so the linter can deduce what should happen here.
        yield QueueFrame(FrameType.END_STREAM, None)

class SentenceAggregator(AIService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_sentence = ""

    def allowed_input_frame_types(self) -> set[FrameType]:
        return set([FrameType.TEXT_CHUNK, FrameType.SENTENCE])

    def possible_output_frame_types(self) -> set[FrameType]:
        return set([FrameType.SENTENCE])

    async def process_frame(self, requested_frame_types: set[FrameType], frame: QueueFrame) -> AsyncGenerator[QueueFrame, None]:
        if not FrameType.SENTENCE in requested_frame_types:
            return

        if frame.frame_type == FrameType.TEXT_CHUNK:
            if type(frame.frame_data) != str:
                raise Exception(
                    "Sentence aggregator requires a string for the data field"
                )

            self.current_sentence += frame.frame_data
            if self.current_sentence.endswith((".", "?", "!")):
                sentence = self.current_sentence
                self.current_sentence = ""
                yield QueueFrame(FrameType.SENTENCE, sentence)
        elif frame.frame_type == FrameType.END_STREAM:
            if self.current_sentence:
                yield QueueFrame(FrameType.SENTENCE, self.current_sentence)
        elif frame.frame_type == FrameType.SENTENCE:
            yield frame


class LLMService(AIService):
    def allowed_input_frame_types(self) -> set[FrameType]:
        return set([FrameType.LLM_MESSAGE, FrameType.SENTENCE, FrameType.TRANSCRIPTION])

    def allowed_output_frame_types(self) -> set[FrameType]:
        return set([FrameType.SENTENCE, FrameType.TEXT_CHUNK])

    @abstractmethod
    async def run_llm_async(self, messages) -> AsyncGenerator[str, None]:
        yield ""

    @abstractmethod
    async def run_llm(self, messages) -> str:
        pass

    async def process_frame(self, requested_frame_types: set[FrameType], frame: QueueFrame) -> AsyncGenerator[QueueFrame, None]:
        if frame.frame_type == FrameType.LLM_MESSAGE:
            if type(frame.frame_data) != list:
                raise Exception("LLM service requires a dict for the data field")

            messages: list[dict[str, str]] = frame.frame_data
            if FrameType.SENTENCE in requested_frame_types:
                yield QueueFrame(FrameType.SENTENCE, await self.run_llm(messages))
            else:
                async for text_chunk in self.run_llm_async(messages):
                    yield QueueFrame(FrameType.TEXT_CHUNK, text_chunk)

        # TODO: handle other frame types! Need to aggregate into messages


class TTSService(AIService):
    # Some TTS services require a specific sample rate. We default to 16k
    def get_mic_sample_rate(self):
        return 16000

    def allowed_input_frame_types(self) -> set[FrameType]:
        return set([FrameType.SENTENCE, FrameType.TRANSCRIPTION, FrameType.TEXT_CHUNK])

    def possible_output_frame_types(self) -> set[FrameType]:
        return set([FrameType.AUDIO])

    # Converts the sentence to audio. Yields a list of audio frames that can
    # be sent to the microphone device
    @abstractmethod
    async def run_tts(self, sentence) -> AsyncGenerator[bytes, None]:
        # yield empty bytes here, so linting can infer what this method does
        yield bytes()

    async def process_frame(self, requested_frame_types: set[FrameType], frame: QueueFrame) -> AsyncGenerator[QueueFrame, None]:
        if not FrameType.AUDIO in requested_frame_types:
            return

        if type(frame.frame_data) != str:
            raise Exception("TTS service requires a string for the data field")

        async for audio_chunk in self.run_tts(frame.frame_data):
            yield QueueFrame(FrameType.AUDIO, audio_chunk)

    # Convenience function to send the audio for a sentence to the given queue
    async def say(self, sentence, queue: asyncio.Queue):
        await self.run_to_queue(queue, [QueueFrame(FrameType.SENTENCE, sentence)])


class ImageGenService(AIService):
    def __init__(self, image_size, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size

    def allowed_input_frame_types(self) -> set[FrameType]:
        return set([FrameType.SENTENCE, FrameType.TRANSCRIPTION, FrameType.TEXT_CHUNK, FrameType.IMAGE_DESCRIPTION])

    def possible_output_frame_types(self) -> set[FrameType]:
        return set([FrameType.IMAGE])

    # Renders the image. Returns an Image object.
    @abstractmethod
    async def run_image_gen(self, sentence) -> tuple[str, bytes]:
        pass

    async def process_frame(self, requested_frame_types: set[FrameType], frame: QueueFrame) -> AsyncGenerator[QueueFrame, None]:
        if not FrameType.IMAGE in requested_frame_types:
            return

        if type(frame.frame_data) != str:
            raise Exception("Image service requires a string for the data field")

        (_, image_data) = await self.run_image_gen(frame.frame_data)
        yield QueueFrame(FrameType.IMAGE, image_data)


@dataclass
class AIServiceConfig:
    tts: TTSService
    image: ImageGenService
    llm: LLMService
