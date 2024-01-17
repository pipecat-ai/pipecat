import asyncio
import logging
import re

from dailyai.queue_frame import QueueFrame, FrameType

from abc import abstractmethod
from typing import AsyncGenerator, Iterable
from dataclasses import dataclass
from typing import AsyncGenerator

from collections.abc import Iterable, AsyncIterable

class AIService:

    def __init__(
        self
    ):
        self.logger = logging.getLogger("dailyai")

    def stop(self):
        pass

    def allowed_input_frame_types(self) -> set[FrameType]:
        return set()

    def possible_output_frame_types(self) -> set[FrameType]:
        return set()

    async def run(
            self,
            requested_frame_types:set[FrameType],
            frames:Iterable[QueueFrame] | AsyncIterable[QueueFrame]
            ) -> AsyncGenerator[QueueFrame, None]:
        if self.possible_output_frame_types().intersection(requested_frame_types) == set():
            raise Exception(f"Requested frame types {requested_frame_types} are not supported by this service.")

        if isinstance(frames, AsyncIterable):
            async for frame in frames:
                output_frame: QueueFrame | None = await self.process_frame(requested_frame_types, frame)
                if output_frame:
                    yield output_frame
        elif isinstance(frames, Iterable):
            for frame in frames:
                output_frame = await self.process_frame(requested_frame_types, frame)
                if output_frame:
                    yield output_frame
        else:
            raise Exception("Frames must be an iterable or async iterable")

    @abstractmethod
    async def process_frame(self, requested_frame_types:set[FrameType], frame:QueueFrame) -> QueueFrame | None:
        pass

class SentenceAggregator(AIService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_sentence = ""

    def allowed_input_frame_types(self) -> set[FrameType]:
        return set([FrameType.TEXT_CHUNK, FrameType.SENTENCE])

    def possible_output_frame_types(self) -> set[FrameType]:
        return set([FrameType.SENTENCE])

    async def process_frame(self, requested_frame_types: set[FrameType], frame: QueueFrame) -> QueueFrame | None:
        if not FrameType.SENTENCE in requested_frame_types:
            return None

        if frame.frame_type == FrameType.TEXT_CHUNK:
            if type(frame.frame_data) != str:
                raise Exception("Sentence aggregator requires a string for the data field")

            self.current_sentence += frame.frame_data
            if self.current_sentence.endswith((".", "?", "!")):
                sentence = self.current_sentence
                self.current_sentence = ""
                return QueueFrame(FrameType.SENTENCE, sentence)
            return None
        elif frame.frame_type == FrameType.END_STREAM:
            if self.current_sentence:
                return QueueFrame(FrameType.SENTENCE, self.current_sentence)
            else:
                return None
        elif frame.frame_type == FrameType.SENTENCE:
            return frame
        else:
            return None


class LLMService(AIService):
    def allowed_input_frame_types(self) -> set[FrameType]:
        return set([FrameType.LLM_MESSAGE, FrameType.SENTENCE, FrameType.TRANSCRIPTION])

    def allowed_output_frame_types(self) -> set[FrameType]:
        return set([FrameType.SENTENCE, FrameType.SENTENCE, FrameType.TEXT_CHUNK])

    async def run_llm_async_sentences(self, messages) -> AsyncGenerator[str, None]:
        current_text = ""
        async for text in self.run_llm_async(messages):
            current_text += text
            if re.match(r"^.*[.!?]$", text):
                yield current_text
                current_text = ""

        if current_text:
            yield current_text

    async def process_frame(self, frame:QueueFrame) -> QueueFrame | None:
        if not self.output_queue:
            raise Exception("Output queue must be set before using the run method.")

        if frame.frame_type == FrameType.LLM_MESSAGE:
            if type(frame.frame_data) != list:
                raise Exception("LLM service requires a dict for the data field")

            messages: list[dict[str, str]] = frame.frame_data
            async for message in self.run_llm_async_sentences(messages):
                await self.output_queue.put(QueueFrame(FrameType.SENTENCE, message))


class TTSService(AIService):
    # Some TTS services require a specific sample rate. We default to 16k
    def get_mic_sample_rate(self):
        return 16000

    # Converts the sentence to audio. Yields a list of audio frames that can
    # be sent to the microphone device
    @abstractmethod
    async def run_tts(self, sentence) -> AsyncGenerator[bytes, None]:
        # yield empty bytes here, so linting can infer what this method does
        yield bytes()

    async def process_frame(self, frame:QueueFrame) -> QueueFrame | None:
        if not self.output_queue:
            raise Exception("Output queue must be set before using the run method.")

        if frame.frame_type == FrameType.SENTENCE:
            if type(frame.frame_data) != str:
                raise Exception("TTS service requires a string for the data field")

            text = frame.frame_data
            async for audio in self.run_tts(text):
                await self.output_queue.put(QueueFrame(FrameType.AUDIO, audio))


class ImageGenService(AIService):
    # Renders the image. Returns an Image object.
    @abstractmethod
    async def run_image_gen(self, sentence, size) -> tuple[str, bytes]:
        pass


@dataclass
class AIServiceConfig:
    tts: TTSService
    image: ImageGenService
    llm: LLMService
