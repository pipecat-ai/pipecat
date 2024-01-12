import asyncio
import logging
import re

from dailyai.queue_frame import QueueFrame, FrameType

from abc import abstractmethod
from typing import AsyncGenerator
from dataclasses import dataclass

class AIService:

    def __init__(
        self,
        input_queue: asyncio.Queue[QueueFrame] | None = None,
        output_queue: asyncio.Queue[QueueFrame] | None = None,
    ):
        self.logger = logging.getLogger("dailyai")
        self.input_queue: asyncio.Queue[QueueFrame] | None = input_queue
        self.output_queue: asyncio.Queue[QueueFrame] | None = output_queue

    def stop(self):
        pass

    async def run(self) -> None:
        if self.input_queue is None or self.output_queue is None:
            raise Exception("Input and output queues must be set before using the run method.")

        while True:
            frame = await self.input_queue.get()
            self.logger.debug(f"{self.__class__.__name__} got frame:", frame.frame_type)
            if frame.frame_type == FrameType.END_STREAM:
                self.input_queue.task_done()
                await self.output_queue.put(QueueFrame(FrameType.END_STREAM, None))
                break

            output_frame = await self.process_frame(frame)
            if output_frame:
                await self.output_queue.put(output_frame)
            self.input_queue.task_done()

    @abstractmethod
    async def process_frame(self, frame) -> QueueFrame | None:
        pass


class LLMService(AIService):
    # Generate a set of responses to a prompt. Yields a list of responses.
    @abstractmethod
    async def run_llm_async(self, messages) -> AsyncGenerator[str, None]:
        # Adding a yield here lets the linter know what this method actually does
        yield ""

    # Generate a responses to a prompt. Returns the response
    @abstractmethod
    async def run_llm(
        self, messages
    ) -> str or None:
        pass

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

        if frame.frame_type == FrameType.LLM_MESSAGE_FRAME:
            if type(frame.frame_data) != list:
                raise Exception("LLM service requires a dict for the data field")

            messages: list[dict[str, str]] = frame.frame_data
            async for message in self.run_llm_async_sentences(messages):
                await self.output_queue.put(QueueFrame(FrameType.SENTENCE_FRAME, message))


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

        if frame.frame_type == FrameType.SENTENCE_FRAME:
            if type(frame.frame_data) != str:
                raise Exception("TTS service requires a string for the data field")

            text = frame.frame_data
            async for audio in self.run_tts(text):
                await self.output_queue.put(QueueFrame(FrameType.AUDIO_FRAME, audio))


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
