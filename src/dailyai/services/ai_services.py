import asyncio
import io
import logging
import time
import wave

from dailyai.queue_frame import (
    AudioQueueFrame,
    EndStreamQueueFrame,
    ImageQueueFrame,
    LLMMessagesQueueFrame,
    LLMResponseEndQueueFrame,
    QueueFrame,
    TextQueueFrame,
    TranscriptionQueueFrame,
)

from abc import abstractmethod
from typing import AsyncGenerator, AsyncIterable, BinaryIO, Iterable

class AbstractPipeService:

    def __init__(
        self,
    ):
        self.out_queue: asyncio.Queue[QueueFrame] = asyncio.Queue()

    @abstractmethod
    async def process_queue(self):
        pass

class PipeService(AbstractPipeService):

    def __init__(
        self,
        source_queue: asyncio.Queue[QueueFrame] | None = None,
        out_queue: asyncio.Queue[QueueFrame] | None = None,
    ):
        super().__init__()
        self.logger: logging.Logger = logging.getLogger("dailyai")
        self.source_queue: asyncio.Queue[QueueFrame] = source_queue or asyncio.Queue()
        self.out_queue = out_queue or asyncio.Queue()

    async def process_queue(self):
        if not self.source_queue:
            return

        while True:
            frame: QueueFrame = await self.source_queue.get()
            print("got frame", frame.__class__.__name__)
            async for output_frame in self.process_frame(frame):
                if isinstance(frame, EndStreamQueueFrame):
                    async for final_frame in self.finalize():
                        await self.out_queue.put(final_frame)
                    await self.out_queue.put(output_frame)
                    return

                await self.out_queue.put(output_frame)

    @abstractmethod
    async def process_frame(self, frame: QueueFrame) -> AsyncGenerator[QueueFrame, None]:
        yield frame

    @abstractmethod
    async def finalize(self) -> AsyncGenerator[QueueFrame, None]:
        # This is a trick for the interpreter (and linter) to know that this is a generator.
        if False:
            yield QueueFrame()


class AIService(PipeService):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
            yield LLMResponseEndQueueFrame()
        else:
            yield frame


class TTSService(AIService):
    def __init__(self, aggregate_sentences=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
    async def say(self, sentence, queue: asyncio.Queue|None=None):
        queue = queue or self.out_queue
        if not queue:
            raise Exception("No queue to send audio to")
        await self.run_to_queue(queue, [TextQueueFrame(sentence)])


class ImageGenService(AIService):
    def __init__(self, image_size, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size

    # Renders the image. Returns an Image object.
    @abstractmethod
    async def run_image_gen(self, sentence: str) -> tuple[str, bytes]:
        pass

    async def process_frame(self, frame: QueueFrame) -> AsyncGenerator[QueueFrame, None]:
        if not isinstance(frame, TextQueueFrame):
            yield frame
            return

        (url, image_data) = await self.run_image_gen(frame.text)
        yield ImageQueueFrame(url, image_data)


class STTService(AIService):
    """STTService is a base class for speech-to-text services."""

    _frame_rate: int

    def __init__(self, frame_rate: int = 16000, **kwargs):
        super().__init__(**kwargs)
        self._frame_rate = frame_rate

    @abstractmethod
    async def run_stt(self, audio: BinaryIO) -> str:
        """Returns transcript as a string"""
        pass

    async def process_frame(self, frame: QueueFrame) -> AsyncGenerator[QueueFrame, None]:
        """Processes a frame of audio data, either buffering or transcribing it."""
        if not isinstance(frame, AudioQueueFrame):
            return

        data = frame.data
        content = io.BufferedRandom(io.BytesIO())
        ww = wave.open(self._content, "wb")
        ww.setnchannels(1)
        ww.setsampwidth(2)
        ww.setframerate(self._frame_rate)
        ww.writeframesraw(data)
        ww.close()
        content.seek(0)
        text = await self.run_stt(content)
        yield TranscriptionQueueFrame(text, '', str(time.time()))


class FrameLogger(AIService):
    def __init__(self, prefix="Frame", **kwargs):
        super().__init__(**kwargs)
        self.prefix = prefix

    async def process_frame(self, frame: QueueFrame) -> AsyncGenerator[QueueFrame, None]:
        if isinstance(frame, (AudioQueueFrame, ImageQueueFrame)):
            self.logger.info(f"{self.prefix}: {type(frame)}")
        else:
            print(f"{self.prefix}: {frame}")

        yield frame
