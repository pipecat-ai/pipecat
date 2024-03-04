import asyncio
import re

from tblib import Frame
from dailyai.pipeline.frame_processor import FrameProcessor

from dailyai.pipeline.frames import (
    ControlQueueFrame,
    EndParallelPipeQueueFrame,
    EndStreamQueueFrame,
    LLMMessagesQueueFrame,
    LLMResponseEndQueueFrame,
    QueueFrame,
    TextQueueFrame,
    TranscriptionQueueFrame,
)
from dailyai.pipeline.pipeline import Pipeline
from dailyai.services.ai_services import AIService

from typing import AsyncGenerator, Coroutine, List, Text


class LLMContextAggregator(AIService):
    def __init__(
        self,
        messages: list[dict],
        role: str,
        bot_participant_id=None,
        complete_sentences=True,
        pass_through=True,
    ):
        super().__init__()
        self.messages = messages
        self.bot_participant_id = bot_participant_id
        self.role = role
        self.sentence = ""
        self.complete_sentences = complete_sentences
        self.pass_through = pass_through

    async def process_frame(
        self, frame: QueueFrame
    ) -> AsyncGenerator[QueueFrame, None]:
        # We don't do anything with non-text frames, pass it along to next in the pipeline.
        if not isinstance(frame, TextQueueFrame):
            yield frame
            return

        # Ignore transcription frames from the bot
        if isinstance(frame, TranscriptionQueueFrame):
            if frame.participantId == self.bot_participant_id:
                return

        # The common case for "pass through" is receiving frames from the LLM that we'll
        # use to update the "assistant" LLM messages, but also passing the text frames
        # along to a TTS service to be spoken to the user.
        if self.pass_through:
            yield frame

        # TODO: split up transcription by participant
        if self.complete_sentences:
            # type: ignore -- the linter thinks this isn't a TextQueueFrame, even
            # though we check it above
            self.sentence += frame.text
            if self.sentence.endswith((".", "?", "!")):
                self.messages.append({"role": self.role, "content": self.sentence})
                self.sentence = ""
                yield LLMMessagesQueueFrame(self.messages)
        else:
            # type: ignore -- the linter thinks this isn't a TextQueueFrame, even
            # though we check it above
            self.messages.append({"role": self.role, "content": frame.text})
            yield LLMMessagesQueueFrame(self.messages)

    async def finalize(self) -> AsyncGenerator[QueueFrame, None]:
        # Send any dangling words that weren't finished with punctuation.
        if self.complete_sentences and self.sentence:
            self.messages.append({"role": self.role, "content": self.sentence})
            yield LLMMessagesQueueFrame(self.messages)


class LLMUserContextAggregator(LLMContextAggregator):
    def __init__(
        self, messages: list[dict], bot_participant_id=None, complete_sentences=True
    ):
        super().__init__(
            messages, "user", bot_participant_id, complete_sentences, pass_through=False
        )


class LLMAssistantContextAggregator(LLMContextAggregator):
    def __init__(
        self, messages: list[dict], bot_participant_id=None, complete_sentences=True
    ):
        super().__init__(
            messages,
            "assistant",
            bot_participant_id,
            complete_sentences,
            pass_through=True,
        )


class SentenceAggregator(FrameProcessor):

    def __init__(self):
        self.aggregation = ""

    async def process_frame(
        self, frame: QueueFrame
    ) -> AsyncGenerator[QueueFrame, None]:
        if isinstance(frame, TextQueueFrame):
            m = re.search("(.*[?.!])(.*)", frame.text)
            if m:
                yield TextQueueFrame(self.aggregation + m.group(1))
                self.aggregation = m.group(2)
            else:
                self.aggregation += frame.text
        elif isinstance(frame, EndStreamQueueFrame):
            if self.aggregation:
                yield TextQueueFrame(self.aggregation)
            yield frame
        else:
            yield frame


class LLMFullResponseAggregator(FrameProcessor):
    def __init__(self):
        self.aggregation = ""

    async def process_frame(
        self, frame: QueueFrame
    ) -> AsyncGenerator[QueueFrame, None]:
        if isinstance(frame, TextQueueFrame):
            self.aggregation += frame.text
        elif isinstance(frame, LLMResponseEndQueueFrame):
            yield TextQueueFrame(self.aggregation)
            self.aggregation = ""
        else:
            yield frame



class StatelessTextTransformer(FrameProcessor):
    def __init__(self, transform_fn):
        self.transform_fn = transform_fn

    async def process_frame(self, frame: QueueFrame) -> AsyncGenerator[QueueFrame, None]:
        if isinstance(frame, TextQueueFrame):
            result = self.transform_fn(frame.text)
            if isinstance(result, Coroutine):
                result = await result

            yield TextQueueFrame(result)
        else:
            yield frame

class ParallelPipeline(FrameProcessor):
    def __init__(self, pipeline_definitions: List[List[FrameProcessor]]):
        self.sources = [asyncio.Queue() for _ in pipeline_definitions]
        self.sink: asyncio.Queue[QueueFrame] = asyncio.Queue()
        self.pipelines: list[Pipeline] = [
            Pipeline(source, self.sink, pipeline_definition)
            for source, pipeline_definition in zip(self.sources, pipeline_definitions)
        ]

    async def process_frame(self, frame: QueueFrame) -> AsyncGenerator[QueueFrame, None]:
        for source in self.sources:
            await source.put(frame)
            await source.put(EndParallelPipeQueueFrame())

        await asyncio.gather(*[pipeline.run_pipeline() for pipeline in self.pipelines])

        while not self.sink.empty():
            frame = await self.sink.get()
            # Skip passing along EndParallelPipeQueueFrame, because we use them for our own flow control.
            if not isinstance(frame, EndParallelPipeQueueFrame):
                yield frame

class GatedAggregator(FrameProcessor):
    def __init__(self, gate_open_fn, gate_close_fn, start_open):
        self.gate_open_fn = gate_open_fn
        self.gate_close_fn = gate_close_fn
        self.gate_open = start_open
        self.accumulator: List[QueueFrame] = []

    async def process_frame(self, frame: QueueFrame) -> AsyncGenerator[QueueFrame, None]:
        if self.gate_open:
            if self.gate_close_fn(frame):
                self.gate_open = False
        else:
            if self.gate_open_fn(frame):
                self.gate_open = True

        if self.gate_open:
            yield frame
            if self.accumulator:
                for frame in self.accumulator:
                    yield frame
            self.accumulator = []
        else:
            self.accumulator.append(frame)
