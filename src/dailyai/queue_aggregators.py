import asyncio

from dailyai.queue_frame import EndStreamQueueFrame, LLMMessagesQueueFrame, QueueFrame, TextQueueFrame, TranscriptionQueueFrame
from dailyai.services.ai_services import AIService, PipeService

from typing import Any, AsyncGenerator, Callable, List, Tuple

@dataclass
class SinkAndQueuePair:
    sink: PipeService
    queue: asyncio.Queue


class QueueTee(PipeService):
    def __init__(
        self, source: PipeService, sinks: list[PipeService]
    ):
        self.source: PipeService = source

        self.sinks: List[SinkAndQueuePair] = []
        for sink in sinks:
            pair = SinkAndQueuePair()
            pair.sink = sink
            pair.queue = asyncio.Queue()
            self.sinks.append(pair)

    async def process_queue(self):
        while True:
            frame = await self.source.sink_queue.get()
            for sink in self.sinks:
                await sink.queue.put(frame)
            if isinstance(frame, EndStreamQueueFrame):
                break


class QueueGateOnFrame(PipeService):

    def __init__(
        self,
        source: PipeService,
        aggregator: Callable[[Any, QueueFrame], Tuple[Any, QueueFrame | None]],
    ):
        self.source = source
        self.aggregator = aggregator
        self.accumulation = None

    async def process_frame(
        self, frame: QueueFrame
    ) -> AsyncGenerator[QueueFrame, None]:
        output_frame: QueueFrame | None = None
        (self.aggregation, output_frame) = self.aggregator(
            self.aggregation, frame
        )
        if output_frame:
            yield output_frame

class QueueMergeGateOnFirst(PipeService):

    def __init__(
        self, sources: List[PipeService], sink: asyncio.Queue[QueueFrame]
    ):
        self.sources = sources
        self.sink = sink

    async def process_queue(self):
        (frames): list[QueueFrame] = await asyncio.gather(
            *[
                source.sink_queue.get() for source in self.sources
            ]
        )
        for idx, frame in enumerate(frames):
            await self.sink.put(frame)

            # if the first frame we got from a source is an EndStreamQueueFrame, remove that source
            if isinstance(frame, EndStreamQueueFrame):
                self.sources.pop(idx)

        async def pass_through(sink, source):
            while True:
                frame = await source.get()
                await sink.put(frame)
                if isinstance(frame, EndStreamQueueFrame):
                    break

        await asyncio.gather(
            *[pass_through(self.sink, source) for source in self.sources]
        )


class LLMContextAggregator(AIService):
    def __init__(
            self,
            messages: list[dict],
            role: str,
            bot_participant_id=None,
            complete_sentences=True,
            pass_through=True):
        super().__init__()
        self.messages = messages
        self.bot_participant_id = bot_participant_id
        self.role = role
        self.sentence = ""
        self.complete_sentences = complete_sentences
        self.pass_through = pass_through

    async def process_frame(self, frame: QueueFrame) -> AsyncGenerator[QueueFrame, None]:
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
    def __init__(self,
                 messages: list[dict],
                 bot_participant_id=None,
                 complete_sentences=True):
        super().__init__(messages, "user", bot_participant_id, complete_sentences, pass_through=False)


class LLMAssistantContextAggregator(LLMContextAggregator):
    def __init__(
        self, messages: list[dict], bot_participant_id=None, complete_sentences=True
    ):
        super().__init__(
            messages, "assistant", bot_participant_id, complete_sentences, pass_through=True
        )
