import asyncio

from attr import dataclass

from dailyai.queue_frame import (
    EndStreamQueueFrame,
    LLMMessagesQueueFrame,
    LLMResponseEndQueueFrame,
    QueueFrame,
    TextQueueFrame,
    TranscriptionQueueFrame,
)
from dailyai.services.ai_services import AIService, PipeService

from typing import Any, AsyncGenerator, Callable, List, Tuple


class QueueTee(PipeService):
    def __init__(self, out_services: list[PipeService], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.out_services: List[PipeService] = []
        for out_service in out_services:
            out_service.source_queue = asyncio.Queue()
            self.out_services.append(out_service)

    async def process_queue(self):
        if not self.source_queue:
            return

        while True:
            frame: QueueFrame = await self.source_queue.get()
            for out_service in self.out_services:
                if out_service.source_queue:
                    await out_service.source_queue.put(frame)

            if isinstance(frame, EndStreamQueueFrame):
                break


class QueueFrameAggregator(PipeService):
    def __init__(
        self,
        aggregator: Callable[[Any, QueueFrame], Tuple[Any, QueueFrame | None]],
        finalizer: Callable[[Any], QueueFrame | None],
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.aggregator = aggregator
        self.finalizer = finalizer
        self.aggregation = None

    async def process_frame(
        self, frame: QueueFrame
    ) -> AsyncGenerator[QueueFrame, None]:
        output_frame: QueueFrame | None = None
        (self.aggregation, output_frame) = self.aggregator(self.aggregation, frame)
        if output_frame:
            yield output_frame

    async def finalize(self) -> AsyncGenerator[QueueFrame, None]:
        output_frame = self.finalizer(self.aggregation)
        if output_frame:
            yield output_frame


class QueueMergeGateOnFirst(PipeService):

    def __init__(
        self,
        source_queues: List[asyncio.Queue[QueueFrame]],
        ungate_frame_class=LLMResponseEndQueueFrame,
    ):
        super().__init__()
        self.source_queues = source_queues
        self.ungate_frame_class = ungate_frame_class

    async def process_queue(self):
        while len(self.source_queues) > 0:
            (frames): list[QueueFrame] = await asyncio.gather(
                *[source_queue.get() for source_queue in self.source_queues]
            )
            for idx, frame in enumerate(frames):
                # if the frame we got from a source is an EndStreamQueueFrame, remove that source
                if isinstance(frame, EndStreamQueueFrame):
                    self.source_queues.pop(idx)
                else:
                    await self.out_queue.put(frame)

            async def pass_through(out_queue, source_queue):
                while True:
                    frame = await source_queue.get()
                    if isinstance(frame, EndStreamQueueFrame) or isinstance(
                        frame, self.ungate_frame_class
                    ):
                        break
                    else:
                        await out_queue.put(frame)

            await asyncio.gather(
                *[pass_through(self.out_queue, source) for source in self.source_queues]
            )

        await self.out_queue.put(EndStreamQueueFrame())


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
