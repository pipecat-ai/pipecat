import asyncio

from dailyai.queue_frame import LLMMessagesQueueFrame, QueueFrame, TextQueueFrame, TranscriptionQueueFrame
from dailyai.services.ai_services import AIService

from typing import AsyncGenerator, List


class QueueTee:
    async def run_to_queue_and_generate(
        self,
        output_queue: asyncio.Queue,
        generator: AsyncGenerator[QueueFrame, None]
    ) -> AsyncGenerator[QueueFrame, None]:
        async for frame in generator:
            await output_queue.put(frame)
            yield frame

    async def run_to_queues(
        self,
        output_queues: List[asyncio.Queue],
        generator: AsyncGenerator[QueueFrame, None]
    ):
        async for frame in generator:
            for queue in output_queues:
                await queue.put(frame)


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
