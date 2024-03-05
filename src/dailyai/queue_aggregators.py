import asyncio

from dailyai.queue_frame import BotTranscriptionFrame, LLMMessagesQueueFrame, QueueFrame, TextQueueFrame, TranscriptionQueueFrame
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
            context: list[dict],
            bot_participant_id=None,
        ):
        super().__init__()
        self._context = context
        self.bot_participant_id = bot_participant_id

    async def process_frame(self, frame: QueueFrame) -> AsyncGenerator[QueueFrame, None]:
        # We don't do anything with non-text frames, pass it along to next in the pipeline.
        
        if not isinstance(frame, TextQueueFrame):
            yield frame
            return

        # TODO-CB: This should be a no-op now
        # Ignore transcription frames from the bot
        if isinstance(frame, TranscriptionQueueFrame):
            if frame.participantId == self.bot_participant_id:
                return
            else:
                self.append_to_context("user", frame.text)
        elif isinstance(frame, BotTranscriptionFrame):
            if frame.save_in_context == True:
                self.append_to_context("assistant", frame.text)
            else:
                print("save in context false")
        else:
            yield frame
    
    def append_to_context(self, role, text):
        if len(self._context) > 0 and self._context[-1] and self._context[-1]['role'] == role:
            self._context[-1]['content'] += f" {text}"
        else:
            self._context.append({"role": role, "content": text})

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
