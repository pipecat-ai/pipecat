import asyncio

from dailyai.queue_frame import QueueFrame, FrameType
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

class TranscriptionToLLMMessageAggregator(AIService):
    def __init__(self, messages, bot_participant_id):
        self.messages = messages
        self.bot_participant_id = bot_participant_id
        self.sentence = ""

    async def process_frame(self, frame:QueueFrame) -> AsyncGenerator[QueueFrame, None]:
        if frame.frame_type != FrameType.TRANSCRIPTION:
            return

        message = frame.frame_data
        if not isinstance(message, dict):
            return

        if message["session_id"] == self.bot_participant_id:
            return

        print("transcription to message", frame)

        # todo: we could differentiate between transcriptions from different participants
        self.sentence += message["text"]
        if self.sentence.endswith((".", "?", "!")):
            self.messages.append({"role": "user", "content": self.sentence})
            self.sentence = ""
            yield QueueFrame(FrameType.LLM_MESSAGE, self.messages)


class LLMResponseToLLMMessageAggregator(AIService):
    def __init__(self, messages):
        self.messages = messages
        self.sentence = ""

    async def process_frame(self, frame:QueueFrame) -> AsyncGenerator[QueueFrame, None]:
        if frame.frame_type == FrameType.TEXT and isinstance(frame.frame_data, str):
            print("llmresponse to message", frame)
            self.sentence += frame.frame_data
            if self.sentence.endswith((".", "?", "!")):
                self.messages.append({"role": "assistant", "content": self.sentence})
                self.sentence = ""

        yield frame
