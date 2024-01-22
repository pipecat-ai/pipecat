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

class LLMContextAggregator(AIService):
    def __init__(self, messages: list[dict], role:str, bot_participant_id=None):
        self.messages = messages
        self.bot_participant_id = bot_participant_id
        self.role = role
        self.sentence = ""

    async def process_frame(self, frame:QueueFrame) -> AsyncGenerator[QueueFrame, None]:
        content: str = ""

        if frame.frame_type == FrameType.TRANSCRIPTION:
            message = frame.frame_data
            if not isinstance(message, dict):
                return

            if message["session_id"] == self.bot_participant_id:
                return

            content = message["text"]
        elif frame.frame_type == FrameType.TEXT:
            if not isinstance(frame.frame_data, str):
                return

            content = frame.frame_data

        # todo: we should differentiate between transcriptions from different participants
        self.sentence += content
        if self.sentence.endswith((".", "?", "!")):
            self.messages.append({"role": self.role, "content": self.sentence})
            self.sentence = ""
            yield QueueFrame(FrameType.LLM_MESSAGE, self.messages)

        yield frame
