import asyncio
from typing import Optional

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    TextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class DifyProcessor(FrameProcessor):
    def __init__(self, dify_service, transcript_key: str = "input"):
        super().__init__()
        self._dify_service = dify_service
        self._transcript_key = transcript_key
        self._participant_id: Optional[str] = None

    def set_participant_id(self, participant_id: str):
        self._participant_id = participant_id

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMMessagesFrame):
            text: str = frame.messages[-1]["content"]
            await self._ainvoke(text.strip())
        else:
            await self.push_frame(frame, direction)

    async def _ainvoke(self, text: str):
        await self.push_frame(LLMFullResponseStartFrame())

        conversation_id = self._dify_service.get_conversation_id()
        if conversation_id:
            logger.info(f"Using conversation ID {conversation_id} for subsequent calls")

        try:
            async for token in self._dify_service.create_chat_message(
                inputs={self._transcript_key: text},
                query=text,
                user=self._dify_service._user,  # Use the user from the service
                conversation_id=self._dify_service.get_conversation_id(),
                response_mode="streaming",
            ):
                await self.push_frame(TextFrame(token))

        except asyncio.CancelledError:
            logger.warning("Task was cancelled.")
            raise
        except Exception as e:
            logger.exception(f"An error occurred: {e}")
        finally:
            await self.push_frame(LLMFullResponseEndFrame())
