# pipecat/processors/frameworks/dify.py

from typing import Optional
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    TextFrame
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from loguru import logger

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
            logger.debug(f"Got transcription frame {frame}")
            text: str = frame.messages[-1]["content"]
            await self._ainvoke(text.strip())
        else:
            await self.push_frame(frame, direction)

    async def _ainvoke(self, text: str):
        logger.debug(f"Invoking Dify service with {text}")
        await self.push_frame(LLMFullResponseStartFrame())

        conversation_id = self._dify_service.get_conversation_id()
        if conversation_id:
            logger.info(f"Using conversation ID {conversation_id} for subsequent calls")
        
        try:
            async for token in self._dify_service.create_chat_message(
                inputs={self._transcript_key: text},
                query=text,
                user=self._participant_id,
                conversation_id=conversation_id,
                response_mode="streaming"
            ):
                await self.push_frame(TextFrame(token))

        except asyncio.CancelledError:
            logger.warning("Task was cancelled.")
            raise
        except Exception as e:
            logger.exception(f"An error occurred: {e}")
        finally:
            await self.push_frame(LLMFullResponseEndFrame())
