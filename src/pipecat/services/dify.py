from typing import Dict, List, Optional

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    TextFrame,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import LLMService

try:
    from dify_client import AsyncClient, models
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Dify, you need to pip install dify-cli. Also, set DIFY_API_KEY environment variable."
    )
    raise Exception(f"Missing module: {e}")


def map_response_mode(response_mode: str) -> models.ResponseMode:
    return (
        models.ResponseMode.STREAMING
        if response_mode == "streaming"
        else models.ResponseMode.BLOCKING
    )


class DifyLLMService(LLMService):
    def __init__(
        self,
        api_key: str,
        api_base: Optional[str] = "https://api.dify.ai/v1",
        inputs: Optional[Dict[str, str]] = None,
        user: Optional[str] = None,
        response_mode: Optional[str] = "streaming",
        conversation_id: Optional[str] = None,
        files: Optional[List[str]] = None,
    ):
        super().__init__()
        self.client = AsyncClient(api_key=api_key, api_base=api_base)
        self._conversation_id = conversation_id
        self._user = user
        self._inputs = inputs or {}
        self._response_mode = map_response_mode(response_mode)
        self._files = files or []
        logger.debug(f"Inputs inside DifyService: {self._inputs}")

    def can_generate_metrics(self) -> bool:
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMMessagesFrame):
            await self._process_message(frame.messages[-1]["content"].strip())
        elif isinstance(frame, OpenAILLMContext):
            await self._process_context(frame)
        else:
            await self.push_frame(frame, direction)

    async def _process_message(self, text: str):
        await self.push_frame(LLMFullResponseStartFrame())
        try:
            await self._stream_response(text)
        except Exception as e:
            logger.exception(f"{self} exception: {e}")
        finally:
            await self.push_frame(LLMFullResponseEndFrame())

    async def _stream_response(self, query: str):
        chat_req = models.ChatRequest(
            query=query,
            inputs=self._inputs,
            user=self._user,
            conversation_id=self._conversation_id,
            response_mode=self._response_mode,
            files=self._files,
        )

        try:
            if self._response_mode == models.ResponseMode.BLOCKING:
                chat_response = await self.client.achat_messages(chat_req, timeout=60.0)
                await self.push_frame(TextFrame(chat_response.answer))
                if not self._conversation_id and chat_response.conversation_id:
                    self._conversation_id = chat_response.conversation_id
            else:
                async for chunk in await self.client.achat_messages(
                    chat_req, timeout=60.0
                ):
                    if chunk.event == models.StreamEvent.MESSAGE:
                        await self.push_frame(TextFrame(chunk.answer))
                        if not self._conversation_id and chunk.conversation_id:
                            self._conversation_id = chunk.conversation_id
                    else:
                        logger.debug(f"Unexpected event type: {chunk.event}")
        except Exception as e:
            logger.exception(f"Error in _stream_response: {e}")

    def get_conversation_id(self) -> Optional[str]:
        return self._conversation_id

    def set_user(self, user: str):
        self._user = user

    def update_inputs(self, inputs: Dict[str, str]):
        self._inputs.update(inputs)
