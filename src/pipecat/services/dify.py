import asyncio
import json
from typing import AsyncGenerator, List, Optional

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
    from dify_client import ChatClient
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Dify, you need to pip install dify-client. Also, set DIFY_API_KEY environment variable."
    )
    raise Exception(f"Missing module: {e}")


class DifyLLMService(LLMService):
    """This class implements inference with Dify's AI models"""

    def __init__(
        self,
        *,
        api_key: str,
        inputs: List[dict],
        conversation_id: Optional[str] = None,
        user: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(api_key=api_key)
        self.client = ChatClient(api_key=api_key)
        self._conversation_id = conversation_id
        self._user = user  # Store user identifier
        self._inputs = inputs
        logger.debug(f"Inputs inside DifyLLMService: {inputs}")

    def can_generate_metrics(self) -> bool:
        return True

    def _get_messages_from_openai_context(
        self, context: OpenAILLMContext
    ) -> List[dict]:
        openai_messages = context.get_messages()
        dify_messages = []

        for message in openai_messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                role = "user"
            elif role == "assistant":
                role = "model"

            dify_messages.append({"role": role, "content": content})

        return dify_messages

    async def _process_context(self, context: OpenAILLMContext):
        await self.push_frame(LLMFullResponseStartFrame())
        try:
            logger.debug(f"Generating chat: {context.get_messages_json()}")

            messages = self._get_messages_from_openai_context(context)

            await self.start_ttfb_metrics()
            (logger.debug(f"Inputs just before create_chat_message{self._inputs}"),)
            async for text in self.create_chat_message(
                inputs=self._inputs,
                query=context.get_prompt(),
                user=self._user,  # Use the user from initialization
                response_mode="streaming",
                conversation_id=self._conversation_id,
            ):
                await self.push_frame(TextFrame(text))

            await self.stop_ttfb_metrics()

        except Exception as e:
            logger.exception(f"{self} exception: {e}")
        finally:
            await self.push_frame(LLMFullResponseEndFrame())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        context = None

        if isinstance(frame, OpenAILLMContext):
            context: OpenAILLMContext = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            context = OpenAILLMContext.from_messages(frame.messages)
        else:
            await self.push_frame(frame, direction)

        if context:
            await self._process_context(context)

    async def _async_generator_wrapper(self, sync_generator):
        for item in sync_generator:
            yield item
            await asyncio.sleep(0)

    async def create_chat_message(
        self,
        inputs: List[dict],
        query: str,
        user: str,
        response_mode: str,
        conversation_id: Optional[str],
    ) -> AsyncGenerator[str, None]:
        """Create a chat message"""
        logger.debug(
            f"Creating chat message with inputs: {inputs}, query: {query}, user: {user}, response_mode: {response_mode}, conversation_id: {conversation_id}"
        )

        chat_response = self.client.create_chat_message(
            inputs=inputs,
            query=query,
            user=user,
            conversation_id=conversation_id,
            response_mode=response_mode,
        )

        chat_response.raise_for_status()

        # Process streaming response
        for line in chat_response.iter_lines(decode_unicode=True):
            line = line.split("data:", 1)[-1]
            if line.strip():
                response_data = json.loads(line.strip())

                if response_data.get("event") == "message":
                    if self._conversation_id is None:
                        self._conversation_id = response_data.get("conversation_id")
                    text = response_data.get("answer", "")
                    logger.debug(f"answer: {text}")
                    yield text

    def get_conversation_id(self) -> Optional[str]:
        """Get the current conversation ID"""
        return self._conversation_id
