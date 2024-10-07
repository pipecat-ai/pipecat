
# doubao is a Chinese llm service belonging to Bytedance
# install doubao sdk
# pip install 'volcengine-python-sdk[ark]'

import os
from dotenv import load_dotenv
from volcenginesdkarkruntime import Ark
from typing import AsyncGenerator
from loguru import logger
import json
from pipecat.services.ai_services import LLMService
from pipecat.frames.frames import Frame, LLMMessagesFrame, TextFrame, LLMFullResponseStartFrame, LLMFullResponseEndFrame, UserStartedSpeakingFrame
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext, OpenAILLMContextFrame

load_dotenv()


class DoubaoLLMService(LLMService):
    def __init__(self, *, model: str, **kwargs):
        super().__init__(model=model, **kwargs)
        self._client = Ark()
        self._model: str = model
        # detect if user start a new conversation during llm generation
        self.llm_user_interrupt = False
        # self._enable_metrics = True
        # self._report_only_initial_ttfb = True

    async def get_chat_completions(
            self,
            messages: list) -> AsyncGenerator[dict, None]:
        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                stream=True
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                char = chunk.choices[0].delta.content
                yield {"choices": [{"delta": {"content": char}}]}
        except Exception as e:
            logger.error(f"Error in doubao API call: {e}")
            yield {"choices": [{"delta": {"content": f"Error: {str(e)}"}}]}

    def _convert_context_to_doubao_format(self, context: OpenAILLMContext) -> list:
        doubao_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            }
        ]

        messages_json = context.get_messages_json()
        try:
            messages = json.loads(messages_json)
            if isinstance(messages, list):
                for message in messages:
                    if isinstance(message, dict) and 'role' in message and 'content' in message:
                        doubao_messages.append({
                            "role": message['role'],
                            "content": message['content']
                        })
                    else:
                        logger.warning(f"Skipping invalid message format: {message}")
            else:
                logger.error(f"Unexpected type for messages after JSON parsing: {type(messages)}")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from get_messages_json(): {e}")

        return doubao_messages

    def can_generate_metrics(self) -> bool:
        return self._enable_metrics

    async def _process_context(self, context: OpenAILLMContext):
        self.llm_user_interrupt = False
        await self.start_ttfb_metrics()
        messages = self._convert_context_to_doubao_format(context)
        chunk_stream = self.get_chat_completions(messages)
        try:
            async for chunk in chunk_stream:
                # detect if user start a new conversation during llm generation
                if self.llm_user_interrupt:
                    break
                await self.stop_ttfb_metrics()

                if len(chunk["choices"]) == 0:
                    continue

                if "content" in chunk["choices"][0]["delta"]:
                    await self.push_frame(TextFrame(chunk["choices"][0]["delta"]["content"]))

        except Exception as e:
            logger.error(f"Error in processing context: {e}")

    async def process_frame(self, frame: Frame, direction):
        if isinstance(frame, UserStartedSpeakingFrame):
            self.llm_user_interrupt = True
        await super().process_frame(frame, direction)

        context = None
        if isinstance(frame, OpenAILLMContextFrame):
            context: OpenAILLMContext = frame.context
        if isinstance(frame, LLMMessagesFrame):
            context = OpenAILLMContext.from_messages(frame.messages)
        if context:
            try:
                await self.push_frame(LLMFullResponseStartFrame())
                await self.start_processing_metrics()
                await self._process_context(context)
                await self.stop_processing_metrics()
                await self.push_frame(LLMFullResponseEndFrame())
            except Exception as e:
                logger.error(f"Error in processing context: {e}")
