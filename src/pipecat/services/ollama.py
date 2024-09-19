

import requests
import json
from typing import AsyncGenerator
from loguru import logger

from pipecat.services.ai_services import LLMService
from pipecat.frames.frames import Frame, LLMMessagesFrame, TextFrame, LLMFullResponseStartFrame, LLMFullResponseEndFrame, UserStartedSpeakingFrame
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext, OpenAILLMContextFrame
from dotenv import load_dotenv
import os


class OLLamaLLMService(LLMService):
    def __init__(self, **kwargs):
        load_dotenv()

        self._model = os.getenv('OLLAMA_MODEL', 'llama3')
        self._base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/api/chat')
        super().__init__(model=self._model, **kwargs)

        self.llm_user_interrupt = False

    async def get_chat_completions(self, messages: list) -> AsyncGenerator[dict, None]:
        try:
            payload = {
                "model": self._model,
                "messages": messages
            }
            response = requests.post(self._base_url, json=payload, stream=True)
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        content = json.loads(line)["message"]["content"]
                        yield content
            else:
                yield f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            logger.error(f"Error in OLLama API call: {e}")
            yield f"Error: {str(e)}"

    async def _process_context(self, context: OpenAILLMContext):
        self.llm_user_interrupt = False

        await self.start_ttfb_metrics()

        messages = context.messages

        chunk_stream = self.get_chat_completions(messages)

        try:
            async for chunk in chunk_stream:

                if self.llm_user_interrupt:
                    break

                await self.stop_ttfb_metrics()

                if chunk:
                    await self.push_frame(TextFrame(chunk))

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
