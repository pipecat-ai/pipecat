from openai import AsyncOpenAI

import json
from collections.abc import AsyncGenerator

from dailyai.services.ai_services import LLMService


class OLLamaLLMService(LLMService):
    def __init__(self, model="llama2", base_url='http://localhost:11434/v1'):
        super().__init__()
        self._model = model
        self._client = AsyncOpenAI(api_key="ollama", base_url=base_url)

    async def get_response(self, messages, stream):
        return await self._client.chat.completions.create(
            stream=stream,
            messages=messages,
            model=self._model
        )

    async def run_llm_async(self, messages) -> AsyncGenerator[str, None]:
        messages_for_log = json.dumps(messages)
        self.logger.debug(f"Generating chat via openai: {messages_for_log}")

        chunks = await self._client.chat.completions.create(model=self._model, stream=True, messages=messages)
        async for chunk in chunks:
            if len(chunk.choices) == 0:
                continue

            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def run_llm(self, messages) -> str | None:
        messages_for_log = json.dumps(messages)
        self.logger.debug(f"Generating chat via openai: {messages_for_log}")

        response = await self._client.chat.completions.create(model=self._model, stream=False, messages=messages)
        if response and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            return None
