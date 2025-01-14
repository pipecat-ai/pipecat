#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Dict, List

from loguru import logger

from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai import OpenAILLMService

try:
    from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam
    from openpipe import AsyncOpenAI as OpenPipeAI
    from openpipe import AsyncStream
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use OpenPipe, you need to `pip install pipecat-ai[openpipe]`. Also, set `OPENPIPE_API_KEY` and `OPENAI_API_KEY` environment variables."
    )
    raise Exception(f"Missing module: {e}")


class OpenPipeLLMService(OpenAILLMService):
    def __init__(
        self,
        *,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        openpipe_api_key: str | None = None,
        openpipe_base_url: str = "https://app.openpipe.ai/api/v1",
        tags: Dict[str, str] | None = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            openpipe_api_key=openpipe_api_key,
            openpipe_base_url=openpipe_base_url,
            **kwargs,
        )
        self._tags = tags

    def create_client(self, api_key=None, base_url=None, **kwargs):
        openpipe_api_key = kwargs.get("openpipe_api_key") or ""
        openpipe_base_url = kwargs.get("openpipe_base_url") or ""
        client = OpenPipeAI(
            api_key=api_key,
            base_url=base_url,
            openpipe={"api_key": openpipe_api_key, "base_url": openpipe_base_url},
        )
        return client

    async def get_chat_completions(
        self, context: OpenAILLMContext, messages: List[ChatCompletionMessageParam]
    ) -> AsyncStream[ChatCompletionChunk]:
        chunks = await self._client.chat.completions.create(
            model=self.model_name,
            stream=True,
            messages=messages,
            openpipe={"tags": self._tags, "log_request": True},
        )
        return chunks
