#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Dict, List

from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai import BaseOpenAILLMService

from loguru import logger

try:
    from openpipe import AsyncOpenAI as OpenPipeAI, AsyncStream
    from openai.types.chat import (ChatCompletionMessageParam, ChatCompletionChunk)
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use OpenPipe, you need to `pip install pipecat-ai[openpipe]`. Also, set `OPENPIPE_API_KEY` and `OPENAI_API_KEY` environment variables.")
    raise Exception(f"Missing module: {e}")


class OpenPipeLLMService(BaseOpenAILLMService):

    def __init__(
            self,
            model: str = "gpt-4o",
            api_key: str | None = None,
            base_url: str | None = None,
            openpipe_api_key: str | None = None,
            openpipe_base_url: str = "https://app.openpipe.ai/api/v1",
            tags: Dict[str, str] | None = None,
            **kwargs):
        super().__init__(
            model,
            api_key,
            base_url,
            openpipe_api_key=openpipe_api_key,
            openpipe_base_url=openpipe_base_url,
            **kwargs)
        self._tags = tags

    def create_client(self, api_key=None, base_url=None, **kwargs):
        openpipe_api_key = kwargs.get("openpipe_api_key") or ""
        openpipe_base_url = kwargs.get("openpipe_base_url") or ""
        client = OpenPipeAI(
            api_key=api_key,
            base_url=base_url,
            openpipe={
                "api_key": openpipe_api_key,
                "base_url": openpipe_base_url
            }
        )
        return client

    async def get_chat_completions(
            self,
            context: OpenAILLMContext,
            messages: List[ChatCompletionMessageParam]) -> AsyncStream[ChatCompletionChunk]:
        chunks = await self._client.chat.completions.create(
            model=self._model,
            stream=True,
            messages=messages,
            openpipe={
                "tags": self._tags,
                "log_request": True
            }
        )
        return chunks
