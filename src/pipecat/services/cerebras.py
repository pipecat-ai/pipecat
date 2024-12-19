#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import List

from loguru import logger

from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai import OpenAILLMService

try:
    from openai import (
        AsyncStream,
    )
    from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Fireworks, you need to `pip install pipecat-ai[cerebras]`. Also, set `CEREBRAS_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


class CerebrasLLMService(OpenAILLMService):
    """A service for interacting with Cerebras's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Cerebras's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.

    Args:
        api_key (str): The API key for accessing Cerebras's API
        base_url (str, optional): The base URL for Cerebras API. Defaults to "https://api.cerebras.ai/v1"
        model (str, optional): The model identifier to use. Defaults to "llama-3.3-70b"
        **kwargs: Additional keyword arguments passed to OpenAILLMService
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.cerebras.ai/v1",
        model: str = "llama-3.3-70b",
        **kwargs,
    ):
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Cerebras API endpoint."""
        logger.debug(f"Creating Cerebras client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)

    async def get_chat_completions(
        self, context: OpenAILLMContext, messages: List[ChatCompletionMessageParam]
    ) -> AsyncStream[ChatCompletionChunk]:
        """Create a streaming chat completion using Cerebras's API.

        Args:
        context (OpenAILLMContext): The context object containing tools configuration
            and other settings for the chat completion.
        messages (List[ChatCompletionMessageParam]): The list of messages comprising
            the conversation history and current request.

        Returns:
        AsyncStream[ChatCompletionChunk]: A streaming response of chat completion
            chunks that can be processed asynchronously.
        """
        params = {
            "model": self.model_name,
            "stream": True,
            "messages": messages,
            "tools": context.tools,
            "tool_choice": context.tool_choice,
            "seed": self._settings["seed"],
            "temperature": self._settings["temperature"],
            "top_p": self._settings["top_p"],
            "max_completion_tokens": self._settings["max_completion_tokens"],
        }

        params.update(self._settings["extra"])

        chunks = await self._client.chat.completions.create(**params)
        return chunks
