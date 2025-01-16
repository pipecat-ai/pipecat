#
# Copyright (c) 2025, Daily
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
        "In order to use DeepSeek, you need to `pip install pipecat-ai[deepseek]`. Also, set `DEEPSEEK_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


class DeepSeekLLMService(OpenAILLMService):
    """A service for interacting with DeepSeek's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to DeepSeek's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.

    Args:
        api_key (str): The API key for accessing DeepSeek's API
        base_url (str, optional): The base URL for DeepSeek API. Defaults to "https://api.deepseek.com/v1"
        model (str, optional): The model identifier to use. Defaults to "deepseek-chat"
        **kwargs: Additional keyword arguments passed to OpenAILLMService
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        **kwargs,
    ):
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for DeepSeek API endpoint."""
        logger.debug(f"Creating DeepSeek client with api {base_url}")
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
            "stream_options": {"include_usage": True},
            "frequency_penalty": self._settings["frequency_penalty"],
            "presence_penalty": self._settings["presence_penalty"],
            "temperature": self._settings["temperature"],
            "top_p": self._settings["top_p"],
            "max_tokens": self._settings["max_tokens"],
        }

        params.update(self._settings["extra"])

        chunks = await self._client.chat.completions.create(**params)
        return chunks
