#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Fireworks AI service implementation using OpenAI-compatible interface."""

from typing import List

from loguru import logger
from openai.types.chat import ChatCompletionMessageParam

from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService


class FireworksLLMService(OpenAILLMService):
    """A service for interacting with Fireworks AI using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Fireworks' API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.

    Args:
        api_key: The API key for accessing Fireworks AI.
        model: The model identifier to use. Defaults to "accounts/fireworks/models/firefunction-v2".
        base_url: The base URL for Fireworks API. Defaults to "https://api.fireworks.ai/inference/v1".
        **kwargs: Additional keyword arguments passed to OpenAILLMService.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "accounts/fireworks/models/firefunction-v2",
        base_url: str = "https://api.fireworks.ai/inference/v1",
        **kwargs,
    ):
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Fireworks API endpoint.

        Args:
            api_key: API key for authentication. If None, uses instance default.
            base_url: Base URL for the API. If None, uses instance default.
            **kwargs: Additional arguments passed to the client constructor.

        Returns:
            Configured OpenAI client instance for Fireworks API.
        """
        logger.debug(f"Creating Fireworks client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)

    async def get_chat_completions(
        self, context: OpenAILLMContext, messages: List[ChatCompletionMessageParam]
    ):
        """Get chat completions from Fireworks API.

        Removes OpenAI-specific parameters not supported by Fireworks and
        configures the request with Fireworks-compatible settings.

        Args:
            context: The OpenAI LLM context containing tools and settings.
            messages: List of chat completion message parameters.

        Returns:
            Async generator yielding chat completion chunks from Fireworks API.
        """
        params = {
            "model": self.model_name,
            "stream": True,
            "messages": messages,
            "tools": context.tools,
            "tool_choice": context.tool_choice,
            "frequency_penalty": self._settings["frequency_penalty"],
            "presence_penalty": self._settings["presence_penalty"],
            "temperature": self._settings["temperature"],
            "top_p": self._settings["top_p"],
            "max_tokens": self._settings["max_tokens"],
        }

        params.update(self._settings["extra"])

        chunks = await self._client.chat.completions.create(**params)
        return chunks
