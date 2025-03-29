#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


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
        api_key (str): The API key for accessing Fireworks AI
        model (str, optional): The model identifier to use. Defaults to "accounts/fireworks/models/firefunction-v2"
        base_url (str, optional): The base URL for Fireworks API. Defaults to "https://api.fireworks.ai/inference/v1"
        **kwargs: Additional keyword arguments passed to OpenAILLMService
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
        """Create OpenAI-compatible client for Fireworks API endpoint."""
        logger.debug(f"Creating Fireworks client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)

    async def get_chat_completions(
        self, context: OpenAILLMContext, messages: List[ChatCompletionMessageParam]
    ):
        """Get chat completions from Fireworks API.

        Removes OpenAI-specific parameters not supported by Fireworks.
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
