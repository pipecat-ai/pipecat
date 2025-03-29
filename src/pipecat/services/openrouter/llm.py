#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Optional

from loguru import logger

from pipecat.services.openai.llm import OpenAILLMService


class OpenRouterLLMService(OpenAILLMService):
    """A service for interacting with OpenRouter's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to OpenRouter's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.

    Args:
        api_key (str): The API key for accessing OpenRouter's API
        base_url (str, optional): The base URL for OpenRouter API. Defaults to "https://openrouter.ai/api/v1"
        model (str, optional): The model identifier to use. Defaults to "openai/gpt-4o-2024-11-20"
        **kwargs: Additional keyword arguments passed to OpenAILLMService
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-4o-2024-11-20",
        base_url: str = "https://openrouter.ai/api/v1",
        **kwargs,
    ):
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            **kwargs,
        )

    def create_client(self, api_key=None, base_url=None, **kwargs):
        logger.debug(f"Creating OpenRouter client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)
