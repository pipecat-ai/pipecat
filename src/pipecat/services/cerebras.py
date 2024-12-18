#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from loguru import logger

from pipecat.services.openai import OpenAILLMService


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
