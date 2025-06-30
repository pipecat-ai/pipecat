#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Azure OpenAI service implementation for the Pipecat AI framework."""

from loguru import logger
from openai import AsyncAzureOpenAI

from pipecat.services.openai.llm import OpenAILLMService


class AzureLLMService(OpenAILLMService):
    """A service for interacting with Azure OpenAI using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Azure's OpenAI endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    """

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str,
        model: str,
        api_version: str = "2024-09-01-preview",
        **kwargs,
    ):
        """Initialize the Azure LLM service.

        Args:
            api_key: The API key for accessing Azure OpenAI.
            endpoint: The Azure endpoint URL.
            model: The model identifier to use.
            api_version: Azure API version. Defaults to "2024-09-01-preview".
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        # Initialize variables before calling parent __init__() because that
        # will call create_client() and we need those values there.
        self._endpoint = endpoint
        self._api_version = api_version
        super().__init__(api_key=api_key, model=model, **kwargs)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Azure OpenAI endpoint.

        Args:
            api_key: API key for authentication. Uses instance key if None.
            base_url: Base URL for the client. Ignored for Azure implementation.
            **kwargs: Additional keyword arguments. Ignored for Azure implementation.

        Returns:
            AsyncAzureOpenAI: Configured Azure OpenAI client instance.
        """
        logger.debug(f"Creating Azure OpenAI client with endpoint {self._endpoint}")
        return AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=self._endpoint,
            api_version=self._api_version,
        )
