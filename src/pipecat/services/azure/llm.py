#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Azure OpenAI service implementation for the Pipecat AI framework."""

from dataclasses import dataclass
from typing import Optional

from loguru import logger
from openai import AsyncAzureOpenAI

from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.settings import _warn_deprecated_param


@dataclass
class AzureLLMSettings(OpenAILLMSettings):
    """Settings for AzureLLMService."""

    pass


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
        model: Optional[str] = None,
        api_version: str = "2024-09-01-preview",
        settings: Optional[AzureLLMSettings] = None,
        **kwargs,
    ):
        """Initialize the Azure LLM service.

        Args:
            api_key: The API key for accessing Azure OpenAI.
            endpoint: The Azure endpoint URL.
            model: The model identifier to use. Defaults to "gpt-4o".

                .. deprecated:: 0.0.105
                    Use ``settings=OpenAILLMSettings(model=...)`` instead.

            api_version: Azure API version. Defaults to "2024-09-01-preview".
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = AzureLLMSettings(model="gpt-4o")

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            _warn_deprecated_param("model", AzureLLMSettings, "model")
            default_settings.model = model

        # 3. (No step 3, as there's no params object to apply)

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        # Initialize variables before calling parent __init__() because that
        # will call create_client() and we need those values there.
        self._endpoint = endpoint
        self._api_version = api_version
        super().__init__(api_key=api_key, settings=default_settings, **kwargs)

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
