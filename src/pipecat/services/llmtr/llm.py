#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LLMTR LLM service implementation using OpenAI-compatible interface."""

from dataclasses import dataclass

from loguru import logger

from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService


@dataclass
class LLMTRLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for LLMTRLLMService."""

    pass


class LLMTRLLMService(OpenAILLMService):
    """A service for interacting with LLMTR's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to LLMTR's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.

    LLMTR is a gateway at ``https://llmtr.com/v1``. Model IDs are namespaced
    ``vendor/model`` values such as ``minimax/minimax-m3.1-flash``. IDs under
    ``llmtr/`` are hosted in Turkey; other vendor/model IDs route to their vendors.
    """

    Settings = LLMTRLLMSettings
    _settings: Settings
    _requires_api_key = True

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://llmtr.com/v1",
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize LLMTR LLM service.

        Args:
            api_key: The API key for accessing LLMTR's API.
            base_url: The base URL for the LLMTR API. Defaults to "https://llmtr.com/v1".
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        default_settings = self.Settings(
            model="minimax/minimax-m3.1-flash",
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            settings=default_settings,
            **kwargs,
        )

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for the LLMTR API endpoint.

        Args:
            api_key: The API key to use for the client. If None, uses instance api_key.
            base_url: The base URL for the API. If None, uses instance base_url.
            **kwargs: Additional keyword arguments passed to the parent create_client method.

        Returns:
            An OpenAI-compatible client configured for LLMTR's API.
        """
        logger.debug(f"Creating LLMTR client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)
