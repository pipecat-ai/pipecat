#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Forge LLM service implementation using OpenAI-compatible interface."""

from dataclasses import dataclass
from typing import Optional

from loguru import logger

from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.settings import _warn_deprecated_param


@dataclass
class ForgeLLMSettings(OpenAILLMSettings):
    """Settings for ForgeLLMService."""

    pass


class ForgeLLMService(OpenAILLMService):
    """A service for interacting with Forge's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Forge's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    """

    _settings: ForgeLLMSettings

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: str = "https://api.forge.tensorblock.co/v1",
        settings: Optional[ForgeLLMSettings] = None,
        **kwargs,
    ):
        """Initialize the Forge LLM service.

        Args:
            api_key: The API key for accessing Forge's API. If None, will attempt
                to read from environment variables.
            model: The model identifier to use. Defaults to "OpenAI/gpt-4o-mini".

                .. deprecated:: 0.0.105
                    Use ``settings=OpenAILLMSettings(model=...)`` instead.

            base_url: The base URL for Forge API. Defaults to "https://api.forge.tensorblock.co/v1".
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = ForgeLLMSettings(model="OpenAI/gpt-4o-mini")

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            _warn_deprecated_param("model", ForgeLLMSettings, "model")
            default_settings.model = model

        # 3. (No step 3, as there's no params object to apply)

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            settings=default_settings,
            **kwargs,
        )

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create an OpenAI-compatible client for Forge.

        Args:
            api_key: The API key to use for authentication. If None, uses instance default.
            base_url: The base URL for the API. If None, uses instance default.
            **kwargs: Additional arguments passed to the parent client creation method.

        Returns:
            The configured Forge API client instance.
        """
        logger.debug(f"Creating Forge client with base URL: {base_url}")
        return super().create_client(api_key, base_url, **kwargs)
