#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Together.ai LLM service implementation using OpenAI-compatible interface."""

from dataclasses import dataclass
from typing import Optional

from loguru import logger

from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.settings import _warn_deprecated_param


@dataclass
class TogetherLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for TogetherLLMService."""

    pass


class TogetherLLMService(OpenAILLMService):
    """A service for interacting with Together.ai's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Together.ai's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    """

    Settings = TogetherLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.together.xyz/v1",
        model: Optional[str] = None,
        settings: Optional[Settings] = None,
        **kwargs,
    ):
        """Initialize Together.ai LLM service.

        Args:
            api_key: The API key for accessing Together.ai's API.
            base_url: The base URL for Together.ai API. Defaults to "https://api.together.xyz/v1".
            model: The model identifier to use. Defaults to "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo".

                .. deprecated:: 0.0.105
                    Use ``settings=TogetherLLMService.Settings(model=...)`` instead.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            _warn_deprecated_param("model", self.Settings, "model")
            default_settings.model = model

        # 3. (No step 3, as there's no params object to apply)

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(api_key=api_key, base_url=base_url, settings=default_settings, **kwargs)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Together.ai API endpoint.

        Args:
            api_key: The API key to use for the client. If None, uses instance api_key.
            base_url: The base URL for the API. If None, uses instance base_url.
            **kwargs: Additional keyword arguments passed to the parent create_client method.

        Returns:
            An OpenAI-compatible client configured for Together.ai's API.
        """
        logger.debug(f"Creating Together.ai client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)
