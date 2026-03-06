#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Groq LLM Service implementation using OpenAI-compatible interface."""

from dataclasses import dataclass
from typing import Optional

from loguru import logger

from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.settings import _warn_deprecated_param


@dataclass
class GroqLLMSettings(OpenAILLMSettings):
    """Settings for Groq LLM service."""

    pass


class GroqLLMService(OpenAILLMService):
    """A service for interacting with Groq's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Groq's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    """

    _settings: GroqLLMSettings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.groq.com/openai/v1",
        model: Optional[str] = None,
        settings: Optional[GroqLLMSettings] = None,
        **kwargs,
    ):
        """Initialize Groq LLM service.

        Args:
            api_key: The API key for accessing Groq's API.
            base_url: The base URL for Groq API. Defaults to "https://api.groq.com/openai/v1".
            model: The model identifier to use. Defaults to "llama-3.3-70b-versatile".

                .. deprecated:: 0.0.105
                    Use ``settings=OpenAILLMSettings(model=...)`` instead.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = GroqLLMSettings(model="llama-3.3-70b-versatile")

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            _warn_deprecated_param("model", GroqLLMSettings, "model")
            default_settings.model = model

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(api_key=api_key, base_url=base_url, settings=default_settings, **kwargs)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Groq API endpoint.

        Args:
            api_key: API key for authentication. If None, uses instance api_key.
            base_url: Base URL for the API. If None, uses instance base_url.
            **kwargs: Additional arguments passed to the client constructor.

        Returns:
            An OpenAI-compatible client configured for Groq's API.
        """
        logger.debug(f"Creating Groq client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)
