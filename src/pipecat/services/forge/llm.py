#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Forge LLM service implementation using OpenAI-compatible interface."""

import os
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.settings import _warn_deprecated_param

DEFAULT_FORGE_BASE_URL = "https://api.forge.tensorblock.co/v1"


@dataclass
class ForgeLLMSettings(OpenAILLMSettings):
    """Settings for ForgeLLMService."""

    pass


class ForgeLLMService(OpenAILLMService):
    """A service for interacting with Forge using the OpenAI-compatible interface."""

    _settings: ForgeLLMSettings

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        settings: Optional[ForgeLLMSettings] = None,
        **kwargs,
    ):
        """Initialize the Forge LLM service.

        Args:
            api_key: Forge API key. If None, uses ``FORGE_API_KEY`` from the environment.
            base_url: Forge API base URL. If None, uses ``FORGE_API_BASE`` or default.
            model: The model identifier to use. Defaults to "OpenAI/gpt-4o-mini".

                .. deprecated:: 0.0.105
                    Use ``settings=OpenAILLMSettings(model=...)`` instead.

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

        resolved_api_key = api_key or os.getenv("FORGE_API_KEY")
        resolved_base_url = base_url or os.getenv("FORGE_API_BASE") or DEFAULT_FORGE_BASE_URL

        super().__init__(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            settings=default_settings,
            **kwargs,
        )

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create an OpenAI-compatible client for Forge.

        Args:
            api_key: API key for authentication.
            base_url: Base URL for the API.
            **kwargs: Additional arguments passed to the parent client creation method.

        Returns:
            An OpenAI-compatible client configured for Forge's API.
        """
        logger.debug(f"Creating Forge client with base URL: {base_url}")
        return super().create_client(api_key, base_url, **kwargs)
