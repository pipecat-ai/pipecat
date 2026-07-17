#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Crusoe Cloud LLM service implementation using OpenAI-compatible interface."""

from dataclasses import dataclass

from loguru import logger

from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService


@dataclass
class CrusoeLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for CrusoeLLMService."""

    pass


class CrusoeLLMService(OpenAILLMService):
    """A service for interacting with Crusoe Cloud's Managed Inference API.

    This service extends OpenAILLMService to connect to Crusoe's OpenAI-compatible
    API endpoint while maintaining full compatibility with OpenAI's interface and
    functionality.
    """

    # Crusoe doesn't support the "developer" message role.
    # This value is used by BaseOpenAILLMService when calling the adapter.
    supports_developer_role = False

    Settings = CrusoeLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.inference.crusoecloud.com/v1/",
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Crusoe LLM service.

        Args:
            api_key: The API key for accessing Crusoe's Managed Inference API.
            base_url: The base URL for the Crusoe API. Defaults to
                ``"https://api.inference.crusoecloud.com/v1/"``.
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        # Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="zai/GLM-5.2",
        )

        # Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(api_key=api_key, base_url=base_url, settings=default_settings, **kwargs)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Crusoe API endpoint.

        Args:
            api_key: The API key for authentication. If None, uses instance default.
            base_url: The base URL for the API. If None, uses instance default.
            **kwargs: Additional keyword arguments for client configuration.

        Returns:
            An OpenAI-compatible client configured for Crusoe's API.
        """
        logger.debug(f"Creating Crusoe client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)
