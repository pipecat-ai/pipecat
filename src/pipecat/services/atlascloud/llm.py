#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Atlas Cloud LLM service implementation using OpenAI-compatible interface."""

from dataclasses import dataclass

from loguru import logger

from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService


@dataclass
class AtlasCloudLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for AtlasCloudLLMService."""

    pass


class AtlasCloudLLMService(OpenAILLMService):
    """A service for interacting with Atlas Cloud's OpenAI-compatible LLM API.

    This service extends OpenAILLMService to connect to Atlas Cloud's API endpoint
    while maintaining full compatibility with OpenAI's interface and functionality.
    """

    # Atlas Cloud's OpenAI-compatible endpoint does not support the "developer"
    # message role across every routed model.
    supports_developer_role = False

    Settings = AtlasCloudLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.atlascloud.ai/v1",
        model: str | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize Atlas Cloud LLM service.

        Args:
            api_key: The API key for accessing Atlas Cloud's API.
            base_url: The base URL for Atlas Cloud API. Defaults to "https://api.atlascloud.ai/v1".
            model: The model identifier to use. Defaults to "qwen/qwen3.5-flash".

                .. deprecated:: 0.0.105
                    Use ``settings=AtlasCloudLLMService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        default_settings = self.Settings(model="qwen/qwen3.5-flash")

        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            settings=default_settings,
            **kwargs,
        )

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Atlas Cloud API endpoint.

        Args:
            api_key: The API key for authentication. If None, uses instance default.
            base_url: The base URL for the API. If None, uses instance default.
            **kwargs: Additional keyword arguments for client configuration.

        Returns:
            An OpenAI-compatible client configured for Atlas Cloud's API.
        """
        logger.debug(f"Creating Atlas Cloud client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)
