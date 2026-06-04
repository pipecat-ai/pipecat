#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Qwen LLM service implementation using OpenAI-compatible interface."""

from dataclasses import dataclass

from loguru import logger

from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService


@dataclass
class QwenLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for QwenLLMService."""

    pass


class QwenLLMService(OpenAILLMService):
    """A service for interacting with Alibaba Cloud's Qwen LLM API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Qwen's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    """

    # Qwen doesn't support the "developer" message role.
    # This value is used by BaseOpenAILLMService when calling the adapter.
    supports_developer_role = False

    Settings = QwenLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        model: str | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Qwen LLM service.

        Args:
            api_key: The API key for accessing Qwen's API (DashScope API key).
            base_url: Base URL for Qwen API. Defaults to "https://dashscope-intl.aliyuncs.com/compatible-mode/v1".
            model: The model identifier to use. Defaults to "qwen-plus".

                .. deprecated:: 0.0.105
                    Use ``settings=QwenLLMService.Settings(model=...)`` instead.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(model="qwen-plus")

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        # 3. (No step 3, as there's no params object to apply)

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(api_key=api_key, base_url=base_url, settings=default_settings, **kwargs)
        logger.info(f"Initialized Qwen LLM service with model: {self._settings.model}")

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Qwen API endpoint.

        Args:
            api_key: API key for authentication. If None, uses instance default.
            base_url: Base URL for the API. If None, uses instance default.
            **kwargs: Additional arguments passed to the parent client creation.

        Returns:
            An OpenAI-compatible client configured for Qwen's API.
        """
        logger.debug(f"Creating Qwen client with base URL: {base_url}")
        return super().create_client(api_key, base_url, **kwargs)
