#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OLLama LLM service implementation for Pipecat AI framework."""

from dataclasses import dataclass
from typing import Optional

from loguru import logger

from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService


@dataclass
class OllamaLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for OLLamaLLMService."""

    pass


class OLLamaLLMService(OpenAILLMService):
    """OLLama LLM service that provides local language model capabilities.

    This service extends OpenAILLMService to work with locally hosted OLLama models,
    providing a compatible interface for running large language models locally.
    """

    Settings = OllamaLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        base_url: str = "http://localhost:11434/v1",
        settings: Optional[Settings] = None,
        **kwargs,
    ):
        """Initialize OLLama LLM service.

        Args:
            model: The OLLama model to use. Defaults to "llama2".

                .. deprecated:: 0.0.105
                    Use ``settings=OLLamaLLMService.Settings(model=...)`` instead.

            base_url: The base URL for the OLLama API endpoint.
                    Defaults to "http://localhost:11434/v1".
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(model="llama2")

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        # 3. (No step 3, as there's no params object to apply)

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(base_url=base_url, api_key="ollama", settings=default_settings, **kwargs)

    def create_client(self, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Ollama.

        Args:
            base_url: The base URL for the API. If None, uses instance base_url.
            **kwargs: Additional keyword arguments passed to the parent create_client method.

        Returns:
            An OpenAI-compatible client configured for Ollama.
        """
        logger.debug(f"Creating Ollama client with api {base_url}")
        return super().create_client(base_url=base_url, **kwargs)
