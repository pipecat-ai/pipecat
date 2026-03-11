#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenPipe LLM service implementation for Pipecat.

This module provides an OpenPipe-specific implementation of the OpenAI LLM service,
enabling integration with OpenPipe's fine-tuning and monitoring capabilities.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from loguru import logger

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService

try:
    from openpipe import AsyncOpenAI as OpenPipeAI
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use OpenPipe, you need to `pip install pipecat-ai[openpipe]`.")
    raise Exception(f"Missing module: {e}")


@dataclass
class OpenPipeLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for OpenPipeLLMService."""

    pass


class OpenPipeLLMService(OpenAILLMService):
    """OpenPipe-powered Large Language Model service.

    Extends OpenAI's LLM service to integrate with OpenPipe's fine-tuning and
    monitoring platform. Provides enhanced request logging and tagging capabilities
    for model training and evaluation.
    """

    Settings = OpenPipeLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        openpipe_api_key: Optional[str] = None,
        openpipe_base_url: str = "https://app.openpipe.ai/api/v1",
        tags: Optional[Dict[str, str]] = None,
        settings: Optional[Settings] = None,
        **kwargs,
    ):
        """Initialize OpenPipe LLM service.

        Args:
            model: The model name to use. Defaults to "gpt-4.1".

                .. deprecated:: 0.0.105
                    Use ``settings=OpenPipeLLMService.Settings(model=...)`` instead.

            api_key: OpenAI API key for authentication. If None, reads from environment.
            base_url: Custom OpenAI API endpoint URL. Uses default if None.
            openpipe_api_key: OpenPipe API key for enhanced features. If None, reads from environment.
            openpipe_base_url: OpenPipe API endpoint URL. Defaults to "https://app.openpipe.ai/api/v1".
            tags: Optional dictionary of tags to apply to all requests for tracking.
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional arguments passed to parent OpenAILLMService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(model="gpt-4.1")

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        # 3. (No step 3, as there's no params object to apply)

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            openpipe_api_key=openpipe_api_key,
            openpipe_base_url=openpipe_base_url,
            settings=default_settings,
            **kwargs,
        )
        self._tags = tags

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create an OpenPipe client instance.

        Args:
            api_key: OpenAI API key for authentication.
            base_url: OpenAI API base URL.
            **kwargs: Additional arguments including openpipe_api_key and openpipe_base_url.

        Returns:
            Configured OpenPipe AsyncOpenAI client instance.
        """
        openpipe_api_key = kwargs.get("openpipe_api_key") or ""
        openpipe_base_url = kwargs.get("openpipe_base_url") or ""
        client = OpenPipeAI(
            api_key=api_key,
            base_url=base_url,
            openpipe={"api_key": openpipe_api_key, "base_url": openpipe_base_url},
        )
        return client

    def build_chat_completion_params(self, params_from_context: OpenAILLMInvocationParams) -> dict:
        """Build parameters for OpenPipe chat completion request.

        Adds OpenPipe-specific logging and tagging parameters.

        Args:
            params_from_context: Parameters, derived from the LLM context, to
                use for the chat completion. Contains messages, tools, and tool
                choice.

        Returns:
            Dictionary of parameters for the chat completion request.
        """
        # Start with base parameters
        params = super().build_chat_completion_params(params_from_context)

        # Add OpenPipe-specific parameters
        params["openpipe"] = {
            "tags": self._tags,
            "log_request": True,
        }

        return params
