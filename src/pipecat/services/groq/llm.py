#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Groq LLM Service implementation using OpenAI-compatible interface."""

from dataclasses import dataclass, field
from typing import Literal

from loguru import logger

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.types import NOT_GIVEN, NotGiven, is_given


@dataclass
class GroqLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for GroqLLMService.

    Parameters:
        reasoning_effort: How much the model thinks before answering. Which
            values a model accepts varies: "low", "medium" and "high" on the
            GPT-OSS models and ``qwen/qwen3.8-27b``, "none" and "default" on the
            Qwen models, and the non-reasoning models reject the parameter
            altogether. "none" also keeps a Qwen model's reasoning out of the
            spoken response, which Groq otherwise streams inline in ``<think>``
            tags. When unset, Groq's per-model default applies.
    """

    reasoning_effort: Literal["none", "default", "low", "medium", "high"] | None | NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )


class GroqLLMService(OpenAILLMService):
    """A service for interacting with Groq's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Groq's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    """

    Settings = GroqLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.groq.com/openai/v1",
        model: str | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize Groq LLM service.

        Args:
            api_key: The API key for accessing Groq's API.
            base_url: The base URL for Groq API. Defaults to "https://api.groq.com/openai/v1".
            model: The model identifier to use. Defaults to "openai/gpt-oss-120b".

                .. deprecated:: 0.0.105
                    Use ``settings=GroqLLMService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="openai/gpt-oss-120b",
            reasoning_effort=None,
        )

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        # 3. (No step 3, as there's no params object to apply)

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

    def build_chat_completion_params(self, params_from_context: OpenAILLMInvocationParams) -> dict:
        """Build parameters for Groq chat completion request.

        Extends the base OpenAI parameters with Groq's reasoning effort control.

        Args:
            params_from_context: Parameters, derived from the LLM context, to
                use for the chat completion. Contains messages, tools, and tool
                choice.

        Returns:
            Dictionary of parameters for the chat completion request.
        """
        params = super().build_chat_completion_params(params_from_context)

        if (
            is_given(self._settings.reasoning_effort)
            and self._settings.reasoning_effort is not None
        ):
            params["reasoning_effort"] = self._settings.reasoning_effort

        return params
