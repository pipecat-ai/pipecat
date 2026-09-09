#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""DeepSeek LLM service implementation using OpenAI-compatible interface."""

from dataclasses import dataclass, field
from typing import Literal

from loguru import logger
from pydantic import BaseModel

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given


class DeepSeekThinkingConfig(BaseModel):
    """Configuration for thinking.

    Parameters:
        type: Thinking mode. DeepSeek's V4 models think before answering unless
            "disabled" turns it off, which for real-time voice cuts the time to
            the first answer token.
    """

    # `| str` keeps the field usable if DeepSeek adds further modes.
    type: Literal["enabled", "disabled"] | str


@dataclass
class DeepSeekLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for DeepSeekLLMService.

    Parameters:
        thinking: Thinking mode configuration. When unset, DeepSeek's own
            default applies: thinking enabled at "high" reasoning effort.
    """

    thinking: DeepSeekThinkingConfig | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)

    def __post_init__(self):
        """Coerce a plain ``thinking`` dict to a :class:`DeepSeekThinkingConfig`."""
        if isinstance(self.thinking, dict):
            self.thinking = DeepSeekThinkingConfig(**self.thinking)


class DeepSeekLLMService(OpenAILLMService):
    """A service for interacting with DeepSeek's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to DeepSeek's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    """

    # DeepSeek doesn't support the "developer" message role.
    # This value is used by BaseOpenAILLMService when calling the adapter.
    supports_developer_role = False

    Settings = DeepSeekLLMSettings
    ThinkingConfig = DeepSeekThinkingConfig
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the DeepSeek LLM service.

        Args:
            api_key: The API key for accessing DeepSeek's API.
            base_url: The base URL for DeepSeek API. Defaults to "https://api.deepseek.com/v1".
            model: The model identifier to use. Defaults to "deepseek-v4-flash".

                .. deprecated:: 0.0.105
                    Use ``settings=DeepSeekLLMService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(model="deepseek-v4-flash", thinking=None)

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
        """Create OpenAI-compatible client for DeepSeek API endpoint.

        Args:
            api_key: The API key for authentication. If None, uses instance default.
            base_url: The base URL for the API. If None, uses instance default.
            **kwargs: Additional keyword arguments for client configuration.

        Returns:
            An OpenAI-compatible client configured for DeepSeek's API.
        """
        logger.debug(f"Creating DeepSeek client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)

    def _build_chat_completion_params(self, params_from_context: OpenAILLMInvocationParams) -> dict:
        """Build parameters for DeepSeek chat completion request.

        DeepSeek doesn't support some OpenAI parameters like seed and max_completion_tokens.

        Args:
            params_from_context: Parameters, derived from the LLM context, to
                use for the chat completion. Contains messages, tools, and tool
                choice.

        Returns:
            Dictionary of parameters for the chat completion request.
        """
        params = {
            "model": self._settings.model,
            "stream": True,
            "stream_options": {"include_usage": True},
            "frequency_penalty": self._settings.frequency_penalty,
            "presence_penalty": self._settings.presence_penalty,
            "temperature": self._settings.temperature,
            "top_p": self._settings.top_p,
            "max_tokens": self._settings.max_tokens,
        }

        # `thinking` is DeepSeek's own field, so it travels in the OpenAI
        # client's `extra_body` rather than as a client keyword argument.
        thinking = assert_given(self._settings.thinking)
        if thinking:
            params["extra_body"] = {"thinking": thinking.model_dump(exclude_none=True)}

        # Messages, tools, tool_choice
        params.update(params_from_context)

        params.update(self._settings.extra)
        return params
