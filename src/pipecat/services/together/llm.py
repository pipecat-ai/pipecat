#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Together.ai LLM service implementation using OpenAI-compatible interface."""

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given


@dataclass
class TogetherLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for TogetherLLMService.

    Parameters:
        reasoning: Together's reasoning toggle, for the models that support one,
            e.g. ``{"enabled": False}``. A reasoning model such as GLM runs a
            reasoning pass before every answer, which delays the first spoken
            token. ``None`` leaves the choice to the model's own default.
    """

    reasoning: dict[str, Any] | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class TogetherLLMService(OpenAILLMService):
    """A service for interacting with Together.ai's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Together.ai's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    """

    # Together.ai doesn't support the "developer" message role (it seems to quietly
    # ignore "developer" messages).
    # This value is used by BaseOpenAILLMService when calling the adapter.
    supports_developer_role = False

    Settings = TogetherLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.together.xyz/v1",
        model: str | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize Together.ai LLM service.

        Args:
            api_key: The API key for accessing Together.ai's API.
            base_url: The base URL for Together.ai API. Defaults to "https://api.together.xyz/v1".
            model: The model identifier to use. Defaults to "zai-org/GLM-5.2".

                .. deprecated:: 0.0.105
                    Use ``settings=TogetherLLMService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(model="zai-org/GLM-5.2", reasoning=None)

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

    def build_chat_completion_params(
        self, params_from_context: OpenAILLMInvocationParams
    ) -> dict[str, Any]:
        """Build parameters for a Together.ai chat completion request.

        Args:
            params_from_context: Parameters, derived from the LLM context, to
                use for the chat completion. Contains messages, tools, and tool
                choice.

        Returns:
            Dictionary of parameters for the chat completion request.
        """
        params = super().build_chat_completion_params(params_from_context)

        # `reasoning` is Together's own field, so it travels in the OpenAI
        # client's `extra_body` rather than as a client keyword argument. An
        # `extra_body` supplied through `Settings.extra` wins key by key.
        reasoning = assert_given(self._settings.reasoning)
        if reasoning is not None:
            params["extra_body"] = {"reasoning": reasoning, **params.get("extra_body", {})}

        return params
