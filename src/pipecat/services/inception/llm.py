#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Inception LLM service implementation using OpenAI-compatible interface."""

from dataclasses import dataclass, field
from typing import Literal

from loguru import logger

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.settings import NOT_GIVEN as _NOT_GIVEN
from pipecat.services.settings import _NotGiven, is_given


@dataclass
class InceptionLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for InceptionLLMService.

    Parameters:
        reasoning_effort: Controls how much reasoning the model applies.
            One of "instant", "low", "medium", or "high". Defaults to "medium".
        realtime: When True, reduces time to first diffusion block (TTFT).
    """

    reasoning_effort: Literal["instant", "low", "medium", "high"] | None | _NotGiven = field(
        default_factory=lambda: _NOT_GIVEN
    )
    realtime: bool | None | _NotGiven = field(default_factory=lambda: _NOT_GIVEN)


class InceptionLLMService(OpenAILLMService):
    """A service for interacting with Inception's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Inception's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    Supports Mercury-2, Inception's diffusion-based reasoning model.
    """

    # Inception doesn't support the "developer" message role.
    supports_developer_role = False

    Settings = InceptionLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.inceptionlabs.ai/v1",
        model: str | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Inception LLM service.

        Args:
            api_key: The API key for accessing Inception's API.
            base_url: The base URL for Inception API. Defaults to "https://api.inceptionlabs.ai/v1".
            model: The model identifier to use. Defaults to "mercury-2".

                .. deprecated:: 0.0.105
                    Use ``settings=InceptionLLMService.Settings(model=...)`` instead.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        default_settings = self.Settings(model="mercury-2", reasoning_effort=None, realtime=None)

        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(api_key=api_key, base_url=base_url, settings=default_settings, **kwargs)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Inception API endpoint.

        Args:
            api_key: The API key for authentication. If None, uses instance default.
            base_url: The base URL for the API. If None, uses instance default.
            **kwargs: Additional keyword arguments for client configuration.

        Returns:
            An OpenAI-compatible client configured for Inception's API.
        """
        logger.debug(f"Creating Inception client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)

    def build_chat_completion_params(self, params_from_context: OpenAILLMInvocationParams) -> dict:
        """Build parameters for Inception chat completion request.

        Extends the base OpenAI parameters with Inception-specific options
        such as reasoning_effort and realtime.

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

        # realtime is Inception-specific and unknown to the OpenAI SDK,
        # so it must be passed via extra_body to avoid validation errors.
        extra_body = {}
        if is_given(self._settings.realtime) and self._settings.realtime is not None:
            extra_body["realtime"] = self._settings.realtime

        if extra_body:
            params["extra_body"] = extra_body

        return params
