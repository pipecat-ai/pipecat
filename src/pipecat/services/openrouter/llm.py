#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenRouter LLM service implementation.

This module provides an OpenAI-compatible interface for interacting with OpenRouter's API,
extending the base OpenAI LLM service functionality.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Union

from loguru import logger
from pydantic import BaseModel

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given, is_given


class OpenRouterProviderPreferences(BaseModel):
    """OpenRouter's preferences for which upstream provider serves a request.

    Every model on OpenRouter can be served by several providers, and by
    default OpenRouter picks among them and falls back to another when one
    fails. These preferences constrain that choice. See
    https://openrouter.ai/docs/features/provider-routing for the provider
    slugs and the full semantics.

    Parameters:
        order: Provider slugs to try, in order, before any others.
        allow_fallbacks: Whether to fall back to providers outside ``order``,
            ``only`` and ``quantizations``. Defaults to true.
        require_parameters: Whether to route only to providers that support
            every parameter in the request.
        data_collection: "deny" restricts routing to providers that do not
            store prompts.
        zdr: Whether to route only to zero-data-retention endpoints.
        only: Provider slugs to route to, to the exclusion of all others.
        ignore: Provider slugs never to route to.
        quantizations: Quantization levels to accept, e.g. ``["fp8", "bf16"]``.
        sort: Attribute to rank providers by — "price", "throughput" or
            "latency" — which turns off load balancing.
        max_price: Ceilings on what a request may cost, per key ("prompt",
            "completion", "request", "image"), in USD per million tokens.
    """

    # Why `| str` and `| dict` on the constrained fields? OpenRouter adds
    # routing options regularly, and a request should not be rejected here
    # for using one that landed after this model was written.
    order: list[str] | None = None
    allow_fallbacks: bool | None = None
    require_parameters: bool | None = None
    data_collection: Literal["allow", "deny"] | str | None = None
    zdr: bool | None = None
    only: list[str] | None = None
    ignore: list[str] | None = None
    quantizations: list[str] | None = None
    sort: Literal["price", "throughput", "latency"] | str | dict[str, Any] | None = None
    max_price: dict[str, float] | None = None


@dataclass
class OpenRouterLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for OpenRouterLLMService.

    Parameters:
        provider: Which upstream providers may serve the request. Left unset,
            OpenRouter routes by its own default order.
    """

    provider: Union["OpenRouterLLMService.ProviderPreferences", dict[str, Any], NotGiven] = field(
        default_factory=lambda: NOT_GIVEN
    )


class OpenRouterLLMService(OpenAILLMService):
    """A service for interacting with OpenRouter's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to OpenRouter's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    """

    Settings = OpenRouterLLMSettings
    _settings: Settings
    supports_developer_role = False

    ProviderPreferences = OpenRouterProviderPreferences

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the OpenRouter LLM service.

        Args:
            api_key: The API key for accessing OpenRouter's API. If None, will attempt
                to read from environment variables.
            model: The model identifier to use. Defaults to "openai/gpt-4.1".

                .. deprecated:: 0.0.105
                    Use ``settings=OpenRouterLLMService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            base_url: The base URL for OpenRouter API. Defaults to "https://openrouter.ai/api/v1".
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(model="openai/gpt-4.1")

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
            settings=default_settings,
            **kwargs,
        )

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create an OpenRouter API client.

        Args:
            api_key: The API key to use for authentication. If None, uses instance default.
            base_url: The base URL for the API. If None, uses instance default.
            **kwargs: Additional arguments passed to the parent client creation method.

        Returns:
            The configured OpenRouter API client instance.
        """
        logger.debug(f"Creating OpenRouter client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)

    def _apply_provider_preferences(self, params: dict[str, Any]):
        """Put the caller's provider preferences in a request.

        ``provider`` is OpenRouter's own request field rather than an OpenAI
        one, so it travels in ``extra_body``, which the OpenAI client merges
        into the JSON body it sends.
        """
        preferences = self._settings.provider
        if not is_given(preferences) or preferences is None:
            return

        if isinstance(preferences, BaseModel):
            preferences = preferences.model_dump(exclude_none=True)

        extra_body = dict(params.get("extra_body") or {})
        extra_body["provider"] = preferences
        params["extra_body"] = extra_body

    def build_chat_completion_params(
        self, params_from_context: OpenAILLMInvocationParams
    ) -> dict[str, Any]:
        """Builds chat parameters, handling model-specific constraints.

        Args:
            params_from_context: Parameters from the LLM context.

        Returns:
            Transformed parameters ready for the API call.
        """
        params = super().build_chat_completion_params(params_from_context)
        self._apply_provider_preferences(params)
        model = assert_given(self._settings.model)
        if model is not None and "gemini" in model.lower():
            messages = params.get("messages", [])
            if not messages:
                return params
            transformed_messages = []
            system_message_seen = False
            for msg in messages:
                if msg.get("role") == "system":
                    if not system_message_seen:
                        transformed_messages.append(msg)
                        system_message_seen = True
                    else:
                        new_msg = msg.copy()
                        new_msg["role"] = "user"
                        transformed_messages.append(new_msg)
                else:
                    transformed_messages.append(msg)
            params["messages"] = transformed_messages

        return params
