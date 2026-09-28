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
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given, is_given


class OpenRouterProviderPreferences(BaseModel):
    """OpenRouter's preferences for which upstream provider serves a request.

    Every model on OpenRouter can be served by several providers, and by
    default OpenRouter picks among them and falls back to another when one
    fails. These preferences constrain that choice. See
    https://openrouter.ai/docs/guides/routing/provider-selection for the provider
    slugs and the full semantics.

    Parameters:
        order: Provider slugs to try, in order, before any others. Setting it
            turns off load balancing.
        allow_fallbacks: Whether to fall back to providers outside ``order``,
            ``only`` and ``quantizations``. Defaults to true.
        require_parameters: Whether to route only to providers that support
            every parameter in the request. Defaults to false.
        data_collection: "deny" restricts routing to providers that do not
            collect user data. Defaults to "allow".
        zdr: Whether to route only to zero-data-retention endpoints.
        enforce_distillable_text: Whether to route only to models that allow
            text distillation.
        only: Provider slugs to route to, to the exclusion of all others.
        ignore: Provider slugs never to route to.
        quantizations: Quantization levels to accept, e.g. ``["fp8", "bf16"]``.
        sort: Attribute to rank providers by — "price", "throughput" or
            "latency" — or an object of the form ``{"by": ..., "partition":
            "model" | "none"}``. Setting it turns off load balancing.
        preferred_min_throughput: Throughput, in tokens per second, below which
            a provider is deprioritized. Either a number or percentile
            thresholds keyed "p50", "p75", "p90" and "p99".
        preferred_max_latency: Latency, in seconds, above which a provider is
            deprioritized. Either a number or percentile thresholds keyed
            "p50", "p75", "p90" and "p99".
        max_price: Ceilings on what a request may cost, in USD: "prompt" and
            "completion" per million tokens, "request" per request, and
            "image" per image.
    """

    # Why `extra="allow"`, and `| str` and `| dict` on the constrained fields?
    # OpenRouter adds routing options regularly, and one that landed after this
    # model was written should still reach the request rather than be rejected
    # or dropped here.
    model_config = ConfigDict(extra="allow")

    order: list[str] | None = None
    allow_fallbacks: bool | None = None
    require_parameters: bool | None = None
    data_collection: Literal["allow", "deny"] | str | None = None
    zdr: bool | None = None
    enforce_distillable_text: bool | None = None
    only: list[str] | None = None
    ignore: list[str] | None = None
    quantizations: list[str] | None = None
    sort: Literal["price", "throughput", "latency"] | str | dict[str, Any] | None = None
    preferred_min_throughput: float | dict[str, float] | None = None
    preferred_max_latency: float | dict[str, float] | None = None
    max_price: dict[str, float] | None = None


@dataclass
class OpenRouterLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for OpenRouterLLMService.

    Parameters:
        provider: Which upstream providers may serve the request. A plain dict
            is converted to :class:`OpenRouterProviderPreferences`. Left unset,
            or set to ``None``, the request omits it and OpenRouter routes by
            its own default order.
    """

    provider: OpenRouterProviderPreferences | dict[str, Any] | None | NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )

    def __post_init__(self):
        """Coerce a plain ``provider`` dict to :class:`OpenRouterProviderPreferences`."""
        if isinstance(self.provider, dict):
            self.provider = OpenRouterProviderPreferences(**self.provider)


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
        into the JSON body it sends. A ``provider`` already in ``extra_body``,
        supplied through ``Settings.extra``, wins.
        """
        preferences = self._settings.provider
        if not is_given(preferences) or preferences is None:
            return

        preferences = OpenRouterProviderPreferences.model_validate(preferences)
        self._merge_extra_body(params, {"provider": preferences.model_dump(exclude_none=True)})

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
