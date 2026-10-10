#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Perplexity LLM service implementations.

This module provides services for Perplexity's Agent API, which answers with web
search over the OpenAI Responses protocol, and for its Sonar chat completions
API, which Perplexity no longer supports.
"""

from dataclasses import dataclass, field
from typing import Any

from openai.types.responses import ResponseUsage

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.adapters.services.open_ai_responses_adapter import (
    OpenAIResponsesLLMInvocationParams,
)
from pipecat.adapters.services.perplexity_adapter import PerplexityLLMAdapter
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.responses.llm import (
    OpenAIResponsesHttpLLMService,
    OpenAIResponsesLLMSettings,
    _model_supports_reasoning,
    _rejects_effort_none,
)
from pipecat.utils.deprecation import deprecated
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given


@dataclass
class PerplexityAgentLLMSettings(OpenAIResponsesLLMSettings):
    """Settings for PerplexityAgentLLMService.

    Parameters:
        preset: Perplexity preset that sets the model, its instructions, its
            search budget and how much it reasons, such as ``"fast"``,
            ``"low"``, ``"medium"``, ``"high"`` or ``"xhigh"``. ``model``
            replaces the preset's model; set ``model=None`` to use it. With
            ``None``, a model must be set.
        web_search: Whether to offer the ``web_search`` tool on every request,
            alongside any tools from the context. A preset searches the web
            whatever this is set to.
        max_steps: Maximum number of search and reasoning steps per response.
            ``None`` leaves Perplexity's default.
    """

    preset: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    web_search: bool | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    max_steps: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


_MODEL_OR_PRESET_REQUIRED = "PerplexityAgentLLMService needs a `model`, a `preset`, or both."


def _has_model_or_preset(settings: PerplexityAgentLLMSettings) -> bool:
    return bool(assert_given(settings.model) or assert_given(settings.preset))


class PerplexityAgentLLMService(OpenAIResponsesHttpLLMService):
    """A service for Perplexity's Agent API.

    The Agent API answers questions with web search, using models from Perplexity
    and other providers. It speaks the OpenAI Responses protocol over HTTP, so
    function calling and multi-turn context work as they do with
    :class:`~pipecat.services.openai.responses.llm.OpenAIResponsesHttpLLMService`.

    By default the service uses Perplexity's ``fast`` preset, its lowest-latency
    configuration, with ``openai/gpt-5.6-luna`` in place of the preset's model and
    with web search on. A preset decides how much to reason; without one, reasoning
    is off for OpenAI models unless ``reasoning`` is set. Answers can carry
    citation markers such as ``[1]``. Anthropic models need
    ``max_completion_tokens`` set.

    Example::

        llm = PerplexityAgentLLMService(
            api_key=os.getenv("PERPLEXITY_API_KEY"),
            settings=PerplexityAgentLLMService.Settings(
                system_instruction="You are a helpful assistant.",
            ),
        )
    """

    Settings = PerplexityAgentLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.perplexity.ai/v1",
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Perplexity Agent API LLM service.

        Args:
            api_key: The API key for accessing Perplexity's API.
            base_url: The base URL for Perplexity's API.
            settings: Runtime-updatable settings.
            **kwargs: Additional keyword arguments passed to
                OpenAIResponsesHttpLLMService.

        Raises:
            ValueError: If the settings leave neither a model nor a preset.
        """
        default_settings = self.Settings(
            model="openai/gpt-5.6-luna", preset="fast", web_search=True, max_steps=None
        )

        if settings is not None:
            default_settings.apply_update(settings)

        if not _has_model_or_preset(default_settings):
            raise ValueError(_MODEL_OR_PRESET_REQUIRED)

        super().__init__(api_key=api_key, base_url=base_url, settings=default_settings, **kwargs)

    async def _update_settings(self, delta: Settings) -> dict[str, Any]:
        """Apply a settings delta, reporting an update that leaves no model or preset.

        Args:
            delta: A settings delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)
        if not _has_model_or_preset(self._settings):
            await self.push_error(error_msg=_MODEL_OR_PRESET_REQUIRED)
        return changed

    def _build_response_params(self, invocation_params: OpenAIResponsesLLMInvocationParams) -> dict:
        """Build parameters for an Agent API call.

        Args:
            invocation_params: Parameters derived from the LLM context.

        Returns:
            Dictionary of parameters for the Agent API call.
        """
        params = super()._build_response_params(invocation_params)

        if not params.get("model"):
            # The preset chooses the model.
            params.pop("model", None)

        if assert_given(self._settings.web_search):
            tools = list(params.get("tools") or [])
            if not any(tool.get("type") == "web_search" for tool in tools):
                tools.append({"type": "web_search"})
            params["tools"] = tools

        # The OpenAI SDK rejects fields it doesn't know as keyword arguments, so
        # Perplexity's own fields go in the request body.
        perplexity_fields: dict[str, Any] = {}
        preset = assert_given(self._settings.preset)
        if preset:
            perplexity_fields["preset"] = preset
        max_steps = assert_given(self._settings.max_steps)
        if max_steps is not None:
            perplexity_fields["max_steps"] = max_steps
        if perplexity_fields:
            params["extra_body"] = {**params.get("extra_body", {}), **perplexity_fields}

        return params

    def _maybe_disable_reasoning(self, params: dict):
        """Disable reasoning by default on OpenAI models for real-time voice.

        Applies the same rule as the OpenAI service to ``openai/`` models when no
        preset is set. A preset decides how much to reason, and other providers'
        models keep Perplexity's default.

        Args:
            params: The response params dict (modified in place).
        """
        if assert_given(self._settings.preset):
            return
        model = assert_given(self._settings.model)
        provider, _, name = (model or "").partition("/")
        if (
            provider == "openai"
            and _model_supports_reasoning(name)
            and not _rejects_effort_none(name)
        ):
            params["reasoning"] = {"effort": "none"}

    def _token_usage(self, usage: ResponseUsage) -> LLMTokenUsage:
        """Convert a completed response's usage into Pipecat's token usage.

        Perplexity reports tokens written to the prompt cache as
        ``cache_creation_input_tokens``.

        Args:
            usage: The usage reported with the completed response.

        Returns:
            The token usage to report.
        """
        tokens = super()._token_usage(usage)
        cache_creation = getattr(usage.input_tokens_details, "cache_creation_input_tokens", None)
        if cache_creation:
            tokens.cache_creation_input_tokens = cache_creation
        return tokens


@dataclass
class PerplexityLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for PerplexityLLMService."""

    pass


@deprecated(
    "`PerplexityLLMService` is deprecated since 1.13.0 and will be removed in 2.0.0. "
    "Use `PerplexityAgentLLMService` instead."
)
class PerplexityLLMService(OpenAILLMService):
    """A service for Perplexity's Sonar chat completions API.

    This service extends OpenAILLMService to work with Perplexity's API while maintaining
    compatibility with the OpenAI-style interface.

    .. deprecated:: 1.13.0
        Use :class:`PerplexityAgentLLMService` instead. Perplexity has ended support
        for Sonar chat completions in favor of its Agent API. Will be removed in 2.0.0.
    """

    adapter_class = PerplexityLLMAdapter
    # Perplexity doesn't support the "developer" message role.
    # This value is used by BaseOpenAILLMService when calling the adapter.
    supports_developer_role = False

    Settings = PerplexityLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.perplexity.ai",
        model: str | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Perplexity LLM service.

        Args:
            api_key: The API key for accessing Perplexity's API.
            base_url: The base URL for Perplexity's API. Defaults to "https://api.perplexity.ai".
            model: The model identifier to use. Defaults to "sonar".

                .. deprecated:: 0.0.105
                    Use ``settings=PerplexityLLMService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(model="sonar")

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        # 3. (No step 3, as there's no params object to apply)

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(api_key=api_key, base_url=base_url, settings=default_settings, **kwargs)

    def build_chat_completion_params(self, params_from_context: OpenAILLMInvocationParams) -> dict:
        """Build parameters for Perplexity chat completion request.

        Perplexity uses a subset of OpenAI parameters and doesn't support tools.

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
            "messages": params_from_context["messages"],
        }

        # Add OpenAI-compatible parameters if they're set
        if self._settings.frequency_penalty is not None:
            params["frequency_penalty"] = self._settings.frequency_penalty
        if self._settings.presence_penalty is not None:
            params["presence_penalty"] = self._settings.presence_penalty
        if self._settings.temperature is not None:
            params["temperature"] = self._settings.temperature
        if self._settings.top_p is not None:
            params["top_p"] = self._settings.top_p
        if self._settings.max_tokens is not None:
            params["max_tokens"] = self._settings.max_tokens

        return params
