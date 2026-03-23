#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Sarvam LLM service implementation using OpenAI-compatible interface."""

from dataclasses import dataclass, field
from typing import Literal, Mapping, Optional

from loguru import logger
from openai import NOT_GIVEN

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.sarvam._sdk import sdk_headers
from pipecat.services.settings import NOT_GIVEN as _NOT_GIVEN
from pipecat.services.settings import _NotGiven, is_given


@dataclass
class SarvamLLMSettings(OpenAILLMSettings):
    """Settings for SarvamLLMService.

    Parameters:
        wiki_grounding: Sarvam wiki grounding toggle.
        reasoning_effort: Reasoning effort level (low, medium, high).
    """

    wiki_grounding: bool | None | _NotGiven = field(default_factory=lambda: _NOT_GIVEN)
    reasoning_effort: Literal["low", "medium", "high"] | None | _NotGiven = field(
        default_factory=lambda: _NOT_GIVEN
    )


class SarvamLLMService(OpenAILLMService):
    """A service for interacting with Sarvam's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Sarvam's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    """

    _SUPPORTED_MODELS = frozenset(
        {"sarvam-30b", "sarvam-30b-16k", "sarvam-105b", "sarvam-105b-32k"}
    )
    _TOOL_CALLING_MODELS = _SUPPORTED_MODELS
    Settings = SarvamLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.sarvam.ai/v1",
        settings: Optional[Settings] = None,
        default_headers: Optional[Mapping[str, str]] = None,
        **kwargs,
    ):
        """Initialize Sarvam LLM service.

        Args:
            api_key: Sarvam API key used for both OpenAI auth and Sarvam subscription header.
            base_url: Sarvam OpenAI-compatible base URL.
            settings: Runtime-updatable settings.
            default_headers: Additional HTTP headers to include in requests.
            **kwargs: Additional keyword arguments passed to ``OpenAILLMService``.
        """
        # Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="sarvam-30b",
            wiki_grounding=None,
            reasoning_effort=None,
        )

        # Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        self._validate_model(default_settings.model)

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            settings=default_settings,
            default_headers=default_headers,
            **kwargs,
        )

    def create_client(
        self,
        api_key=None,
        base_url=None,
        organization=None,
        project=None,
        default_headers=None,
        **kwargs,
    ):
        """Create OpenAI-compatible client for Sarvam API endpoint.

        Ensures Sarvam auth and SDK identification headers are always attached.
        """
        merged_headers = dict(default_headers or {})
        # sdk_headers() carries Pipecat User-Agent and should override caller-provided value.
        merged_headers.update(sdk_headers())
        if api_key:
            merged_headers["api-subscription-key"] = api_key

        logger.debug(f"Creating Sarvam client with API {base_url}")
        return super().create_client(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            project=project,
            default_headers=merged_headers,
            **kwargs,
        )

    def build_chat_completion_params(self, params_from_context: OpenAILLMInvocationParams) -> dict:
        """Build parameters for Sarvam chat completion request.

        Starts from OpenAI-compatible defaults, then removes unsupported
        request fields and applies Sarvam-specific options.
        """
        self._validate_tool_parameters(params_from_context)

        params = super().build_chat_completion_params(params_from_context)
        params.pop("stream_options", None)
        params.pop("max_completion_tokens", None)
        params.pop("service_tier", None)

        if is_given(self._settings.wiki_grounding) and self._settings.wiki_grounding is not None:
            params["wiki_grounding"] = self._settings.wiki_grounding
        if (
            is_given(self._settings.reasoning_effort)
            and self._settings.reasoning_effort is not None
        ):
            params["reasoning_effort"] = self._settings.reasoning_effort

        return params

    def _validate_model(self, model: str):
        if model not in self._SUPPORTED_MODELS:
            allowed = ", ".join(sorted(self._SUPPORTED_MODELS))
            raise ValueError(f"Unsupported Sarvam LLM model '{model}'. Allowed values: {allowed}.")

    def _validate_tool_parameters(self, params_from_context: OpenAILLMInvocationParams):
        tools = params_from_context.get("tools", NOT_GIVEN)
        tool_choice = params_from_context.get("tool_choice", NOT_GIVEN)

        has_tools = (
            tools is not NOT_GIVEN
            and tools is not None
            and (not isinstance(tools, list) or len(tools) > 0)
        )
        has_tool_choice = tool_choice is not NOT_GIVEN and tool_choice is not None

        if has_tool_choice and not has_tools:
            raise ValueError("Sarvam requires non-empty `tools` when `tool_choice` is provided.")
