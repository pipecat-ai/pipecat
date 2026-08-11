#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Sarvam LLM service implementation using OpenAI-compatible interface."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from loguru import logger
from openai import NOT_GIVEN as OPENAI_NOT_GIVEN

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams, openai_is_given
from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.sarvam._sdk import sdk_headers
from pipecat.utils.types import NOT_GIVEN, NotGiven, is_given


@dataclass
class SarvamLLMSettings(OpenAILLMSettings):
    """Settings for SarvamLLMService.

    Parameters:
        wiki_grounding: Sarvam wiki grounding toggle.
        reasoning_effort: Reasoning effort level (low, medium, high).
    """

    wiki_grounding: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    reasoning_effort: Literal["low", "medium", "high"] | None | NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )


class SarvamLLMService(OpenAILLMService):
    """A service for interacting with Sarvam's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Sarvam's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    """

    # Sarvam doesn't support the "developer" message role.
    # This value is used by BaseOpenAILLMService when calling the adapter.
    supports_developer_role = False

    _SUPPORTED_MODELS = frozenset({"gemma4", "sarvam-105b", "sarvam-105b-conversations"})
    _VISION_MODELS = frozenset({"gemma4"})
    _REASONING_MODELS = frozenset({"gemma4", "sarvam-105b"})
    _WIKI_GROUNDING_MODELS = frozenset({"gemma4", "sarvam-105b"})
    _V1_MODELS = frozenset({"sarvam-105b-conversations"})
    Settings = SarvamLLMSettings
    _settings: Settings

    _DEFAULT_BASE_URL = "https://api.sarvam.ai"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        settings: Settings | None = None,
        default_headers: Mapping[str, str] | None = None,
        **kwargs,
    ):
        """Initialize Sarvam LLM service.

        Args:
            api_key: Sarvam API key used for both OpenAI auth and Sarvam subscription header.
            base_url: Sarvam OpenAI-compatible base URL. When ``None``, resolved
                from the model: ``/v1`` for ``sarvam-105b-conversations``,
                ``/v2`` for all other models.
            settings: Runtime-updatable settings.
            default_headers: Additional HTTP headers to include in requests.
            **kwargs: Additional keyword arguments passed to ``OpenAILLMService``.
        """
        # Initialize only Sarvam-specific defaults; inherited defaults are
        # provided by the OpenAI base service initialization.
        default_settings = self.Settings(
            model="sarvam-105b",
            wiki_grounding=None,
            reasoning_effort=None,
        )

        # Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        model = default_settings.model
        if not isinstance(model, str):
            raise ValueError("Sarvam LLM requires a non-empty model string.")
        self._validate_model(model)

        if base_url is None:
            api_version = "v1" if model in self._V1_MODELS else "v2"
            base_url = f"{self._DEFAULT_BASE_URL}/{api_version}"

        self._api_key = api_key
        self._base_url = base_url
        self._default_headers = default_headers
        self._client_kwargs = kwargs

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

    async def _update_settings(self, delta: Settings) -> dict[str, Any]:
        """Apply a settings delta, validating the model and reconnecting if needed.

        When the model changes, the API endpoint version (``/v1`` vs ``/v2``)
        may need to change, requiring a new client. The model is also validated
        against the supported set.
        """
        if is_given(delta.model) and delta.model is not None:
            if isinstance(delta.model, str):
                self._validate_model(delta.model)

        changed = await super()._update_settings(delta)

        if "model" in changed:
            new_model = self._settings.model
            if isinstance(new_model, str):
                new_api_version = "v1" if new_model in self._V1_MODELS else "v2"
                new_base_url = f"{self._DEFAULT_BASE_URL}/{new_api_version}"
                if new_base_url != self._base_url:
                    self._base_url = new_base_url
                    self._client = self.create_client(
                        api_key=self._api_key,
                        base_url=self._base_url,
                        default_headers=self._default_headers,
                        **self._client_kwargs,
                    )
                    logger.info(
                        f"{self.name}: model changed to '{new_model}', "
                        f"recreated client with base_url {self._base_url}"
                    )

        return changed

    def build_chat_completion_params(self, params_from_context: OpenAILLMInvocationParams) -> dict:
        """Build parameters for Sarvam chat completion request.

        Starts from OpenAI-compatible defaults, then removes unsupported
        request fields and applies Sarvam-specific options.
        """
        self._validate_tool_parameters(params_from_context)
        self._validate_vision_support(params_from_context)

        params = super().build_chat_completion_params(params_from_context)
        params.pop("stream_options", None)
        params.pop("max_completion_tokens", None)
        params.pop("service_tier", None)

        model = self._settings.model

        # wiki_grounding is Sarvam-specific and unknown to the OpenAI SDK,
        # so it must be passed via extra_body to avoid TypeError.
        extra_body = {}
        if (
            model in self._WIKI_GROUNDING_MODELS
            and is_given(self._settings.wiki_grounding)
            and self._settings.wiki_grounding is not None
        ):
            extra_body["wiki_grounding"] = self._settings.wiki_grounding

        if extra_body:
            params["extra_body"] = extra_body

        if (
            model in self._REASONING_MODELS
            and is_given(self._settings.reasoning_effort)
            and self._settings.reasoning_effort is not None
        ):
            params["reasoning_effort"] = self._settings.reasoning_effort

        return params

    def _validate_model(self, model: str):
        if model not in self._SUPPORTED_MODELS:
            allowed = ", ".join(sorted(self._SUPPORTED_MODELS))
            raise ValueError(f"Unsupported Sarvam LLM model '{model}'. Allowed values: {allowed}.")

    def _validate_tool_parameters(self, params_from_context: OpenAILLMInvocationParams):
        tools = params_from_context.get("tools", OPENAI_NOT_GIVEN)
        tool_choice = params_from_context.get("tool_choice", OPENAI_NOT_GIVEN)

        has_tools = (
            openai_is_given(tools)
            and tools is not None
            and (not isinstance(tools, list) or len(tools) > 0)
        )
        has_tool_choice = openai_is_given(tool_choice) and tool_choice is not None

        if has_tool_choice and not has_tools:
            raise ValueError("Sarvam requires non-empty `tools` when `tool_choice` is provided.")

    def _validate_vision_support(self, params_from_context: OpenAILLMInvocationParams):
        model = self._settings.model
        if model in self._VISION_MODELS:
            return

        messages = params_from_context.get("messages", [])
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        raise ValueError(
                            f"Model '{model}' does not support image input. "
                            f"Use a vision-capable model instead."
                        )
