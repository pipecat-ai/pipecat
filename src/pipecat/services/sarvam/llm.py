#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Sarvam LLM service implementation using OpenAI-compatible interface."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from loguru import logger
from openai import NOT_GIVEN as OPENAI_NOT_GIVEN

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams, openai_is_given
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.sarvam._sdk import sdk_headers
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given, is_given


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

    _SUPPORTED_MODELS = frozenset({"gemma4", "glm5.2", "sarvam-105b", "sarvam-105b-conversations"})
    _VISION_MODELS = frozenset({"gemma4"})
    _REASONING_MODELS = frozenset({"gemma4", "glm5.2", "sarvam-105b"})
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

    async def _process_context(self, context: LLMContext):
        """Process a context, pushing an error frame on misconfiguration.

        Validates tool parameters, vision support, and capability support
        before delegating to the base class. Validation failures are pushed as
        non-fatal error frames rather than raised, so an ``LLMSwitcher`` can
        fall back to another service at runtime.
        """
        error = self._validate_request(self._invocation_params(context))
        if error:
            await self.push_error(error)
            return

        await super()._process_context(context)

    async def run_inference(
        self,
        context: LLMContext,
        max_tokens: int | None = None,
        system_instruction: str | None = None,
    ) -> str | None:
        """Run inference, pushing an error frame on misconfiguration.

        Validates tool parameters, vision support, and capability support
        before delegating to the base class. Validation failures are pushed as
        non-fatal error frames rather than raised, so an ``LLMSwitcher`` can
        fall back to another service at runtime.

        Args:
            context: The LLM context containing conversation history.
            max_tokens: Optional maximum number of tokens to generate.
            system_instruction: Optional system instruction for this inference.

        Returns:
            The LLM's response, or None if the request is invalid or produced
            no response.
        """
        error = self._validate_request(
            self._invocation_params(context, system_instruction=system_instruction)
        )
        if error:
            await self.push_error(error)
            return None

        return await super().run_inference(
            context, max_tokens=max_tokens, system_instruction=system_instruction
        )

    def _invocation_params(
        self, context: LLMContext, system_instruction: str | None = None
    ) -> OpenAILLMInvocationParams:
        """Derive the invocation params the request will be built from."""
        adapter = self.get_llm_adapter()
        return adapter.get_llm_invocation_params(
            context,
            system_instruction=system_instruction
            or assert_given(self._settings.system_instruction),
            convert_developer_to_user=not self.supports_developer_role,
        )

    def build_chat_completion_params(self, params_from_context: OpenAILLMInvocationParams) -> dict:
        """Build parameters for Sarvam chat completion request.

        Starts from OpenAI-compatible defaults, then removes unsupported
        request fields and applies Sarvam-specific options.
        """
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
            params.setdefault("extra_body", {}).update(extra_body)

        if (
            model in self._REASONING_MODELS
            and is_given(self._settings.reasoning_effort)
            and self._settings.reasoning_effort is not None
        ):
            params["reasoning_effort"] = self._settings.reasoning_effort

        return params

    def _validate_request(self, params_from_context: OpenAILLMInvocationParams) -> str | None:
        """Run all pre-request validations, returning an error message or None.

        Returns a human-readable error string when the current configuration
        is incompatible with the selected model (e.g. image input on a
        non-vision model, or ``reasoning_effort`` on a model that doesn't
        support it). Returns ``None`` when the request is valid.
        """
        error = self._check_tool_parameters(params_from_context)
        if error:
            return error

        error = self._check_vision_support(params_from_context)
        if error:
            return error

        return self._check_capability_support()

    def _validate_model(self, model: str):
        if model not in self._SUPPORTED_MODELS:
            allowed = ", ".join(sorted(self._SUPPORTED_MODELS))
            raise ValueError(f"Unsupported Sarvam LLM model '{model}'. Allowed values: {allowed}.")

    def _check_tool_parameters(self, params_from_context: OpenAILLMInvocationParams) -> str | None:
        tools = params_from_context.get("tools", OPENAI_NOT_GIVEN)
        tool_choice = params_from_context.get("tool_choice", OPENAI_NOT_GIVEN)

        has_tools = (
            openai_is_given(tools)
            and tools is not None
            and (not isinstance(tools, list) or len(tools) > 0)
        )
        has_tool_choice = openai_is_given(tool_choice) and tool_choice is not None

        if has_tool_choice and not has_tools:
            return "Sarvam requires non-empty `tools` when `tool_choice` is provided."
        return None

    def _check_vision_support(self, params_from_context: OpenAILLMInvocationParams) -> str | None:
        model = self._settings.model
        if model in self._VISION_MODELS:
            return None

        messages = params_from_context.get("messages", [])
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        return (
                            f"Model '{model}' does not support image input. "
                            f"Use a vision-capable model instead."
                        )
        return None

    def _check_capability_support(self) -> str | None:
        """Check whether configured capabilities are supported by the current model.

        Returns an error message when ``reasoning_effort`` or ``wiki_grounding``
        is configured for a model that doesn't support it, so the user gets a
        clear error instead of the setting being silently dropped.
        """
        model = self._settings.model

        if (
            is_given(self._settings.reasoning_effort)
            and self._settings.reasoning_effort is not None
            and model not in self._REASONING_MODELS
        ):
            return (
                f"Model '{model}' does not support reasoning_effort. "
                f"Use a reasoning-capable model instead."
            )

        if (
            is_given(self._settings.wiki_grounding)
            and self._settings.wiki_grounding is not None
            and model not in self._WIKI_GROUNDING_MODELS
        ):
            return (
                f"Model '{model}' does not support wiki_grounding. "
                f"Use a model that supports it instead."
            )

        return None
