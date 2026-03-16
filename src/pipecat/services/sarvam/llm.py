#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Sarvam LLM service implementation using OpenAI-compatible interface."""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Literal, Mapping, Optional, TypeVar

import httpx
from loguru import logger
from openai import NOT_GIVEN, APITimeoutError, AsyncStream
from openai.types.chat import ChatCompletionChunk

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.sarvam._sdk import sdk_headers
from pipecat.services.settings import NOT_GIVEN as _NOT_GIVEN
from pipecat.services.settings import _NotGiven, _warn_deprecated_param, is_given

_T = TypeVar("_T")


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
    """Sarvam LLM service using Sarvam's OpenAI-compatible chat completions API.

    This service extends ``OpenAILLMService`` while adding Sarvam-specific behavior:

    - model allow-list validation
    - request shaping for Sarvam-compatible parameters
    - Sarvam auth header wiring (``api-subscription-key``)
    - SDK User-Agent propagation on every API call
    - raw Sarvam server error passthrough
    """

    _SUPPORTED_MODELS = frozenset(
        {"sarvam-30b", "sarvam-30b-16k", "sarvam-105b", "sarvam-105b-32k"}
    )
    _TOOL_CALLING_MODELS = _SUPPORTED_MODELS
    Settings = SarvamLLMSettings
    _settings: SarvamLLMSettings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.sarvam.ai/v1",
        model: Optional[str] = None,
        settings: Optional[SarvamLLMSettings] = None,
        default_headers: Optional[Mapping[str, str]] = None,
        **kwargs,
    ):
        """Initialize Sarvam LLM service.

        Args:
            api_key: Sarvam API key used for both OpenAI auth and Sarvam subscription header.
            base_url: Sarvam OpenAI-compatible base URL.
            model: Sarvam model identifier. Supported values: ``sarvam-30b``,
                ``sarvam-30b-16k``, ``sarvam-105b``, ``sarvam-105b-32k``.

                .. deprecated:: 0.0.105
                    Use ``settings=SarvamLLMSettings(model=...)`` instead.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            default_headers: Additional HTTP headers to include in requests.
            **kwargs: Additional keyword arguments passed to ``OpenAILLMService``.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = SarvamLLMSettings(model="sarvam-30b")

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            # Keep deprecated init arg for backward compatibility while steering callers
            # to settings=SarvamLLMService.Settings(model=...).
            _warn_deprecated_param("model", SarvamLLMSettings, "model")
            default_settings.model = model

        # 3. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        # BaseOpenAILLMService currently stores settings as OpenAILLMSettings.
        # Preserve Sarvam-only runtime knobs in ``extra`` so they survive
        # initialization and future update frames.
        default_settings.extra = dict(default_settings.extra)
        default_settings.extra.update(self._extract_sarvam_extra_from_settings(default_settings))

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

        # Sarvam-only fields are bridged through settings.extra (see __init__ and _update_settings).
        extra = self._settings.extra if isinstance(self._settings.extra, dict) else {}
        if "wiki_grounding" in extra and extra["wiki_grounding"] is not None:
            params["wiki_grounding"] = extra["wiki_grounding"]
        if "reasoning_effort" in extra and extra["reasoning_effort"] is not None:
            params["reasoning_effort"] = extra["reasoning_effort"]

        return params

    async def _update_settings(self, delta: OpenAILLMSettings) -> dict[str, Any]:
        """Apply settings updates, preserving Sarvam-specific runtime knobs."""
        # LLMUpdateSettingsFrame commonly carries OpenAILLMSettings deltas.
        # Lift Sarvam-only fields into delta.extra before delegating to base.
        sarvam_extra = self._extract_sarvam_extra_from_settings(delta)
        if sarvam_extra:
            delta.extra = dict(delta.extra)
            delta.extra.update(sarvam_extra)

        return await super()._update_settings(delta)

    async def _call_with_raw_sarvam_errors(self, awaitable: Awaitable[_T]) -> _T:
        """Await an OpenAI call while preserving Sarvam raw error payloads.

        BaseOpenAILLMService handles pipeline-frame exceptions via push_error(),
        but direct helper methods like ``get_chat_completions`` and
        ``run_inference`` are often consumed directly. We normalize those errors
        here so applications consistently receive server-provided messages.
        """
        try:
            return await awaitable
        except (APITimeoutError, asyncio.TimeoutError, httpx.TimeoutException):
            raise
        except Exception as e:
            raise RuntimeError(self._format_raw_server_error(e)) from e

    async def get_chat_completions(
        self, params_from_context: OpenAILLMInvocationParams
    ) -> AsyncStream[ChatCompletionChunk]:
        """Get streaming chat completions with Sarvam raw error passthrough."""
        return await self._call_with_raw_sarvam_errors(
            super().get_chat_completions(params_from_context)
        )

    async def run_inference(
        self,
        context: LLMContext | OpenAILLMContext,
        max_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None,
    ) -> Optional[str]:
        """Run one-shot inference and preserve Sarvam raw server errors."""
        return await self._call_with_raw_sarvam_errors(
            super().run_inference(
                context,
                max_tokens=max_tokens,
                system_instruction=system_instruction,
            )
        )

    def _validate_model(self, model: str):
        if model not in self._SUPPORTED_MODELS:
            allowed = ", ".join(sorted(self._SUPPORTED_MODELS))
            raise ValueError(f"Unsupported Sarvam LLM model '{model}'. Allowed values: {allowed}.")

    def _extract_sarvam_extra_from_settings(self, settings_obj: Any) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        wiki_grounding = getattr(settings_obj, "wiki_grounding", _NOT_GIVEN)
        if is_given(wiki_grounding):
            updates["wiki_grounding"] = wiki_grounding

        reasoning_effort = getattr(settings_obj, "reasoning_effort", _NOT_GIVEN)
        if is_given(reasoning_effort):
            updates["reasoning_effort"] = reasoning_effort

        return updates

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

        # Validate early to provide deterministic errors before network calls.
        if has_tools and self._settings.model not in self._TOOL_CALLING_MODELS:
            allowed = ", ".join(sorted(self._TOOL_CALLING_MODELS))
            raise ValueError(
                f"Model '{self._settings.model}' does not support tools. "
                f"Supported models: {allowed}."
            )

    def _format_raw_server_error(self, error: Exception) -> str:
        raw_message = self._extract_raw_server_message(error)
        return f"Sarvam server error: {raw_message}"

    def _extract_raw_server_message(self, error: Exception) -> str:
        body = getattr(error, "body", None)
        if body is not None:
            return self._payload_to_message(body)

        response = getattr(error, "response", None)
        if response is not None:
            try:
                return self._payload_to_message(response.json())
            except Exception:
                text = getattr(response, "text", None)
                if text:
                    return str(text)

        return str(error)

    def _payload_to_message(self, payload: Any) -> str:
        if isinstance(payload, dict):
            error_obj = payload.get("error")
            if isinstance(error_obj, dict) and isinstance(error_obj.get("message"), str):
                return error_obj["message"]
            if isinstance(payload.get("message"), str):
                return payload["message"]
            return json.dumps(payload, ensure_ascii=False)
        if isinstance(payload, list):
            return json.dumps(payload, ensure_ascii=False)
        return str(payload)
