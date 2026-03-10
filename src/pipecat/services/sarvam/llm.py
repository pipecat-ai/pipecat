#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Sarvam LLM service implementation using OpenAI-compatible interface."""

import asyncio
import json
from typing import Any, Literal, Mapping, Optional

import httpx
from loguru import logger
from openai import NOT_GIVEN, APITimeoutError
from pydantic import BaseModel

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.sarvam._sdk import sdk_headers

__all__ = ["SarvamLLMService"]


class SarvamLLMService(OpenAILLMService):
    """Sarvam LLM service using Sarvam's OpenAI-compatible chat completions API.

    This service extends ``OpenAILLMService`` while adding Sarvam-specific behavior:

    - model allow-list validation
    - request shaping for Sarvam-compatible parameters
    - Sarvam auth header wiring (``api-subscription-key``)
    - SDK User-Agent propagation on every API call
    - raw Sarvam server error passthrough
    """

    SUPPORTED_MODELS = frozenset({"sarvam-30b", "sarvam-30b-16k", "sarvam-105b", "sarvam-105b-32k"})
    TOOL_CALLING_MODELS = frozenset(
        {"sarvam-30b", "sarvam-30b-16k", "sarvam-105b", "sarvam-105b-32k"}
    )

    class InputParams(OpenAILLMService.InputParams):
        """Configuration parameters for Sarvam LLM service.

        Parameters:
            frequency_penalty: Penalty for frequent tokens (-2.0 to 2.0).
            presence_penalty: Penalty for new tokens (-2.0 to 2.0).
            seed: Random seed for deterministic outputs.
            temperature: Sampling temperature (0.0 to 2.0).
            top_k: Top-k sampling parameter (currently ignored by OpenAI client).
            top_p: Top-p (nucleus) sampling parameter (0.0 to 1.0).
            max_tokens: Maximum tokens in response.
            max_completion_tokens: Maximum completion tokens (not sent to Sarvam API).
            service_tier: Service tier (not sent to Sarvam API).
            extra: Additional model-specific parameters.
            wiki_grounding: Sarvam wiki grounding toggle.
            reasoning_effort: Reasoning effort level (low, medium, high).
        """

        wiki_grounding: Optional[bool] = None
        reasoning_effort: Optional[Literal["low", "medium", "high"]] = None

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "sarvam-30b",
        base_url: str = "https://api.sarvam.ai/v1",
        default_headers: Optional[Mapping[str, str]] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize Sarvam LLM service.

        Args:
            api_key: Sarvam API key used for both OpenAI auth and Sarvam subscription header.
            model: Sarvam model identifier. Supported values: ``sarvam-30b``, ``sarvam-105b``.
            base_url: Sarvam OpenAI-compatible base URL.
            default_headers: Additional HTTP headers to include in requests.
            params: Input parameters for model configuration.
            **kwargs: Additional keyword arguments passed to ``OpenAILLMService``.
        """
        self._validate_model(model)

        params = (params or SarvamLLMService.InputParams()).model_copy(deep=True)
        params.extra = self._build_extra_params(params)

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            default_headers=default_headers,
            params=params,
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
        merged_headers.update(sdk_headers())
        if api_key:
            merged_headers["api-subscription-key"] = api_key

        # Keep SDK User-Agent stable even when caller-provided headers include User-Agent.
        merged_headers["User-Agent"] = sdk_headers()["User-Agent"]

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

        extra = self._settings.extra if isinstance(self._settings.extra, dict) else {}
        if "wiki_grounding" in extra and extra["wiki_grounding"] is not None:
            params["wiki_grounding"] = extra["wiki_grounding"]
        if "reasoning_effort" in extra and extra["reasoning_effort"] is not None:
            params["reasoning_effort"] = extra["reasoning_effort"]

        return params

    async def get_chat_completions(self, params_from_context: OpenAILLMInvocationParams):
        """Get streaming chat completions with Sarvam raw error passthrough."""
        try:
            return await super().get_chat_completions(params_from_context)
        except (APITimeoutError, asyncio.TimeoutError, httpx.TimeoutException):
            raise
        except Exception as e:
            raise RuntimeError(self._format_raw_server_error(e)) from e

    async def run_inference(
        self, context: LLMContext | OpenAILLMContext, max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """Run one-shot inference and preserve Sarvam raw server errors."""
        try:
            return await super().run_inference(context, max_tokens=max_tokens)
        except (APITimeoutError, asyncio.TimeoutError, httpx.TimeoutException):
            raise
        except Exception as e:
            raise RuntimeError(self._format_raw_server_error(e)) from e

    def _validate_model(self, model: str):
        if model not in self.SUPPORTED_MODELS:
            allowed = ", ".join(sorted(self.SUPPORTED_MODELS))
            raise ValueError(f"Unsupported Sarvam LLM model '{model}'. Allowed values: {allowed}.")

    def _build_extra_params(self, params: BaseModel) -> dict[str, Any]:
        extra = dict(getattr(params, "extra", {}) or {})
        if getattr(params, "wiki_grounding", None) is not None:
            extra["wiki_grounding"] = params.wiki_grounding
        if getattr(params, "reasoning_effort", None) is not None:
            extra["reasoning_effort"] = params.reasoning_effort
        return extra

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

        if has_tools and self._settings.model not in self.TOOL_CALLING_MODELS:
            allowed = ", ".join(sorted(self.TOOL_CALLING_MODELS))
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
