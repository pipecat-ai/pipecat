#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Mistral LLM service implementation using OpenAI-compatible interface."""

from collections.abc import Sequence
from dataclasses import dataclass

from loguru import logger

from pipecat.adapters.services.mistral_adapter import MistralLLMAdapter
from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.frames.frames import FunctionCallFromLLM
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService


@dataclass
class MistralLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for MistralLLMService."""

    pass


class MistralLLMService(OpenAILLMService):
    """A service for interacting with Mistral's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Mistral's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    """

    # Mistral doesn't support the "developer" message role.
    # This value is used by BaseOpenAILLMService when calling the adapter.
    supports_developer_role = False

    adapter_class = MistralLLMAdapter

    Settings = MistralLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.mistral.ai/v1",
        model: str | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Mistral LLM service.

        Args:
            api_key: The API key for accessing Mistral's API.
            base_url: The base URL for Mistral API. Defaults to "https://api.mistral.ai/v1".
            model: The model identifier to use. Defaults to "mistral-small-latest".

                .. deprecated:: 0.0.105
                    Use ``settings=MistralLLMService.Settings(model=...)`` instead.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(model="mistral-small-latest")

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
        """Create OpenAI-compatible client for Mistral API endpoint.

        Args:
            api_key: The API key for authentication. If None, uses instance key.
            base_url: The base URL for the API. If None, uses instance URL.
            **kwargs: Additional arguments passed to the client constructor.

        Returns:
            An OpenAI-compatible client configured for Mistral API.
        """
        logger.debug(f"Creating Mistral client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)

    async def run_function_calls(self, function_calls: Sequence[FunctionCallFromLLM]):
        """Execute function calls, filtering out already-completed ones.

        Mistral and OpenAI have different function call detection patterns:

        OpenAI (Stream-based detection):

        - Detects function calls only from streaming chunks as the LLM generates them
        - Second LLM completion doesn't re-detect existing tool_calls in message history
        - Function calls execute exactly once

        Mistral (Message-based detection):

        - Detects function calls from the complete message history on each completion
        - Second LLM completion with the response re-detects the same tool_calls from
          previous messages
        - Without filtering, function calls would execute twice

        This method prevents duplicate execution by:

        1. Checking message history for existing tool result messages
        2. Filtering out function calls that already have corresponding results
        3. Only executing function calls that haven't been completed yet

        Note: This filtering prevents duplicate function execution, but the
        on_function_calls_started event may still fire twice due to the detection
        pattern difference. This is expected behavior.

        Args:
            function_calls: The function calls to potentially execute.
        """
        if not function_calls:
            return

        # Filter out function calls that already have results
        calls_to_execute = []

        # Get messages from the first function call's context (they should all have the same context)
        messages = function_calls[0].context.get_messages() if function_calls else []

        # Get all tool_call_ids that already have results
        executed_call_ids = set()
        for msg in messages:
            if msg.get("role") == "tool" and msg.get("tool_call_id"):
                executed_call_ids.add(msg.get("tool_call_id"))

        # Only include function calls that haven't been executed yet
        for call in function_calls:
            if call.tool_call_id not in executed_call_ids:
                calls_to_execute.append(call)
            else:
                logger.trace(
                    f"Skipping already-executed function call: {call.function_name}:{call.tool_call_id}"
                )

        # Call parent method with filtered list
        if calls_to_execute:
            await super().run_function_calls(calls_to_execute)

    def build_chat_completion_params(self, params_from_context: OpenAILLMInvocationParams) -> dict:
        """Build parameters for Mistral chat completion request.

        Handles Mistral-specific parameter mapping (``random_seed`` in place
        of ``seed``). Message-shape fixups required by Mistral are applied
        by :class:`MistralLLMAdapter` upstream.
        """
        params = {
            "model": self._settings.model,
            "stream": True,
            "messages": params_from_context["messages"],
            "tools": params_from_context["tools"],
            "tool_choice": params_from_context["tool_choice"],
            "frequency_penalty": self._settings.frequency_penalty,
            "presence_penalty": self._settings.presence_penalty,
            "temperature": self._settings.temperature,
            "top_p": self._settings.top_p,
            "max_tokens": self._settings.max_tokens,
        }

        # Handle Mistral-specific parameter mapping
        # Mistral uses "random_seed" instead of "seed"
        if self._settings.seed:
            params["random_seed"] = self._settings.seed

        # Add any extra parameters
        params.update(self._settings.extra)

        return params
