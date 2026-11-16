#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Smallest AI LLM service implementation using OpenAI-compatible interface.

This module provides an integration with Electron, Smallest AI's LLM, served
through the Waves API's OpenAI-compatible chat completions endpoint.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from loguru import logger

from pipecat import version as pipecat_version
from pipecat.adapters.llm.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.types import NOT_GIVEN, NotGiven


@dataclass
class SmallestLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for SmallestLLMService.

    Parameters:
        prompt_cache_key: Key grouping the requests of one conversation, so
            each turn is served where that conversation is already cached.
            Use a fresh value per conversation and never share one between
            conversations.
    """

    prompt_cache_key: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class SmallestLLMService(OpenAILLMService):
    """A service for interacting with Smallest AI's Electron model using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Smallest AI's Waves API
    endpoint while maintaining full compatibility with OpenAI's interface and
    functionality.
    """

    # Electron doesn't support the "developer" message role.
    # This value is used by BaseOpenAILLMService when calling the adapter.
    supports_developer_role = False

    Settings = SmallestLLMSettings
    _settings: Settings
    _requires_api_key = True

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.smallest.ai/waves/v1",
        settings: Settings | None = None,
        default_headers: Mapping[str, str] | None = None,
        **kwargs,
    ):
        """Initialize Smallest AI LLM service.

        Args:
            api_key: The API key for accessing Smallest AI's API.
            base_url: The base URL for Smallest AI's API. Defaults to
                "https://api.smallest.ai/waves/v1".
            settings: Runtime-updatable settings. Defaults to Electron, Smallest
                AI's LLM, via ``settings=SmallestLLMService.Settings(model="electron")``.
            default_headers: Additional HTTP headers to include in requests.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        default_settings = self.Settings(model="electron", prompt_cache_key=None)

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            settings=default_settings,
            default_headers=default_headers,
            **kwargs,
        )

    def create_client(self, api_key=None, base_url=None, default_headers=None, **kwargs):
        """Create OpenAI-compatible client for Smallest AI's API endpoint.

        Args:
            api_key: The API key to use for the client. If None, uses instance api_key.
            base_url: The base URL for the API. If None, uses instance base_url.
            default_headers: Additional HTTP headers. Smallest's own source and
                version identification headers are always included.
            **kwargs: Additional keyword arguments passed to the parent create_client method.

        Returns:
            An OpenAI-compatible client configured for Smallest AI's API.
        """
        merged_headers = dict(default_headers or {})
        merged_headers.update(
            {
                "X-Source": "pipecat",
                "X-Pipecat-Version": pipecat_version(),
            }
        )

        logger.debug(f"Creating Smallest AI client with api {base_url}")
        return super().create_client(
            api_key=api_key, base_url=base_url, default_headers=merged_headers, **kwargs
        )

    def build_chat_completion_params(self, params_from_context: OpenAILLMInvocationParams) -> dict:
        """Build parameters for a Smallest AI chat completion request.

        Extends the base OpenAI parameters with Electron's ``prompt_cache_key``.

        Args:
            params_from_context: Parameters, derived from the LLM context, to
                use for the chat completion. Contains messages, tools, and tool
                choice.

        Returns:
            Dictionary of parameters for the chat completion request.
        """
        params = super().build_chat_completion_params(params_from_context)

        self._merge_extra_body(params, {"prompt_cache_key": self._settings.prompt_cache_key})

        return params
