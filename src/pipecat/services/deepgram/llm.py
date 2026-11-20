#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deepgram LLM service, backed by Deepgram's OpenAI-compatible ``/v1/think`` endpoint."""

from dataclasses import dataclass
from typing import Any

from loguru import logger
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from openai._types import NotGiven as OpenAINotGiven

from pipecat.adapters.llm.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.types import NotGiven

# `/v1/think` rejects a request carrying any other body field.
_THINK_FIELDS = {
    "model",
    "temperature",
    "reasoning_effort",
    "messages",
    "tools",
    "stream",
    "stream_options",
}
# Options the OpenAI SDK consumes itself rather than sending as body fields.
_SDK_OPTIONS = {"extra_headers", "extra_query", "extra_body", "timeout"}


@dataclass
class DeepgramLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for DeepgramLLMService."""

    pass


async def _route_to_think(request: Any):  # httpx or httpx2, per the openai major
    # The OpenAI SDK hardcodes `/chat/completions`; Deepgram serves the same body at `/think`.
    if request.url.path.endswith("/chat/completions"):
        path = request.url.path.removesuffix("/chat/completions") + "/think"
        request.url = request.url.copy_with(path=path)


class DeepgramLLMService(OpenAILLMService):
    """LLM service for Deepgram-hosted models via ``/v1/think``.

    ``/v1/think`` takes an OpenAI chat-completions request with ``Authorization: Token
    <key>``. Of the settings, only ``model``, ``temperature`` and ``reasoning_effort``
    reach it; others are dropped with a warning. It does not accept images, multi-part
    message content, or tools without a description.
    """

    Settings = DeepgramLLMSettings
    _settings: Settings

    supports_developer_role = False
    _requires_api_key = True

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.deepgram.com/v1",
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Deepgram LLM service.

        Args:
            api_key: Deepgram API key.
            base_url: Deepgram API base URL.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to OpenAILLMService.
        """
        default_settings = self.Settings(model="think-fast")
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(api_key=api_key, base_url=base_url, settings=default_settings, **kwargs)
        self._warned_dropped: set[str] = set()

    def build_chat_completion_params(self, params_from_context: OpenAILLMInvocationParams) -> dict:
        """Build chat completion parameters, keeping only those ``/v1/think`` accepts.

        Args:
            params_from_context: Messages, tools and tool choice from the context.

        Returns:
            Parameters for the chat completion request.
        """
        params = super().build_chat_completion_params(params_from_context)
        kept = {k: v for k, v in params.items() if k in _THINK_FIELDS | _SDK_OPTIONS}
        dropped = {
            k
            for k, v in params.items()
            if k not in kept and v is not None and not isinstance(v, (NotGiven, OpenAINotGiven))
        }
        if new := dropped - self._warned_dropped:
            logger.warning(f"{self}: Deepgram /v1/think does not support {sorted(new)}; ignoring")
            self._warned_dropped |= new
        return kept

    def create_client(self, api_key=None, base_url=None, default_headers=None, **kwargs):
        """Create an OpenAI client that speaks Deepgram's path and auth scheme.

        Args:
            api_key: Deepgram API key.
            base_url: Deepgram API base URL.
            default_headers: Additional HTTP headers.
            **kwargs: Ignored; ``organization`` and ``project`` are OpenAI-only.

        Returns:
            Configured AsyncOpenAI client instance.
        """
        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=DefaultAsyncHttpxClient(event_hooks={"request": [_route_to_think]}),
            default_headers={**(default_headers or {}), "Authorization": f"Token {api_key}"},
        )
