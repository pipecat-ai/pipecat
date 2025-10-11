#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""CometAPI LLM service implementation with model curation utilities."""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence

from loguru import logger

from pipecat.services.openai.llm import OpenAILLMService

# The ignore patterns come directly from the CometAPI integration brief.
_COMETAPI_IGNORE_PATTERNS: Sequence[str] = (
    "dall-e",
    "dalle",
    "midjourney",
    "mj_",
    "stable-diffusion",
    "sd-",
    "flux-",
    "playground-v",
    "ideogram",
    "recraft-",
    "black-forest-labs",
    "/recraft-v3",
    "recraftv3",
    "stability-ai/",
    "sdxl",
    "suno_",
    "tts",
    "whisper",
    "runway",
    "luma_",
    "luma-",
    "veo",
    "kling_",
    "minimax_video",
    "hunyuan-t1",
    "embedding",
    "search-gpts",
    "files_retrieve",
    "moderation",
    "deepl-",
)


_COMETAPI_RECOMMENDED_MODELS: Sequence[str] = (
    "gpt-5-chat-latest",
    "chatgpt-4o-latest",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5",
    "gpt-4.1",
    "gpt-4o-mini",
    "o4-mini-2025-04-16",
    "o3-pro-2025-06-10",
    "claude-opus-4-1-20250805",
    "claude-opus-4-1-20250805-thinking",
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-20250514-thinking",
    "claude-3-7-sonnet-latest",
    "claude-3-5-haiku-latest",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "grok-4-0709",
    "grok-4-fast-non-reasoning",
    "grok-4-fast-reasoning",
    "deepseek-v3.1",
    "deepseek-v3",
    "deepseek-r1-0528",
    "deepseek-chat",
    "deepseek-reasoner",
    "qwen3-30b-a3b",
    "qwen3-coder-plus-2025-07-22",
)


class CometAPILLMService(OpenAILLMService):
    """LLM service for the CometAPI provider.

    The API is OpenAI-compatible, so we delegate the streaming logic to
    :class:`~pipecat.services.openai.llm.OpenAILLMService`.  We additionally
    expose helper methods to surface CometAPI's recommended chat model list
    and to filter the provider's `/v1/models` response so that only chat-ready
    models are returned to application code.
    """

    DEFAULT_BASE_URL = "https://api.cometapi.com/v1"
    _IGNORE_REGEXPS = tuple(
        re.compile(pattern, re.IGNORECASE) for pattern in _COMETAPI_IGNORE_PATTERNS
    )

    def __init__(
        self,
        *,
        api_key: str,
        model: str = _COMETAPI_RECOMMENDED_MODELS[0],
        base_url: str = DEFAULT_BASE_URL,
        **kwargs,
    ):
        """Instantiate the CometAPI LLM service.

        Args:
            api_key: CometAPI access token (`COMETAPI_KEY`).
            model: Default model to use for completions. Defaults to the first
                recommended model (``"gpt-5-chat-latest"``).
            base_url: Base URL for the CometAPI endpoint.
            **kwargs: Forwarded to :class:`OpenAILLMService`.
        """
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    # ------------------------------------------------------------------
    # Model utilities
    # ------------------------------------------------------------------
    @classmethod
    def recommended_models(cls) -> List[str]:
        """Return the recommended CometAPI chat model list.

        The order matches the integration brief exactly.
        """
        return list(_COMETAPI_RECOMMENDED_MODELS)

    @classmethod
    def ignore_patterns(cls) -> List[str]:
        """Expose the ignore pattern list for external tooling/tests."""
        return list(_COMETAPI_IGNORE_PATTERNS)

    @classmethod
    def is_chat_model(cls, model_id: str) -> bool:
        """Check whether a model identifier should be treated as chat-capable."""
        if not model_id:
            return False
        return not any(regex.search(model_id) for regex in cls._IGNORE_REGEXPS)

    async def fetch_chat_models(self, *, include_recommended: bool = True) -> List[str]:
        """Retrieve chat-ready model identifiers from CometAPI.

        Args:
            include_recommended: If True (default), ensure the returned list
                contains the full recommended model list in canonical order even
                when the remote API omits entries.

        Returns:
            Ordered list of chat-ready model identifiers.
        """
        models: List[str] = []

        try:
            response = await self._client.models.list()
            ids = (getattr(model, "id", None) for model in getattr(response, "data", []))
            models = [model_id for model_id in ids if model_id and self.is_chat_model(model_id)]
        except Exception as error:  # pragma: no cover - defensive logging branch
            logger.warning("Failed to fetch CometAPI models; falling back to defaults: {}", error)
            models = []

        ordered: List[str] = []

        def _extend_unique(values: Iterable[str]):
            for value in values:
                if value not in ordered:
                    ordered.append(value)

        if include_recommended:
            _extend_unique(_COMETAPI_RECOMMENDED_MODELS)

        _extend_unique(models)
        return ordered