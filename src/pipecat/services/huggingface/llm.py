#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Hugging Face LLM service implementation.

Uses Hugging Face's OpenAI-compatible Inference Providers chat-completions
endpoint for hosted, usage-based model inference.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService


@dataclass
class HuggingFaceLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for HuggingFaceLLMService."""

    pass


class HuggingFaceLLMService(OpenAILLMService):
    """OpenAI-compatible LLM service for Hugging Face Inference Providers."""

    Settings = HuggingFaceLLMSettings
    _settings: Settings
    supports_developer_role = False

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://router.huggingface.co/v1",
        bill_to: str | None = None,
        model: str | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Hugging Face LLM service.

        Args:
            api_key: Hugging Face user access token with Inference Providers permission.
            base_url: OpenAI-compatible Hugging Face router base URL.
            bill_to: Optional Hugging Face organization/user to bill via X-HF-Bill-To.
            model: Model identifier to use.

                .. deprecated:: 0.0.105
                    Use ``settings=HuggingFaceLLMService.Settings(model=...)`` instead.

            settings: Runtime-updatable settings.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        default_settings = self.Settings(model="openai/gpt-oss-120b:cerebras")

        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        if settings is not None:
            default_settings.apply_update(settings)

        default_headers: Mapping[str, str] | None = None
        if bill_to:
            default_headers = {"X-HF-Bill-To": bill_to}

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
            settings=default_settings,
            **kwargs,
        )
