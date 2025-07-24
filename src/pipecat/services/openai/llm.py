#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI LLM service implementation with context aggregators."""

import json
from dataclasses import dataclass
from typing import Any, Optional

from pipecat.frames.frames import (
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    UserImageRawFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantContextAggregator_Universal,
    LLMAssistantContextAggregatorParams,
    LLMUserContextAggregator_Universal,
    LLMUserContextAggregatorParams,
)
from pipecat.services.openai.base_llm import BaseOpenAILLMService


class OpenAILLMService(BaseOpenAILLMService):
    """OpenAI LLM service implementation.

    Provides a complete OpenAI LLM service with context aggregation support.
    Uses the BaseOpenAILLMService for core functionality and adds OpenAI-specific
    context aggregator creation.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4.1",
        params: Optional[BaseOpenAILLMService.InputParams] = None,
        **kwargs,
    ):
        """Initialize OpenAI LLM service.

        Args:
            model: The OpenAI model name to use. Defaults to "gpt-4.1".
            params: Input parameters for model configuration.
            **kwargs: Additional arguments passed to the parent BaseOpenAILLMService.
        """
        super().__init__(model=model, params=params, **kwargs)
