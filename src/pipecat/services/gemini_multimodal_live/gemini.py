#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Google Gemini Live API service implementation.

This module provides real-time conversational AI capabilities using Google's
Gemini Live API, supporting both text and audio modalities with
voice transcription, streaming responses, and tool usage.

.. deprecated:: 0.0.90
    This module is deprecated. Please use the equivalent types from
    pipecat.services.google.gemini_live.llm instead. Note that the new type names
    do not include 'Multimodal'.
"""

import warnings

from pipecat.services.google.gemini_live.llm import (
    ContextWindowCompressionParams as _ContextWindowCompressionParams,
)
from pipecat.services.google.gemini_live.llm import (
    GeminiLiveAssistantContextAggregator,
    GeminiLiveContext,
    GeminiLiveContextAggregatorPair,
    GeminiLiveLLMService,
    GeminiLiveUserContextAggregator,
    GeminiModalities,
)
from pipecat.services.google.gemini_live.llm import GeminiMediaResolution as _GeminiMediaResolution
from pipecat.services.google.gemini_live.llm import GeminiVADParams as _GeminiVADParams
from pipecat.services.google.gemini_live.llm import InputParams as _InputParams

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "Types in pipecat.services.gemini_multimodal_live.gemini are deprecated. "
        "Please use the equivalent types from "
        "pipecat.services.google.gemini_live.llm instead. Note that the new type "
        "names do not include 'Multimodal' "
        "(e.g. `GeminiMultimodalLiveLLMService` is now `GeminiLiveLLMService`).",
        DeprecationWarning,
        stacklevel=2,
    )

GeminiMultimodalLiveContext = GeminiLiveContext
GeminiMultimodalLiveUserContextAggregator = GeminiLiveUserContextAggregator
GeminiMultimodalLiveAssistantContextAggregator = GeminiLiveAssistantContextAggregator
GeminiMultimodalLiveContextAggregatorPair = GeminiLiveContextAggregatorPair
GeminiMultimodalModalities = GeminiModalities
GeminiMediaResolution = _GeminiMediaResolution
GeminiVADParams = _GeminiVADParams
ContextWindowCompressionParams = _ContextWindowCompressionParams
InputParams = _InputParams
GeminiMultimodalLiveLLMService = GeminiLiveLLMService
