#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Google Gemini Live API service implementation.

This module provides real-time conversational AI capabilities using Google's
Gemini Live API, supporting both text and audio modalities with
voice transcription, streaming responses, and tool usage.
"""

import warnings

from pipecat.services.gemini_live.gemini import (
    ContextWindowCompressionParams as _ContextWindowCompressionParams,
)
from pipecat.services.gemini_live.gemini import (
    GeminiLiveAssistantContextAggregator,
    GeminiLiveContext,
    GeminiLiveContextAggregatorPair,
    GeminiLiveLLMService,
    GeminiLiveUserContextAggregator,
    GeminiModalities,
)
from pipecat.services.gemini_live.gemini import GeminiMediaResolution as _GeminiMediaResolution
from pipecat.services.gemini_live.gemini import GeminiVADParams as _GeminiVADParams
from pipecat.services.gemini_live.gemini import InputParams as _InputParams

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "Types in pipecat.services.gemini_multimodal_live.gemini are deprecated. "
        "Please import the equivalent types from "
        "pipecat.services.gemini_live.gemini instead. Note that the new type "
        "names do not include 'Multimodal'.",
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
