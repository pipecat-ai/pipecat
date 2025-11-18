#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import warnings

from pipecat.services.azure.realtime.llm import AzureRealtimeLLMService
from pipecat.services.openai.realtime.events import (
    InputAudioNoiseReduction,
    InputAudioTranscription,
    SemanticTurnDetection,
    SessionProperties,
    TurnDetection,
)
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "Types in pipecat.services.openai_realtime are deprecated. "
        "Please use the equivalent types from "
        "pipecat.services.openai.realtime instead.",
        DeprecationWarning,
        stacklevel=2,
    )
