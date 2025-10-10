#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Custom frame types for OpenAI Realtime API integration."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pipecat.frames.frames import DataFrame, FunctionCallResultFrame

if TYPE_CHECKING:
    from pipecat.services.openai.realtime.context import OpenAIRealtimeLLMContext


@dataclass
class RealtimeMessagesUpdateFrame(DataFrame):
    """Frame indicating that the realtime context messages have been updated.

    Parameters:
        context: The updated OpenAI realtime LLM context.
    """

    context: "OpenAIRealtimeLLMContext"


@dataclass
class RealtimeFunctionCallResultFrame(DataFrame):
    """Frame containing function call results for the realtime service.

    Parameters:
        result_frame: The function call result frame to send to the realtime API.
    """

    result_frame: FunctionCallResultFrame
