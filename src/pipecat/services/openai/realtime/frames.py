#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Custom frame types for OpenAI Realtime API integration.

.. deprecated:: 0.0.92
    OpenAI Realtime no longer uses types from this module under the hood.

    It now works more like most LLM services in Pipecat, relying on updates to
    its context, pushed by context aggregators, to update its internal state.

    Listen for ``LLMContextFrame`` s for context updates.
"""

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "Types in pipecat.services.openai.realtime.frames are deprecated. \n"
        "OpenAI Realtime no longer uses types from this module under the hood. \n\n"
        "It now works more like other LLM services in Pipecat, relying on updates to \n"
        "its context, pushed by context aggregators, to update its internal state.\n\n"
        "Listen for `LLMContextFrame`s for context updates.\n"
    )

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
