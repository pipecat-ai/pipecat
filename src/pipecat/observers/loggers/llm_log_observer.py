#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from loguru import logger

from pipecat.frames.frames import (
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMTextFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService


class LLMLogObserver(BaseObserver):
    """Observer to log LLM activity to the console.

    Logs all frame instances (only from/to LLM service) of:

    - LLMFullResponseStartFrame
    - LLMFullResponseEndFrame
    - LLMTextFrame
    - FunctionCallInProgressFrame
    - LLMMessagesFrame
    - OpenAILLMContextFrame

    This allows you to track when the LLM starts responding, what it generates,
    and when it finishes.

    """

    async def on_push_frame(self, data: FramePushed):
        src = data.source
        dst = data.destination
        frame = data.frame
        direction = data.direction
        timestamp = data.timestamp

        if not isinstance(src, LLMService) and not isinstance(dst, LLMService):
            return

        time_sec = timestamp / 1_000_000_000

        arrow = "→"

        # Log LLM start/end frames (output)
        if isinstance(frame, (LLMFullResponseStartFrame, LLMFullResponseEndFrame)):
            event = "START" if isinstance(frame, LLMFullResponseStartFrame) else "END"
            logger.debug(f"🧠 {src} {arrow} LLM {event} RESPONSE at {time_sec:.2f}s")
        # Log all LLMTextFrames (output)
        elif isinstance(frame, LLMTextFrame):
            logger.debug(f"🧠 {src} {arrow} LLM GENERATING: {frame.text!r} at {time_sec:.2f}s")
        # Log function calling (output)
        elif (
            isinstance(frame, FunctionCallInProgressFrame)
            and direction != FrameDirection.DOWNSTREAM
        ):
            logger.debug(
                f"🧠 {src} {arrow} LLM FUNCTION CALL ({frame.tool_call_id}): {frame.function_name!r}({frame.arguments}) at {time_sec:.2f}s"
            )
        # Log LLMMessagesFrame (input)
        elif isinstance(frame, LLMMessagesFrame):
            logger.debug(
                f"🧠 {arrow} {dst} LLM MESSAGES FRAME: {frame.messages} at {time_sec:.2f}s"
            )
        # Log OpenAILLMContextFrame (input)
        elif isinstance(frame, OpenAILLMContextFrame):
            logger.debug(
                f"🧠 {arrow} {dst} LLM CONTEXT FRAME: {frame.context.messages} at {time_sec:.2f}s"
            )
        # Log function call result (input)
        elif isinstance(frame, FunctionCallResultFrame):
            logger.debug(
                f"🧠 {arrow} {src} LLM FUNCTION CALL RESULT ({frame.tool_call_id}): {frame.result} at {time_sec:.2f}s"
            )
