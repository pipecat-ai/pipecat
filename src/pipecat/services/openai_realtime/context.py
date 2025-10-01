#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI Realtime LLM context and aggregator implementations."""

import copy
import json

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    FunctionCallResultFrame,
    InterimTranscriptionFrame,
    LLMMessagesUpdateFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
    TranscriptionFrame,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.llm import (
    OpenAIAssistantContextAggregator,
    OpenAIUserContextAggregator,
)

from . import events
from .frames import RealtimeFunctionCallResultFrame, RealtimeMessagesUpdateFrame


class OpenAIRealtimeLLMContext(OpenAILLMContext):
    """OpenAI Realtime LLM context with session management and message conversion.

    Extends the standard OpenAI LLM context to support real-time session properties,
    instruction management, and conversion between standard message formats and
    realtime conversation items.
    """

    @staticmethod
    def upgrade_to_realtime(obj: OpenAILLMContext) -> "OpenAIRealtimeLLMContext":
        """Upgrade a standard OpenAI LLM context to a realtime context.

        Args:
            obj: The OpenAILLMContext instance to upgrade.

        Returns:
            The upgraded OpenAIRealtimeLLMContext instance.
        """
        if isinstance(obj, OpenAILLMContext) and not isinstance(obj, OpenAIRealtimeLLMContext):
            obj.__class__ = OpenAIRealtimeLLMContext
            obj.__setup_local()
        return obj

    def add_user_content_item_as_message(self, item):
        """Add a user content item as a standard message to the context.

        Args:
            item: The conversation item to add as a user message.
        """
        message = {
            "role": "user",
            "content": [{"type": "text", "text": item.content[0].transcript}],
        }
        self.add_message(message)


class OpenAIRealtimeUserContextAggregator(OpenAIUserContextAggregator):
    """User context aggregator for OpenAI Realtime API.

    Handles user input frames and generates appropriate context updates
    for the realtime conversation, including message updates and tool settings.

    Args:
        context: The OpenAI realtime LLM context.
        **kwargs: Additional arguments passed to parent aggregator.
    """

    async def process_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ):
        """Process incoming frames and handle realtime-specific frame types.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)
        # Parent does not push LLMMessagesUpdateFrame. This ensures that in a typical pipeline,
        # messages are only processed by the user context aggregator, which is generally what we want. But
        # we also need to send new messages over the websocket, so the openai realtime API has them
        # in its context.
        if isinstance(frame, LLMMessagesUpdateFrame):
            await self.push_frame(RealtimeMessagesUpdateFrame(context=self._context))

        # Parent also doesn't push the LLMSetToolsFrame.
        if isinstance(frame, LLMSetToolsFrame):
            await self.push_frame(frame, direction)

    async def push_aggregation(self):
        """Push user input aggregation.

        Currently ignores all user input coming into the pipeline as realtime
        audio input is handled directly by the service.
        """
        # for the moment, ignore all user input coming into the pipeline.
        # todo: think about whether/how to fix this to allow for text input from
        #       upstream (transport/transcription, or other sources)
        pass


class OpenAIRealtimeAssistantContextAggregator(OpenAIAssistantContextAggregator):
    """Assistant context aggregator for OpenAI Realtime API.

    Handles assistant output frames from the realtime service, filtering
    out duplicate text frames and managing function call results.

    Args:
        context: The OpenAI realtime LLM context.
        **kwargs: Additional arguments passed to parent aggregator.
    """

    # The LLMAssistantContextAggregator uses TextFrames to aggregate the LLM output,
    # but the OpenAIRealtimeLLMService pushes LLMTextFrames and TTSTextFrames. We
    # need to override this proces_frame for LLMTextFrame, so that only the TTSTextFrames
    # are process. This ensures that the context gets only one set of messages.
    # OpenAIRealtimeLLMService also pushes TranscriptionFrames and InterimTranscriptionFrames,
    # so we need to ignore pushing those as well, as they're also TextFrames.
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process assistant frames, filtering out duplicate text content.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        if not isinstance(frame, (LLMTextFrame, TranscriptionFrame, InterimTranscriptionFrame)):
            await super().process_frame(frame, direction)

    async def handle_function_call_result(self, frame: FunctionCallResultFrame):
        """Handle function call result and notify the realtime service.

        Args:
            frame: The function call result frame to handle.
        """
        await super().handle_function_call_result(frame)

        # The standard function callback code path pushes the FunctionCallResultFrame from the llm itself,
        # so we didn't have a chance to add the result to the openai realtime api context. Let's push a
        # special frame to do that.
        await self.push_frame(
            RealtimeFunctionCallResultFrame(result_frame=frame), FrameDirection.UPSTREAM
        )
