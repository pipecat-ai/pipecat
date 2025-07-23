#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LLM response aggregators for handling conversation context and message aggregation.

This module provides aggregators that process and accumulate LLM responses, user inputs,
and conversation context. These aggregators handle the flow between speech-to-text,
LLM processing, and text-to-speech components in conversational AI pipelines.
"""

import asyncio
from dataclasses import dataclass
from typing import List, Literal, Optional

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.aggregators.llm_context import LLMContext, LLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


@dataclass
class LLMUserContextAggregatorParams:
    """Parameters for configuring LLM user context aggregation behavior.

    Parameters:
        aggregation_timeout: Maximum time in seconds to wait for additional
            transcription content before pushing aggregated result. This
            timeout is used only when the transcription is slow to arrive.
        turn_emulated_vad_timeout: Maximum time in seconds to wait for emulated
            VAD when using turn-based analysis. Applied when transcription is
            received but VAD didn't detect speech (e.g., whispered utterances).
    """

    aggregation_timeout: float = 0.5
    turn_emulated_vad_timeout: float = 0.8


@dataclass
class LLMAssistantContextAggregatorParams:
    """Parameters for configuring LLM assistant context aggregation behavior.

    Parameters:
        expect_stripped_words: Whether to expect and handle stripped words
            in text frames by adding spaces between tokens.
    """

    expect_stripped_words: bool = True


class LLMContextAggregator(FrameProcessor):
    """Base LLM aggregator that uses an LLMContext for conversation storage.

    This aggregator maintains conversation state using an LLMContext and
    pushes LLMContextFrame objects as aggregation frames. It provides
    common functionality for context-based conversation management.
    """

    def __init__(self, *, context: LLMContext, role: str, **kwargs):
        """Initialize the context response aggregator.

        Args:
            context: The LLM context to use for conversation storage.
            role: The role this aggregator represents (e.g. "user", "assistant").
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._context = context
        self._role = role

        self._aggregation: str = ""

    @property
    def messages(self) -> List[dict]:
        """Get messages from the LLM context.

        Returns:
            List of message dictionaries from the context.
        """
        return self._context.messages

    @property
    def role(self) -> str:
        """Get the role for this aggregator.

        Returns:
            The role string for this aggregator.
        """
        return self._role

    @property
    def context(self):
        """Get the LLM context.

        Returns:
            The LLMContext instance used by this aggregator.
        """
        return self._context

    def get_context_frame(self) -> LLMContextFrame:
        """Create a context frame with the current context.

        Returns:
            LLMContextFrame containing the current context.
        """
        return LLMContextFrame(context=self._context)

    async def push_context_frame(self, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a context frame in the specified direction.

        Args:
            direction: The direction to push the frame (upstream or downstream).
        """
        frame = self.get_context_frame()
        await self.push_frame(frame, direction)

    def add_messages(self, messages):
        """Add messages to the context.

        Args:
            messages: Messages to add to the conversation context.
        """
        self._context.add_messages(messages)

    def set_messages(self, messages):
        """Set the context messages.

        Args:
            messages: Messages to replace the current context messages.
        """
        self._context.set_messages(messages)

    def set_tools(self, tools: List):
        """Set tools in the context.

        Args:
            tools: List of tool definitions to set in the context.
        """
        self._context.set_tools(tools)

    # TODO: should we be using LLMContextToolChoice here?
    def set_tool_choice(self, tool_choice: Literal["none", "auto", "required"] | dict):
        """Set tool choice in the context.

        Args:
            tool_choice: Tool choice configuration for the context.
        """
        self._context.set_tool_choice(tool_choice)

    async def reset(self):
        """Reset the aggregation state."""
        self._aggregation = ""


class LLMUserContextAggregator(LLMContextAggregator):
    """User LLM aggregator that processes speech-to-text transcriptions.

    This aggregator handles the complex logic of aggregating user speech transcriptions
    from STT services. It manages multiple scenarios including:

    - Transcriptions received between VAD events
    - Transcriptions received outside VAD events
    - Interim vs final transcriptions
    - User interruptions during bot speech
    - Emulated VAD for whispered or short utterances

    The aggregator uses timeouts to handle cases where transcriptions arrive
    after VAD events or when no VAD is available.
    """

    def __init__(
        self,
        context: LLMContext,
        *,
        params: Optional[LLMUserContextAggregatorParams] = None,
        **kwargs,
    ):
        """Initialize the user context aggregator.

        Args:
            context: The LLM context for conversation storage.
            params: Configuration parameters for aggregation behavior.
            **kwargs: Additional arguments. Supports deprecated 'aggregation_timeout'.
        """
        super().__init__(context=context, role="user", **kwargs)
        self._params = params or LLMUserContextAggregatorParams()
        self._vad_params: Optional[VADParams] = None
        self._turn_params: Optional[SmartTurnParams] = None

        if "aggregation_timeout" in kwargs:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "Parameter 'aggregation_timeout' is deprecated, use 'params' instead.",
                    DeprecationWarning,
                )

            self._params.aggregation_timeout = kwargs["aggregation_timeout"]

        self._user_speaking = False
        self._bot_speaking = False
        self._was_bot_speaking = False
        self._emulating_vad = False
        self._seen_interim_results = False
        self._waiting_for_aggregation = False

        self._aggregation_event = asyncio.Event()
        self._aggregation_task = None

    async def reset(self):
        """Reset the aggregation state and interruption strategies."""
        await super().reset()
        self._was_bot_speaking = False
        self._seen_interim_results = False
        self._waiting_for_aggregation = False
        [await s.reset() for s in self._interruption_strategies]

    async def handle_aggregation(self, aggregation: str):
        """Add the aggregated user text to the context.

        Args:
            aggregation: The aggregated user text to add as a user message.
        """
        self._context.add_message({"role": self.role, "content": aggregation})

    # TODO: continue porting things over from LLMUserContextAggregator in backup file
