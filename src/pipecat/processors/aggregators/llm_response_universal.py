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
import json
import warnings
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set, Type

from loguru import logger

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    AssistantImageRawFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextAssistantTimestampFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
    LLMRunFrame,
    LLMSetToolChoiceFrame,
    LLMSetToolsFrame,
    LLMThoughtEndFrame,
    LLMThoughtStartFrame,
    LLMThoughtTextFrame,
    SpeechControlParamsFrame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
    UserImageRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
    LLMContextMessage,
    LLMSpecificMessage,
    NotGiven,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.mute import BaseUserMuteStrategy
from pipecat.turns.user_start import BaseUserTurnStartStrategy, UserTurnStartedParams
from pipecat.turns.user_stop import BaseUserTurnStopStrategy, UserTurnStoppedParams
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies, UserTurnStrategies
from pipecat.utils.string import TextPartForConcatenation, concatenate_aggregated_text
from pipecat.utils.time import time_now_iso8601


@dataclass
class LLMUserAggregatorParams:
    """Parameters for configuring LLM user aggregation behavior.

    Parameters:
        user_turn_strategies: User turn start and stop strategies.
        user_mute_strategies: List of user mute strategies.
        user_turn_stop_timeout: Time in seconds to wait before considering the
            user's turn finished.
    """

    user_turn_strategies: Optional[UserTurnStrategies] = None
    user_mute_strategies: List[BaseUserMuteStrategy] = field(default_factory=list)
    user_turn_stop_timeout: float = 5.0


@dataclass
class LLMAssistantAggregatorParams:
    """Parameters for configuring LLM assistant aggregation behavior.

    Parameters:
        expect_stripped_words: Whether to expect and handle stripped words
            in text frames by adding spaces between tokens. This parameter is
            ignored when used with the newer LLMAssistantAggregator, which
            handles word spacing automatically.
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

        self._aggregation: List[TextPartForConcatenation] = []

    @property
    def messages(self) -> List[LLMContextMessage]:
        """Get messages from the LLM context.

        Returns:
            List of message dictionaries from the context.
        """
        return self._context.get_messages()

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

    def _get_context_frame(self) -> LLMContextFrame:
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
        frame = self._get_context_frame()
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

    def set_tools(self, tools: ToolsSchema | NotGiven):
        """Set tools in the context.

        Args:
            tools: List of tool definitions to set in the context.
        """
        self._context.set_tools(tools)

    def set_tool_choice(self, tool_choice: Literal["none", "auto", "required"] | dict):
        """Set tool choice in the context.

        Args:
            tool_choice: Tool choice configuration for the context.
        """
        self._context.set_tool_choice(tool_choice)

    async def reset(self):
        """Reset the aggregation state."""
        self._aggregation = []

    @abstractmethod
    async def push_aggregation(self):
        """Push the current aggregation downstream."""
        pass

    def aggregation_string(self) -> str:
        """Get the current aggregation as a string.

        Returns:
            The concatenated aggregation string.
        """
        return concatenate_aggregated_text(self._aggregation)


class LLMUserAggregator(LLMContextAggregator):
    """User LLM aggregator that aggregates user input during active user turns.

    This aggregator operates within turn boundaries defined by the configured
    user and bot turn start strategies. User turn start strategies indicate when
    a user turn begins, while bot turn start strategies signal when the user
    turn has ended and control transitions to the bot turn.

    The aggregator collects and aggregates speech-to-text transcriptions that
    occur while a user turn is active and pushes the final aggregation when the
    user turn is finished.

    Event handlers available:

    - on_user_turn_started: Called when the user turn starts
    - on_user_turn_stopped: Called when the user turn ends
    - on_user_turn_stop_timeout: Called when no user turn stop strategy triggers

    Example::

        @aggregator.event_handler("on_user_turn_started")
        async def on_user_turn_started(aggregator, Optional[strategy]):
            ...

        @aggregator.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(aggregator, Optional[strategy]):
            ...

        @aggregator.event_handler("on_user_turn_stop_timeout")
        async def on_user_turn_stop_timeout(aggregator):
            ...

    """

    def __init__(
        self,
        context: LLMContext,
        *,
        params: Optional[LLMUserAggregatorParams] = None,
        **kwargs,
    ):
        """Initialize the user context aggregator.

        Args:
            context: The LLM context for conversation storage.
            params: Configuration parameters for aggregation behavior.
            **kwargs: Additional arguments.
        """
        super().__init__(context=context, role="user", **kwargs)
        self._params = params or LLMUserAggregatorParams()

        # Initialize default user turn strategies.
        self._user_turn_strategies = self._params.user_turn_strategies or UserTurnStrategies()

        self._vad_user_speaking = False

        self._user_turn = False
        self._user_is_muted = False
        self._user_turn_stop_timeout_event = asyncio.Event()
        self._user_turn_stop_timeout_task: Optional[asyncio.Task] = None

        self._register_event_handler("on_user_turn_started")
        self._register_event_handler("on_user_turn_stopped")
        self._register_event_handler("on_user_turn_stop_timeout")

    async def cleanup(self):
        """Clean up processor resources."""
        await super().cleanup()
        await self._cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames for user speech aggregation and context management.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if await self._maybe_mute_frame(frame):
            return

        if isinstance(frame, StartFrame):
            # Push StartFrame before start(), because we want StartFrame to be
            # processed by every processor before any other frame is processed.
            await self.push_frame(frame, direction)
            await self._start(frame)
        elif isinstance(frame, EndFrame):
            # Push EndFrame before stop(), because stop() waits on the task to
            # finish and the task finishes when EndFrame is processed.
            await self.push_frame(frame, direction)
            await self._stop(frame)
        elif isinstance(frame, CancelFrame):
            await self._cancel(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            await self._handle_vad_user_started_speaking(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._handle_vad_user_stopped_speaking(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, TranscriptionFrame):
            await self._handle_transcription(frame)
        elif isinstance(frame, LLMRunFrame):
            await self._handle_llm_run(frame)
        elif isinstance(frame, LLMMessagesAppendFrame):
            await self._handle_llm_messages_append(frame)
        elif isinstance(frame, LLMMessagesUpdateFrame):
            await self._handle_llm_messages_update(frame)
        elif isinstance(frame, LLMSetToolsFrame):
            self.set_tools(frame.tools)
            # Push the LLMSetToolsFrame as well, since speech-to-speech LLM
            # services (like OpenAI Realtime) may need to know about tool
            # changes; unlike text-based LLM services they won't just "pick up
            # the change" on the next LLM run, as the LLM is continuously
            # running.
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMSetToolChoiceFrame):
            self.set_tool_choice(frame.tool_choice)
        elif isinstance(frame, SpeechControlParamsFrame):
            await self._handle_speech_control_params(frame)
        else:
            await self.push_frame(frame, direction)

        await self._user_turn_strategies_process_frame(frame)

    async def push_aggregation(self):
        """Push the current aggregation."""
        if len(self._aggregation) == 0:
            return

        aggregation = self.aggregation_string()
        await self.reset()
        self._context.add_message({"role": self.role, "content": aggregation})
        await self.push_context_frame()

    async def _start(self, frame: StartFrame):
        if not self._user_turn_stop_timeout_task:
            self._user_turn_stop_timeout_task = self.create_task(
                self._user_turn_stop_timeout_task_handler()
            )

        await self._setup_user_turn_strategies()
        await self._setup_user_mute_strategies()

    async def _setup_user_mute_strategies(self):
        for s in self._params.user_mute_strategies:
            await s.setup(self.task_manager)

    async def _setup_user_turn_strategies(self):
        if self._user_turn_strategies.start:
            for s in self._user_turn_strategies.start:
                await s.setup(self.task_manager)
                s.add_event_handler("on_push_frame", self._on_push_frame)
                s.add_event_handler("on_broadcast_frame", self._on_broadcast_frame)
                s.add_event_handler("on_user_turn_started", self._on_user_turn_started)

        if self._user_turn_strategies.stop:
            for s in self._user_turn_strategies.stop:
                await s.setup(self.task_manager)
                s.add_event_handler("on_push_frame", self._on_push_frame)
                s.add_event_handler("on_broadcast_frame", self._on_broadcast_frame)
                s.add_event_handler("on_user_turn_stopped", self._on_user_turn_stopped)

    async def _stop(self, frame: EndFrame):
        await self._cleanup()

    async def _cancel(self, frame: CancelFrame):
        await self._cleanup()

    async def _cleanup(self):
        if self._user_turn_stop_timeout_task:
            await self.cancel_task(self._user_turn_stop_timeout_task)
            self._user_turn_stop_timeout_task = None

        await self._cleanup_user_turn_strategies()
        await self._cleanup_user_mute_strategies()

    async def _cleanup_user_turn_strategies(self):
        if self._user_turn_strategies.start:
            for s in self._user_turn_strategies.start:
                await s.cleanup()

        if self._user_turn_strategies.stop:
            for s in self._user_turn_strategies.stop:
                await s.cleanup()

    async def _cleanup_user_mute_strategies(self):
        for s in self._params.user_mute_strategies:
            await s.cleanup()

    async def _maybe_mute_frame(self, frame: Frame):
        should_mute_frame = self._user_is_muted and isinstance(
            frame,
            (
                InterruptionFrame,
                VADUserStartedSpeakingFrame,
                VADUserStoppedSpeakingFrame,
                UserStartedSpeakingFrame,
                UserStoppedSpeakingFrame,
                InputAudioRawFrame,
                InterimTranscriptionFrame,
                TranscriptionFrame,
            ),
        )

        if should_mute_frame:
            logger.trace(f"{frame.name} suppressed - user currently muted")

        should_mute_next_time = False
        for s in self._params.user_mute_strategies:
            should_mute_next_time |= await s.process_frame(frame)

        if should_mute_next_time != self._user_is_muted:
            logger.debug(f"{self}: user is now {'muted' if should_mute_next_time else 'unmuted'}")
            self._user_is_muted = should_mute_next_time

        return should_mute_frame

    async def _user_turn_strategies_process_frame(self, frame: Frame):
        if self._user_turn_strategies.start:
            for strategy in self._user_turn_strategies.start:
                await strategy.process_frame(frame)

        if self._user_turn_strategies.stop:
            for strategy in self._user_turn_strategies.stop:
                await strategy.process_frame(frame)

    async def _handle_llm_run(self, frame: LLMRunFrame):
        await self.push_context_frame()

    async def _handle_llm_messages_append(self, frame: LLMMessagesAppendFrame):
        self.add_messages(frame.messages)
        if frame.run_llm:
            await self.push_context_frame()

    async def _handle_llm_messages_update(self, frame: LLMMessagesUpdateFrame):
        self.set_messages(frame.messages)
        if frame.run_llm:
            await self.push_context_frame()

    async def _handle_speech_control_params(self, frame: SpeechControlParamsFrame):
        if not frame.turn_params:
            return

        logger.warning(
            f"{self}: `turn_analyzer` in base input transport is deprecated. "
            "Use `LLMUserAggregator`'s new `user_turn_strategies` parameter with "
            "`TurnAnalyzerUserTurnStopStrategy` instead:\n"
            "\n"
            "    context_aggregator = LLMContextAggregatorPair(\n"
            "        context,\n"
            "        user_params=LLMUserAggregatorParams(\n"
            "            ...,\n"
            "            user_turn_strategies=UserTurnStrategies(\n"
            "                stop=[\n"
            "                    TurnAnalyzerUserTurnStopStrategy(\n"
            "                        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams())\n"
            "                    )\n"
            "                ],\n"
            "            )\n"
            "        ),\n"
            "    )"
        )

        await self._cleanup_user_turn_strategies()
        self._user_turn_strategies = ExternalUserTurnStrategies()
        await self._setup_user_turn_strategies()

    async def _handle_vad_user_started_speaking(self, frame: VADUserStartedSpeakingFrame):
        self._vad_user_speaking = True

        # The user started talking, let's reset the user turn timeout.
        self._user_turn_stop_timeout_event.set()

    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        self._vad_user_speaking = False

        # The user stopped talking, let's reset the user turn timeout.
        self._user_turn_stop_timeout_event.set()

    async def _handle_transcription(self, frame: TranscriptionFrame):
        text = frame.text

        # Make sure we really have some text.
        if not text.strip():
            return

        # We have creceived a transcription, let's reset the user turn timeout.
        self._user_turn_stop_timeout_event.set()

        # Transcriptions never include inter-part spaces (so far).
        self._aggregation.append(
            TextPartForConcatenation(
                text, includes_inter_part_spaces=frame.includes_inter_frame_spaces
            )
        )

    async def _on_user_turn_started(
        self,
        strategy: BaseUserTurnStartStrategy,
        params: UserTurnStartedParams,
    ):
        await self._trigger_user_turn_start(strategy, params)

    async def _on_user_turn_stopped(
        self, strategy: BaseUserTurnStopStrategy, params: UserTurnStoppedParams
    ):
        await self._trigger_user_turn_stop(strategy, params)

    async def _on_push_frame(
        self,
        strategy: BaseUserTurnStartStrategy | BaseUserTurnStopStrategy,
        frame: Frame,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ):
        await self.push_frame(frame, direction)

    async def _on_broadcast_frame(
        self,
        strategy: BaseUserTurnStartStrategy | BaseUserTurnStopStrategy,
        frame_cls: Type[Frame],
        **kwargs,
    ):
        await self.broadcast_frame(frame_cls, **kwargs)

    async def _trigger_user_turn_start(
        self, strategy: Optional[BaseUserTurnStartStrategy], params: UserTurnStartedParams
    ):
        # Prevent two consecutive user turn starts.
        if self._user_turn:
            return

        logger.debug(f"User started speaking (user turn start strategy: {strategy})")

        self._user_turn = True
        self._user_turn_stop_timeout_event.set()

        # Reset all user turn start strategies to start fresh.
        if self._user_turn_strategies.start:
            for s in self._user_turn_strategies.start:
                await s.reset()

        if params.enable_user_speaking_frames:
            # TODO(aleix): This frame should really come from the top of the pipeline.
            await self.broadcast_frame(UserStartedSpeakingFrame)

        if params.enable_interruptions and self._allow_interruptions:
            # TODO(aleix): This frame should really come from the top of the pipeline.
            await self.broadcast_frame(InterruptionFrame)

        await self._call_event_handler("on_user_turn_started", strategy)

    async def _trigger_user_turn_stop(
        self, strategy: Optional[BaseUserTurnStopStrategy], params: UserTurnStoppedParams
    ):
        # Prevent two consecutive user turn stops.
        if not self._user_turn:
            return

        logger.debug(f"User stopped speaking (user turn stop strategy: {strategy})")

        self._user_turn = False
        self._user_turn_stop_timeout_event.set()

        # Reset all user turn stop strategies to start fresh.
        if self._user_turn_strategies.stop:
            for s in self._user_turn_strategies.stop:
                await s.reset()

        if params.enable_user_speaking_frames:
            # TODO(aleix): This frame should really come from the top of the pipeline.
            await self.broadcast_frame(UserStoppedSpeakingFrame)

        await self._call_event_handler("on_user_turn_stopped", strategy)

        # Always push context frame.
        await self.push_aggregation()

    async def _user_turn_stop_timeout_task_handler(self):
        while True:
            try:
                await asyncio.wait_for(
                    self._user_turn_stop_timeout_event.wait(),
                    timeout=self._params.user_turn_stop_timeout,
                )
                self._user_turn_stop_timeout_event.clear()
            except asyncio.TimeoutError:
                if self._user_turn and not self._vad_user_speaking:
                    await self._call_event_handler("on_user_turn_stop_timeout")
                    await self._trigger_user_turn_stop(
                        None, UserTurnStoppedParams(enable_user_speaking_frames=True)
                    )


class LLMAssistantAggregator(LLMContextAggregator):
    """Assistant LLM aggregator that processes bot responses and function calls.

    This aggregator handles the complex logic of processing assistant responses including:

    - Text frame aggregation between response start/end markers
    - Function call lifecycle management
    - Context updates with timestamps
    - Tool execution and result handling
    - Interruption handling during responses

    The aggregator manages function calls in progress and coordinates between
    text generation and tool execution phases of LLM responses.
    """

    def __init__(
        self,
        context: LLMContext,
        *,
        params: Optional[LLMAssistantAggregatorParams] = None,
        **kwargs,
    ):
        """Initialize the assistant context aggregator.

        Args:
            context: The OpenAI LLM context for conversation storage.
            params: Configuration parameters for aggregation behavior.
            **kwargs: Additional arguments.
        """
        super().__init__(context=context, role="assistant", **kwargs)
        self._params = params or LLMAssistantAggregatorParams()

        if "expect_stripped_words" in kwargs:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "Parameter 'expect_stripped_words' is deprecated. "
                    "LLMAssistantAggregator now handles word spacing automatically.",
                    DeprecationWarning,
                )

            self._params.expect_stripped_words = kwargs["expect_stripped_words"]

        if params and not params.expect_stripped_words:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "params.expect_stripped_words is deprecated. "
                    "LLMAssistantAggregator now handles word spacing automatically.",
                    DeprecationWarning,
                )

        self._started = 0
        self._function_calls_in_progress: Dict[str, Optional[FunctionCallInProgressFrame]] = {}
        self._context_updated_tasks: Set[asyncio.Task] = set()

        self._thought_aggregation_enabled = False
        self._thought_llm: str = ""
        self._thought_aggregation: List[TextPartForConcatenation] = []

    @property
    def has_function_calls_in_progress(self) -> bool:
        """Check if there are any function calls currently in progress.

        Returns:
            True if function calls are in progress, False otherwise.
        """
        return bool(self._function_calls_in_progress)

    async def reset(self):
        """Reset the aggregation state."""
        await super().reset()
        await self._reset_thought_aggregation()  # Just to be safe

    async def _reset_thought_aggregation(self):
        """Reset the thought aggregation state."""
        self._thought_aggregation_enabled = False
        self._thought_llm = ""
        self._thought_aggregation = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames for assistant response aggregation and function call management.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            await self._handle_interruptions(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMFullResponseStartFrame):
            await self._handle_llm_start(frame)
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._handle_llm_end(frame)
        elif isinstance(frame, TextFrame):
            await self._handle_text(frame)
        elif isinstance(frame, LLMThoughtStartFrame):
            await self._handle_thought_start(frame)
        elif isinstance(frame, LLMThoughtTextFrame):
            await self._handle_thought_text(frame)
        elif isinstance(frame, LLMThoughtEndFrame):
            await self._handle_thought_end(frame)
        elif isinstance(frame, LLMRunFrame):
            await self._handle_llm_run(frame)
        elif isinstance(frame, LLMMessagesAppendFrame):
            await self._handle_llm_messages_append(frame)
        elif isinstance(frame, LLMMessagesUpdateFrame):
            await self._handle_llm_messages_update(frame)
        elif isinstance(frame, LLMSetToolsFrame):
            self.set_tools(frame.tools)
        elif isinstance(frame, LLMSetToolChoiceFrame):
            self.set_tool_choice(frame.tool_choice)
        elif isinstance(frame, FunctionCallsStartedFrame):
            await self._handle_function_calls_started(frame)
        elif isinstance(frame, FunctionCallInProgressFrame):
            await self._handle_function_call_in_progress(frame)
        elif isinstance(frame, FunctionCallResultFrame):
            await self._handle_function_call_result(frame)
        elif isinstance(frame, FunctionCallCancelFrame):
            await self._handle_function_call_cancel(frame)
        elif isinstance(frame, UserImageRawFrame):
            await self._handle_user_image_frame(frame)
        elif isinstance(frame, AssistantImageRawFrame):
            await self._handle_assistant_image_frame(frame)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self.push_aggregation()
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def push_aggregation(self):
        """Push the current assistant aggregation with timestamp."""
        if not self._aggregation:
            return

        aggregation = self.aggregation_string()
        await self.reset()

        if aggregation:
            self._context.add_message({"role": "assistant", "content": aggregation})

        # Push context frame
        await self.push_context_frame()

        # Push timestamp frame with current time
        timestamp_frame = LLMContextAssistantTimestampFrame(timestamp=time_now_iso8601())
        await self.push_frame(timestamp_frame)

    async def _handle_llm_run(self, frame: LLMRunFrame):
        await self.push_context_frame(FrameDirection.UPSTREAM)

    async def _handle_llm_messages_append(self, frame: LLMMessagesAppendFrame):
        self.add_messages(frame.messages)
        if frame.run_llm:
            await self.push_context_frame(FrameDirection.UPSTREAM)

    async def _handle_llm_messages_update(self, frame: LLMMessagesUpdateFrame):
        self.set_messages(frame.messages)
        if frame.run_llm:
            await self.push_context_frame(FrameDirection.UPSTREAM)

    async def _handle_interruptions(self, frame: InterruptionFrame):
        await self.push_aggregation()
        self._started = 0
        await self.reset()

    async def _handle_function_calls_started(self, frame: FunctionCallsStartedFrame):
        function_names = [f"{f.function_name}:{f.tool_call_id}" for f in frame.function_calls]
        logger.debug(f"{self} FunctionCallsStartedFrame: {function_names}")
        for function_call in frame.function_calls:
            self._function_calls_in_progress[function_call.tool_call_id] = None

    async def _handle_function_call_in_progress(self, frame: FunctionCallInProgressFrame):
        logger.debug(
            f"{self} FunctionCallInProgressFrame: [{frame.function_name}:{frame.tool_call_id}]"
        )

        # Update context with the in-progress function call
        self._context.add_message(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": frame.tool_call_id,
                        "function": {
                            "name": frame.function_name,
                            "arguments": json.dumps(frame.arguments),
                        },
                        "type": "function",
                    }
                ],
            }
        )
        self._context.add_message(
            {
                "role": "tool",
                "content": "IN_PROGRESS",
                "tool_call_id": frame.tool_call_id,
            }
        )

        self._function_calls_in_progress[frame.tool_call_id] = frame

    async def _handle_function_call_result(self, frame: FunctionCallResultFrame):
        logger.debug(
            f"{self} FunctionCallResultFrame: [{frame.function_name}:{frame.tool_call_id}]"
        )
        if frame.tool_call_id not in self._function_calls_in_progress:
            logger.warning(
                f"FunctionCallResultFrame tool_call_id [{frame.tool_call_id}] is not running"
            )
            return

        del self._function_calls_in_progress[frame.tool_call_id]

        properties = frame.properties

        # Update context with the function call result
        if frame.result:
            result = json.dumps(frame.result)
            self._update_function_call_result(frame.function_name, frame.tool_call_id, result)
        else:
            self._update_function_call_result(frame.function_name, frame.tool_call_id, "COMPLETED")

        run_llm = False

        # Run inference if the function call result requires it.
        if frame.result:
            if properties and properties.run_llm is not None:
                # If the tool call result has a run_llm property, use it.
                run_llm = properties.run_llm
            elif frame.run_llm is not None:
                # If the frame is indicating we should run the LLM, do it.
                run_llm = frame.run_llm
            else:
                # If this is the last function call in progress, run the LLM.
                run_llm = not bool(self._function_calls_in_progress)

        if run_llm:
            await self.push_context_frame(FrameDirection.UPSTREAM)

        # Call the `on_context_updated` callback once the function call result
        # is added to the context. Also, run this in a separate task to make
        # sure we don't block the pipeline.
        if properties and properties.on_context_updated:
            task_name = f"{frame.function_name}:{frame.tool_call_id}:on_context_updated"
            task = self.create_task(properties.on_context_updated(), task_name)
            self._context_updated_tasks.add(task)
            task.add_done_callback(self._context_updated_task_finished)

    async def _handle_function_call_cancel(self, frame: FunctionCallCancelFrame):
        logger.debug(
            f"{self} FunctionCallCancelFrame: [{frame.function_name}:{frame.tool_call_id}]"
        )
        if frame.tool_call_id not in self._function_calls_in_progress:
            return

        if self._function_calls_in_progress[frame.tool_call_id].cancel_on_interruption:
            # Update context with the function call cancellation
            self._update_function_call_result(frame.function_name, frame.tool_call_id, "CANCELLED")
            del self._function_calls_in_progress[frame.tool_call_id]

    def _update_function_call_result(self, function_name: str, tool_call_id: str, result: Any):
        for message in self._context.get_messages():
            if (
                not isinstance(message, LLMSpecificMessage)
                and message["role"] == "tool"
                and message["tool_call_id"]
                and message["tool_call_id"] == tool_call_id
            ):
                message["content"] = result

    async def _handle_user_image_frame(self, frame: UserImageRawFrame):
        if not frame.append_to_context:
            return

        logger.debug(f"{self} Appending UserImageRawFrame to LLM context (size: {frame.size})")

        await self._context.add_image_frame_message(
            format=frame.format,
            size=frame.size,
            image=frame.image,
            text=frame.text,
        )

        await self.push_aggregation()
        await self.push_context_frame(FrameDirection.UPSTREAM)

    async def _handle_assistant_image_frame(self, frame: AssistantImageRawFrame):
        logger.debug(f"{self} Appending AssistantImageRawFrame to LLM context (size: {frame.size})")

        if frame.original_data and frame.original_mime_type:
            await self._context.add_image_frame_message(
                format=frame.original_mime_type,
                size=frame.size,  # Technically doesn't matter, since already encoded
                image=frame.original_data,
                role="assistant",
            )
        else:
            await self._context.add_image_frame_message(
                format=frame.format,
                size=frame.size,
                image=frame.image,
                role="assistant",
            )

    async def _handle_llm_start(self, _: LLMFullResponseStartFrame):
        self._started += 1

    async def _handle_llm_end(self, _: LLMFullResponseEndFrame):
        self._started -= 1
        await self.push_aggregation()

    async def _handle_text(self, frame: TextFrame):
        if not self._started or not frame.append_to_context:
            return

        # Make sure we really have text (spaces count, too!)
        if len(frame.text) == 0:
            return

        self._aggregation.append(
            TextPartForConcatenation(
                frame.text, includes_inter_part_spaces=frame.includes_inter_frame_spaces
            )
        )

    async def _handle_thought_start(self, frame: LLMThoughtStartFrame):
        if not self._started:
            return

        await self._reset_thought_aggregation()
        self._thought_aggregation_enabled = frame.append_to_context
        self._thought_llm = frame.llm

    async def _handle_thought_text(self, frame: LLMThoughtTextFrame):
        if not self._started or not self._thought_aggregation_enabled:
            return

        # Make sure we really have text (spaces count, too!)
        if len(frame.text) == 0:
            return

        self._thought_aggregation.append(
            TextPartForConcatenation(
                frame.text, includes_inter_part_spaces=frame.includes_inter_frame_spaces
            )
        )

    async def _handle_thought_end(self, frame: LLMThoughtEndFrame):
        if not self._started or not self._thought_aggregation_enabled:
            return

        thought = concatenate_aggregated_text(self._thought_aggregation)
        llm = self._thought_llm
        await self._reset_thought_aggregation()

        self._context.add_message(
            LLMSpecificMessage(
                llm=llm,
                message={
                    "type": "thought",
                    "text": thought,
                    "signature": frame.signature,
                },
            )
        )

    def _context_updated_task_finished(self, task: asyncio.Task):
        self._context_updated_tasks.discard(task)


class LLMContextAggregatorPair:
    """Pair of LLM context aggregators for updating context with user and assistant messages."""

    def __init__(
        self,
        context: LLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ):
        """Initialize the LLM context aggregator pair.

        Args:
            context: The context to be managed by the aggregators.
            user_params: Parameters for the user context aggregator.
            assistant_params: Parameters for the assistant context aggregator.
        """
        self._user = LLMUserAggregator(context, params=user_params)
        self._assistant = LLMAssistantAggregator(context, params=assistant_params)

    def user(self) -> LLMUserAggregator:
        """Get the user context aggregator.

        Returns:
            The user context aggregator instance.
        """
        return self._user

    def assistant(self) -> LLMAssistantAggregator:
        """Get the assistant context aggregator.

        Returns:
            The assistant context aggregator instance.
        """
        return self._assistant
