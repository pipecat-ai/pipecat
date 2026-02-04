#
# Copyright (c) 2024-2026, Daily
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
from pipecat.audio.vad.vad_analyzer import VADAnalyzer
from pipecat.audio.vad.vad_controller import VADController
from pipecat.frames.frames import (
    AssistantImageRawFrame,
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
    LLMUpdateSettingsFrame,
    SpeechControlParamsFrame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
    UserImageRawFrame,
    UserSpeakingFrame,
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
from pipecat.processors.frame_processor import FrameCallback, FrameDirection, FrameProcessor
from pipecat.turns.user_idle_controller import UserIdleController
from pipecat.turns.user_mute import BaseUserMuteStrategy
from pipecat.turns.user_start import BaseUserTurnStartStrategy, UserTurnStartedParams
from pipecat.turns.user_stop import BaseUserTurnStopStrategy, UserTurnStoppedParams
from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionConfig
from pipecat.turns.user_turn_controller import UserTurnController
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
        user_idle_timeout: Optional timeout in seconds for detecting user idle state.
            If set, the aggregator will emit an `on_user_turn_idle` event when the user
            has been idle (not speaking) for this duration. Set to None to disable
            idle detection.
        vad_analyzer: Voice Activity Detection analyzer instance.
        filter_incomplete_user_turns: Whether to filter out incomplete user turns.
            When enabled, the LLM outputs a turn completion marker at the start of
            each response: ✓ (complete), ○ (incomplete short), or ◐ (incomplete long).
            Incomplete responses are suppressed and timeouts trigger re-prompting.
        user_turn_completion_config: Configuration for turn completion behavior including
            custom instructions, timeouts, and prompts. Only used when
            filter_incomplete_user_turns is True.
    """

    user_turn_strategies: Optional[UserTurnStrategies] = None
    user_mute_strategies: List[BaseUserMuteStrategy] = field(default_factory=list)
    user_turn_stop_timeout: float = 5.0
    user_idle_timeout: Optional[float] = None
    vad_analyzer: Optional[VADAnalyzer] = None
    filter_incomplete_user_turns: bool = False
    user_turn_completion_config: Optional[UserTurnCompletionConfig] = None


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


@dataclass
class UserTurnStoppedMessage:
    """A user turn stopped message containing a user transcript update.

    A message in a conversation transcript containing the user content. This is
    the aggregated transcript that is then used in the context.

    Parameters:
        content: The message content/text.
        timestamp: When the user turn started.
        user_id: Optional identifier for the user.

    """

    content: str
    timestamp: str
    user_id: Optional[str] = None


@dataclass
class AssistantTurnStoppedMessage:
    """An assistant turn stopped message containing an assistant transcript update.

    A message in a conversation transcript containing the assistant
    content. This is the aggregated transcript that is then used in the context.

    Parameters:
        content: The message content/text.
        timestamp: When the assistant turn started.

    """

    content: str
    timestamp: str


@dataclass
class AssistantThoughtMessage:
    """An assistant thought message containing an assistant thought update.

    A message in a conversation transcript containing the assistant thought
    content.

    Parameters:
        content: The message content/text.
        timestamp: When the thought started.

    """

    content: str
    timestamp: str


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
    async def push_aggregation(self) -> str:
        """Push the current aggregation downstream.

        Returns:
            The pushed aggregation.
        """
        pass

    def aggregation_string(self) -> str:
        """Get the current aggregation as a string.

        Returns:
            The concatenated aggregation string.
        """
        return concatenate_aggregated_text(self._aggregation)


class LLMUserAggregator(LLMContextAggregator):
    """User LLM aggregator that aggregates user input during active user turns.

    This aggregator uses a turn controller and operates within turn boundaries
    defined by the controller's configured user turn strategies. User turn start
    strategies indicate when a user turn begins, while user turn stop strategies
    signal when the user turn has ended.

    The aggregator collects and aggregates speech-to-text transcriptions that
    occur while a user turn is active and pushes the final aggregation when the
    user turn is finished.

    Event handlers available:

    - on_user_turn_started: Called when the user turn starts
    - on_user_turn_stopped: Called when the user turn ends
    - on_user_turn_stop_timeout: Called when no user turn stop strategy triggers
    - on_user_turn_idle: Called when the user has been idle for the configured timeout
    - on_user_mute_started: Called when the user becomes muted
    - on_user_mute_stopped: Called when the user becomes unmuted

    Example::

        @aggregator.event_handler("on_user_turn_started")
        async def on_user_turn_started(aggregator, strategy: BaseUserTurnStartStrategy):
            ...

        @aggregator.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(aggregator, strategy: BaseUserTurnStopStrategy, message: UserTurnStoppedMessage):
            ...

        @aggregator.event_handler("on_user_turn_stop_timeout")
        async def on_user_turn_stop_timeout(aggregator):
            ...

        @aggregator.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(aggregator):
            ...

        @aggregator.event_handler("on_user_mute_started")
        async def on_user_mute_started(aggregator):
            ...

        @aggregator.event_handler("on_user_mute_stopped")
        async def on_user_mute_stopped(aggregator):
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

        self._register_event_handler("on_user_turn_started")
        self._register_event_handler("on_user_turn_stopped")
        self._register_event_handler("on_user_turn_stop_timeout")
        self._register_event_handler("on_user_turn_idle")
        self._register_event_handler("on_user_mute_started")
        self._register_event_handler("on_user_mute_stopped")

        user_turn_strategies = self._params.user_turn_strategies or UserTurnStrategies()

        self._user_is_muted = False
        self._user_turn_start_timestamp = ""

        self._user_turn_controller = UserTurnController(
            user_turn_strategies=user_turn_strategies,
            user_turn_stop_timeout=self._params.user_turn_stop_timeout,
        )
        self._user_turn_controller.add_event_handler("on_push_frame", self._on_push_frame)
        self._user_turn_controller.add_event_handler("on_broadcast_frame", self._on_broadcast_frame)
        self._user_turn_controller.add_event_handler(
            "on_user_turn_started", self._on_user_turn_started
        )
        self._user_turn_controller.add_event_handler(
            "on_user_turn_stopped", self._on_user_turn_stopped
        )
        self._user_turn_controller.add_event_handler(
            "on_user_turn_stop_timeout", self._on_user_turn_stop_timeout
        )

        # Optional user idle controller
        self._user_idle_controller: Optional[UserIdleController] = None
        if self._params.user_idle_timeout:
            self._user_idle_controller = UserIdleController(
                user_idle_timeout=self._params.user_idle_timeout
            )
            self._user_idle_controller.add_event_handler(
                "on_user_turn_idle", self._on_user_turn_idle
            )

        # VAD controller
        self._vad_controller: Optional[VADController] = None
        if self._params.vad_analyzer:
            self._vad_controller = VADController(self._params.vad_analyzer)
            self._vad_controller.add_event_handler("on_speech_started", self._on_vad_speech_started)
            self._vad_controller.add_event_handler("on_speech_stopped", self._on_vad_speech_stopped)
            self._vad_controller.add_event_handler(
                "on_speech_activity", self._on_vad_speech_activity
            )
            self._vad_controller.add_event_handler("on_push_frame", self._on_push_frame)
            self._vad_controller.add_event_handler("on_broadcast_frame", self._on_broadcast_frame)

        # NOTE(aleix): Probably just needed temporarily. This was added to
        # prevent processing self-queued frames (SpeechControlParamsFrame)
        # pushed by strategies.
        self._self_queued_frames = set()

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

        if self._vad_controller:
            await self._vad_controller.process_frame(frame)

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

        await self._user_turn_controller.process_frame(frame)

        if self._user_idle_controller:
            await self._user_idle_controller.process_frame(frame)

    async def push_aggregation(self) -> str:
        """Push the current aggregation."""
        if len(self._aggregation) == 0:
            return ""

        aggregation = self.aggregation_string()
        await self.reset()
        self._context.add_message({"role": self.role, "content": aggregation})
        await self.push_context_frame()

        return aggregation

    async def _start(self, frame: StartFrame):
        await self._user_turn_controller.setup(self.task_manager)

        if self._user_idle_controller:
            await self._user_idle_controller.setup(self.task_manager)

        for s in self._params.user_mute_strategies:
            await s.setup(self.task_manager)

        # Enable incomplete turn filtering on the LLM if configured
        if self._params.filter_incomplete_user_turns:
            # Get config or use defaults
            config = self._params.user_turn_completion_config or UserTurnCompletionConfig()

            # Enable the feature on the LLM with config
            await self.push_frame(
                LLMUpdateSettingsFrame(
                    settings={
                        "filter_incomplete_user_turns": True,
                        "user_turn_completion_config": config,
                    }
                )
            )

            # Auto-inject turn completion instructions into context
            self._context.add_message({"role": "system", "content": config.completion_instructions})

    async def _stop(self, frame: EndFrame):
        await self._maybe_emit_user_turn_stopped(on_session_end=True)
        await self._cleanup()

    async def _cancel(self, frame: CancelFrame):
        await self._maybe_emit_user_turn_stopped(on_session_end=True)
        await self._cleanup()

    async def _cleanup(self):
        await self._user_turn_controller.cleanup()

        if self._user_idle_controller:
            await self._user_idle_controller.cleanup()

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

            # Emit mute state change events
            if self._user_is_muted:
                await self._call_event_handler("on_user_mute_started")
            else:
                await self._call_event_handler("on_user_mute_stopped")

        return should_mute_frame

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
        if frame.id in self._self_queued_frames:
            return

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

        await self._user_turn_controller.update_strategies(ExternalUserTurnStrategies())

    async def _handle_transcription(self, frame: TranscriptionFrame):
        text = frame.text

        # Make sure we really have some text.
        if not text.strip():
            return

        # Transcriptions never include inter-part spaces (so far).
        self._aggregation.append(
            TextPartForConcatenation(
                text, includes_inter_part_spaces=frame.includes_inter_frame_spaces
            )
        )

    async def _internal_queue_frame(
        self,
        frame: Frame,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
        callback: Optional[FrameCallback] = None,
    ):
        """Queues the given frame to ourselves."""
        self._self_queued_frames.add(frame.id)
        await self.queue_frame(frame, direction, callback)

    async def _queued_broadcast_frame(self, frame_cls: Type[Frame], **kwargs):
        """Broadcasts a frame upstream and queues it for internal processing.

        Queues the frame so it flows through `process_frame` and is handled
        internally (e.g. by the `UserTurnController`). The upstream frame is
        pushed directly.

        Args:
            frame_cls: The class of the frame to be broadcasted.
            **kwargs: Keyword arguments to be passed to the frame's constructor.

        """
        await self._internal_queue_frame(frame_cls(**kwargs))
        await self.push_frame(frame_cls(**kwargs), FrameDirection.UPSTREAM)

    async def _on_push_frame(
        self, controller, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ):
        await self._internal_queue_frame(frame, direction)

    async def _on_broadcast_frame(self, controller, frame_cls: Type[Frame], **kwargs):
        await self._queued_broadcast_frame(frame_cls, **kwargs)

    async def _on_vad_speech_started(self, controller):
        await self._queued_broadcast_frame(VADUserStartedSpeakingFrame)

    async def _on_vad_speech_stopped(self, controller):
        await self._queued_broadcast_frame(VADUserStoppedSpeakingFrame)

    async def _on_vad_speech_activity(self, controller):
        await self._queued_broadcast_frame(UserSpeakingFrame)

    async def _on_user_turn_started(
        self,
        controller: UserTurnController,
        strategy: BaseUserTurnStartStrategy,
        params: UserTurnStartedParams,
    ):
        logger.debug(f"{self}: User started speaking (strategy: {strategy})")

        self._user_turn_start_timestamp = time_now_iso8601()

        if params.enable_user_speaking_frames:
            await self.broadcast_frame(UserStartedSpeakingFrame)

        if params.enable_interruptions and self._allow_interruptions:
            await self.push_interruption_task_frame_and_wait()

        await self._call_event_handler("on_user_turn_started", strategy)

    async def _on_user_turn_stopped(
        self,
        controller: UserTurnController,
        strategy: BaseUserTurnStopStrategy,
        params: UserTurnStoppedParams,
    ):
        logger.debug(f"{self}: User stopped speaking (strategy: {strategy})")

        if params.enable_user_speaking_frames:
            await self.broadcast_frame(UserStoppedSpeakingFrame)

        await self._maybe_emit_user_turn_stopped(strategy)

    async def _on_user_turn_stop_timeout(self, controller):
        await self._call_event_handler("on_user_turn_stop_timeout")

    async def _on_user_turn_idle(self, controller):
        await self._call_event_handler("on_user_turn_idle")

    async def _maybe_emit_user_turn_stopped(
        self,
        strategy: Optional[BaseUserTurnStopStrategy] = None,
        on_session_end: bool = False,
    ):
        """Maybe emit user turn stopped event.

        Args:
            strategy: The strategy that triggered the turn stop.
            on_session_end: If True, only emit if there's unemitted content
                (avoids duplicate events when session ends).
        """
        aggregation = await self.push_aggregation()
        if not on_session_end or aggregation:
            message = UserTurnStoppedMessage(
                content=aggregation, timestamp=self._user_turn_start_timestamp
            )
            await self._call_event_handler("on_user_turn_stopped", strategy, message)
            self._user_turn_start_timestamp = ""


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

    Event handlers available:

    - on_assistant_turn_started: Called when the assistant turn starts
    - on_assistant_turn_stopped: Called when the assistant turn ends
    - on_assistant_thought: Called when an assistant thought is available

    Example::

        @aggregator.event_handler("on_assistant_turn_started")
        async def on_assistant_turn_started(aggregator):
            ...

        @aggregator.event_handler("on_assistant_turn_stopped")
        async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
            ...

        @aggregator.event_handler("on_assistant_thought")
        async def on_assistant_thought(aggregator, message: AssistantThoughtMessage):
            ...

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
        self._function_calls_image_results: Dict[str, UserImageRawFrame] = {}
        self._context_updated_tasks: Set[asyncio.Task] = set()

        self._assistant_turn_start_timestamp = ""

        self._thought_append_to_context = False
        self._thought_llm: str = ""
        self._thought_aggregation: List[TextPartForConcatenation] = []
        self._thought_start_time: str = ""

        self._register_event_handler("on_assistant_turn_started")
        self._register_event_handler("on_assistant_turn_stopped")
        self._register_event_handler("on_assistant_thought")

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
        self._thought_append_to_context = False
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
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._handle_end_or_cancel(frame)
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
        else:
            await self.push_frame(frame, direction)

    async def push_aggregation(self) -> str:
        """Push the current assistant aggregation with timestamp."""
        if not self._aggregation:
            return ""

        aggregation = self.aggregation_string()
        await self.reset()

        self._context.add_message({"role": "assistant", "content": aggregation})

        # Push context frame
        await self.push_context_frame()

        # Push timestamp frame with current time
        timestamp_frame = LLMContextAssistantTimestampFrame(timestamp=time_now_iso8601())
        await self.push_frame(timestamp_frame)

        return aggregation

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
        await self._trigger_assistant_turn_stopped()
        self._started = 0
        await self.reset()

    async def _handle_end_or_cancel(self, frame: Frame):
        await self._trigger_assistant_turn_stopped()
        self._started = 0

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
                            "arguments": json.dumps(frame.arguments, ensure_ascii=False),
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
            result = json.dumps(frame.result, ensure_ascii=False)
            self._update_function_call_result(frame.function_name, frame.tool_call_id, result)
        else:
            self._update_function_call_result(frame.function_name, frame.tool_call_id, "COMPLETED")

        run_llm = False

        # Append any images that were generated by function calls.
        if frame.tool_call_id in self._function_calls_image_results:
            image_frame = self._function_calls_image_results[frame.tool_call_id]

            del self._function_calls_image_results[frame.tool_call_id]

            # If an image frame has been added to the context, let's run inference.
            run_llm = await self._maybe_append_image_to_context(image_frame)

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
        function_call = self._function_calls_in_progress.get(frame.tool_call_id)
        if function_call and function_call.cancel_on_interruption:
            # Update context with the function call cancellation
            self._update_function_call_result(frame.function_name, frame.tool_call_id, "CANCELLED")
            del self._function_calls_in_progress[frame.tool_call_id]

    async def _handle_user_image_frame(self, frame: UserImageRawFrame):
        image_appended = False

        # Check if this image is a result of a function call.
        if (
            frame.request
            and frame.request.tool_call_id
            and frame.request.tool_call_id in self._function_calls_in_progress
        ):
            self._function_calls_image_results[frame.request.tool_call_id] = frame

            # Call the result_callback if provided. This signals that the image
            # has been retrieved and the function call can now complete.
            if frame.request.result_callback:
                await frame.request.result_callback(None)
        else:
            image_appended = await self._maybe_append_image_to_context(frame)

        if image_appended:
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
        await self._trigger_assistant_turn_started()

    async def _handle_llm_end(self, _: LLMFullResponseEndFrame):
        self._started -= 1
        await self._trigger_assistant_turn_stopped()

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
        self._thought_append_to_context = frame.append_to_context
        self._thought_llm = frame.llm
        self._thought_start_time = time_now_iso8601()

    async def _handle_thought_text(self, frame: LLMThoughtTextFrame):
        if not self._started:
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
        if not self._started:
            return

        thought = concatenate_aggregated_text(self._thought_aggregation)

        if self._thought_append_to_context:
            llm = self._thought_llm
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

        message = AssistantThoughtMessage(content=thought, timestamp=self._thought_start_time)

        await self._reset_thought_aggregation()

        await self._call_event_handler("on_assistant_thought", message)

    async def _maybe_append_image_to_context(self, frame: UserImageRawFrame) -> bool:
        if not frame.append_to_context:
            return False

        logger.debug(f"{self} Appending UserImageRawFrame to LLM context (size: {frame.size})")

        await self._context.add_image_frame_message(
            format=frame.format,
            size=frame.size,
            image=frame.image,
            text=frame.text,
        )

        return True

    def _update_function_call_result(self, function_name: str, tool_call_id: str, result: Any):
        for message in self._context.get_messages():
            if (
                not isinstance(message, LLMSpecificMessage)
                and message["role"] == "tool"
                and message["tool_call_id"]
                and message["tool_call_id"] == tool_call_id
            ):
                message["content"] = result

    def _context_updated_task_finished(self, task: asyncio.Task):
        self._context_updated_tasks.discard(task)

    async def _trigger_assistant_turn_started(self):
        self._assistant_turn_start_timestamp = time_now_iso8601()

        await self._call_event_handler("on_assistant_turn_started")

    async def _trigger_assistant_turn_stopped(self):
        aggregation = await self.push_aggregation()
        if aggregation:
            # Strip turn completion markers from the transcript
            content = self._maybe_strip_turn_completion_markers(aggregation)
            message = AssistantTurnStoppedMessage(
                content=content, timestamp=self._assistant_turn_start_timestamp
            )
            await self._call_event_handler("on_assistant_turn_stopped", message)

            self._assistant_turn_start_timestamp = ""

    def _maybe_strip_turn_completion_markers(self, text: str) -> str:
        """Strip turn completion markers from assistant transcript.

        These markers (✓, ○, ◐) are used internally for turn completion
        detection and shouldn't appear in the final transcript.
        """
        from pipecat.turns.user_turn_completion_mixin import (
            USER_TURN_COMPLETE_MARKER,
            USER_TURN_INCOMPLETE_LONG_MARKER,
            USER_TURN_INCOMPLETE_SHORT_MARKER,
        )

        marker_found = False
        for marker in (
            USER_TURN_COMPLETE_MARKER,
            USER_TURN_INCOMPLETE_SHORT_MARKER,
            USER_TURN_INCOMPLETE_LONG_MARKER,
        ):
            if marker in text:
                text = text.replace(marker, "")
                marker_found = True

        # Only strip whitespace if we removed a marker
        return text.strip() if marker_found else text


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

    def __iter__(self):
        """Allow tuple unpacking of the aggregator pair.

        This enables both usage patterns::
            pair = LLMContextAggregatorPair(context)  # Returns the instance
            user, assistant = LLMContextAggregatorPair(context)  # Unpacks into tuple

        Yields:
            The user aggregator, then the assistant aggregator.
        """
        return iter((self._user, self._assistant))
