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
from typing import Any, Dict, List, Literal, Optional, Set

from loguru import logger

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
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
    StartFrame,
    TextFrame,
    TranscriptionFrame,
    UserImageRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
    LLMContextMessage,
    LLMSpecificMessage,
    NotGiven,
)
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantAggregatorParams,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.base_interruption_strategy import BaseInterruptionStrategy
from pipecat.turns.base_speaking_strategy import BaseSpeakingStrategy
from pipecat.utils.string import concatenate_aggregated_text
from pipecat.utils.time import time_now_iso8601


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

        self._aggregation: List[str] = []

        # Whether to add spaces between text parts.
        # (Currently only used by LLMAssistantAggregator, but could be expanded
        # to LLMUserAggregator in the future if needed; that would require
        # additional work since LLMUserAggregator currently trims spaces from
        # incoming frames before determining whether it "really" received any
        # text).
        self._add_spaces = True

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
        return concatenate_aggregated_text(self._aggregation, self._add_spaces)


class LLMUserAggregator(LLMContextAggregator):
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
        params: Optional[LLMUserAggregatorParams] = None,
        **kwargs,
    ):
        """Initialize the user context aggregator.

        Args:
            context: The LLM context for conversation storage.
            params: Configuration parameters for aggregation behavior.
            **kwargs: Additional arguments. Supports deprecated 'aggregation_timeout'.
        """
        super().__init__(context=context, role="user", **kwargs)
        self._params = params or LLMUserAggregatorParams()

        self._user_speaking = False

    async def cleanup(self):
        """Clean up processor resources."""
        await super().cleanup()
        await self._cleanup()

    async def reset(self):
        """Reset the aggregation state and interruption strategies."""
        await super().reset()
        for s in self.interruption_strategies:
            s.reset()
        for s in self.speaking_strategies:
            s.reset()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames for user speech aggregation and context management.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        for strategy in self.interruption_strategies:
            await strategy.process_frame(frame)

        for strategy in self.speaking_strategies:
            await strategy.process_frame(frame)

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
        else:
            await self.push_frame(frame, direction)

    async def push_aggregation(self):
        """Push the current aggregation."""
        if len(self._aggregation) == 0:
            return

        aggregation = self.aggregation_string()
        await self.reset()
        self._context.add_message({"role": self.role, "content": aggregation})
        frame = LLMContextFrame(self._context)
        await self.push_frame(frame)

    async def _start(self, frame: StartFrame):
        for s in self.interruption_strategies:
            await s.setup(self.task_manager)
            s.add_event_handler("on_push_frame", self._on_push_frame)
            s.add_event_handler("on_should_interrupt", self._on_should_interrupt)

        for s in self.speaking_strategies:
            await s.setup(self.task_manager)
            s.add_event_handler("on_push_frame", self._on_push_frame)
            s.add_event_handler("on_should_speak", self._on_should_speak)

    async def _stop(self, frame: EndFrame):
        await self._cleanup()

    async def _cancel(self, frame: CancelFrame):
        await self._cleanup()

    async def _cleanup(self):
        for s in self.interruption_strategies:
            await s.cleanup()
        for s in self.speaking_strategies:
            await s.cleanup()

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

    async def _handle_transcription(self, frame: TranscriptionFrame):
        text = frame.text

        # Make sure we really have some text.
        if not text.strip():
            return

        self._aggregation.append(text)

    async def _on_should_interrupt(self, strategy: BaseInterruptionStrategy):
        await self._trigger_bot_interruption(strategy)

    async def _on_should_speak(self, strategy: BaseSpeakingStrategy):
        await self._trigger_bot_speech(strategy)

    async def _on_push_frame(
        self,
        strategy: BaseInterruptionStrategy | BaseSpeakingStrategy,
        frame: Frame,
        direction: FrameDirection,
    ):
        await self.push_frame(frame, direction)

    async def _trigger_bot_interruption(self, strategy: BaseInterruptionStrategy):
        """Generate an interruption if one of the strategies conditions is met."""
        if self._user_speaking:
            return

        self._user_speaking = True

        logger.debug(f"User started speaking ({strategy})")

        # Once we are interrupting, we reset all the interruption strategies.
        for s in self.interruption_strategies:
            s.reset()

        await self.push_frame(UserStartedSpeakingFrame())
        await self.push_frame(InterruptionFrame())

    async def _trigger_bot_speech(self, strategy: BaseSpeakingStrategy):
        """Generate an interruption if one of the strategies conditions is met."""
        if not self._user_speaking:
            return

        self._user_speaking = False

        logger.debug(f"User stopped speaking ({strategy})")

        # Reset all speaking strategies to start fresh.
        for s in self.speaking_strategies:
            s.reset()

        await self.push_frame(UserStoppedSpeakingFrame())
        await self.push_aggregation()


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

    @property
    def has_function_calls_in_progress(self) -> bool:
        """Check if there are any function calls currently in progress.

        Returns:
            True if function calls are in progress, False otherwise.
        """
        return bool(self._function_calls_in_progress)

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

        self._context.add_image_frame_message(
            format=frame.format,
            size=frame.size,
            image=frame.image,
            text=frame.text,
        )

        await self.push_aggregation()
        await self.push_context_frame(FrameDirection.UPSTREAM)

    async def _handle_llm_start(self, _: LLMFullResponseStartFrame):
        self._started += 1

    async def _handle_llm_end(self, _: LLMFullResponseEndFrame):
        self._started -= 1
        await self.push_aggregation()

    async def _handle_text(self, frame: TextFrame):
        if not self._started:
            return

        # Make sure we really have text (spaces count, too!)
        if len(frame.text) == 0:
            return

        # Track whether we need to add spaces between text parts
        # Assumption: we can just keep track of the latest frame's value
        self._add_spaces = not frame.includes_inter_frame_spaces

        self._aggregation.append(frame.text)

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
