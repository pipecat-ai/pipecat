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
from abc import abstractmethod
from typing import Any, Dict, List, Literal, Optional, Set

from loguru import logger

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.interruptions.base_interruption_strategy import BaseInterruptionStrategy
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EmulateUserStartedSpeakingFrame,
    EmulateUserStoppedSpeakingFrame,
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
    SpeechControlParamsFrame,
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

        self._aggregation: str = ""

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
        self._aggregation = ""

    @abstractmethod
    async def push_aggregation(self):
        """Push the current aggregation downstream."""
        pass


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

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames for user speech aggregation and context management.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

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
        elif isinstance(frame, InputAudioRawFrame):
            await self._handle_input_audio(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStartedSpeakingFrame):
            await self._handle_user_started_speaking(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._handle_user_stopped_speaking(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, BotStartedSpeakingFrame):
            await self._handle_bot_started_speaking(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._handle_bot_stopped_speaking(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, TranscriptionFrame):
            await self._handle_transcription(frame)
        elif isinstance(frame, InterimTranscriptionFrame):
            await self._handle_interim_transcription(frame)
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
        elif isinstance(frame, SpeechControlParamsFrame):
            self._vad_params = frame.vad_params
            self._turn_params = frame.turn_params
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _process_aggregation(self):
        """Process the current aggregation and push it downstream."""
        aggregation = self._aggregation
        await self.reset()
        self._context.add_message({"role": self.role, "content": aggregation})
        frame = LLMContextFrame(self._context)
        await self.push_frame(frame)

    async def push_aggregation(self):
        """Push the current aggregation based on interruption strategies and conditions."""
        if len(self._aggregation) > 0:
            if self.interruption_strategies and self._bot_speaking:
                should_interrupt = await self._should_interrupt_based_on_strategies()

                if should_interrupt:
                    logger.debug(
                        "Interruption conditions met - pushing interruption and aggregation"
                    )
                    await self.push_interruption_task_frame_and_wait()
                    await self._process_aggregation()
                else:
                    logger.debug("Interruption conditions not met - not pushing aggregation")
                    # Don't process aggregation, just reset it
                    await self.reset()
            else:
                # No interruption config - normal behavior (always push aggregation)
                await self._process_aggregation()
        # Handles the case where both the user and the bot are not speaking,
        # and the bot was previously speaking before the user interruption.
        # Normally, when the user stops speaking, new text is expected,
        # which triggers the bot to respond. However, if no new text
        # is received, this safeguard ensures
        # the bot doesn't hang indefinitely while waiting to speak again.
        elif not self._seen_interim_results and self._was_bot_speaking and not self._bot_speaking:
            logger.warning("User stopped speaking but no new aggregation received.")
            # Resetting it so we don't trigger this twice
            self._was_bot_speaking = False
            # TODO: we are not enabling this for now, due to some STT services which can take as long as 2 seconds two return a transcription
            # So we need more tests and probably make this feature configurable, disabled it by default.
            # We are just pushing the same previous context to be processed again in this case
            # await self.push_frame(LLMContextFrame(self._context))

    async def _should_interrupt_based_on_strategies(self) -> bool:
        """Check if interruption should occur based on configured strategies.

        Returns:
            True if any interruption strategy indicates interruption should occur.
        """

        async def should_interrupt(strategy: BaseInterruptionStrategy):
            await strategy.append_text(self._aggregation)
            return await strategy.should_interrupt()

        return any([await should_interrupt(s) for s in self._interruption_strategies])

    async def _start(self, frame: StartFrame):
        self._create_aggregation_task()

    async def _stop(self, frame: EndFrame):
        await self._cancel_aggregation_task()

    async def _cancel(self, frame: CancelFrame):
        await self._cancel_aggregation_task()

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

    async def _handle_input_audio(self, frame: InputAudioRawFrame):
        for s in self.interruption_strategies:
            await s.append_audio(frame.audio, frame.sample_rate)

    async def _handle_user_started_speaking(self, frame: UserStartedSpeakingFrame):
        self._user_speaking = True
        self._waiting_for_aggregation = True
        self._was_bot_speaking = self._bot_speaking

        # If we get a non-emulated UserStartedSpeakingFrame but we are in the
        # middle of emulating VAD, let's stop emulating VAD (i.e. don't send the
        # EmulateUserStoppedSpeakingFrame).
        if not frame.emulated and self._emulating_vad:
            self._emulating_vad = False

    async def _handle_user_stopped_speaking(self, _: UserStoppedSpeakingFrame):
        self._user_speaking = False
        # We just stopped speaking. Let's see if there's some aggregation to
        # push. If the last thing we saw is an interim transcription, let's wait
        # pushing the aggregation as we will probably get a final transcription.
        if len(self._aggregation) > 0:
            if not self._seen_interim_results:
                await self.push_aggregation()
        # Handles the case where both the user and the bot are not speaking,
        # and the bot was previously speaking before the user interruption.
        # So in this case we are resetting the aggregation timer
        elif not self._seen_interim_results and self._was_bot_speaking and not self._bot_speaking:
            # Reset aggregation timer.
            self._aggregation_event.set()

    async def _handle_bot_started_speaking(self, _: BotStartedSpeakingFrame):
        self._bot_speaking = True

    async def _handle_bot_stopped_speaking(self, _: BotStoppedSpeakingFrame):
        self._bot_speaking = False

    async def _handle_transcription(self, frame: TranscriptionFrame):
        text = frame.text

        # Make sure we really have some text.
        if not text.strip():
            return

        self._aggregation += f" {text}" if self._aggregation else text
        # We just got a final result, so let's reset interim results.
        self._seen_interim_results = False
        # Reset aggregation timer.
        self._aggregation_event.set()

    async def _handle_interim_transcription(self, _: InterimTranscriptionFrame):
        self._seen_interim_results = True

    def _create_aggregation_task(self):
        if not self._aggregation_task:
            self._aggregation_task = self.create_task(self._aggregation_task_handler())

    async def _cancel_aggregation_task(self):
        if self._aggregation_task:
            await self.cancel_task(self._aggregation_task)
            self._aggregation_task = None

    async def _aggregation_task_handler(self):
        while True:
            try:
                # The _aggregation_task_handler handles two distinct timeout scenarios:
                #
                # 1. When emulating_vad=True: Wait for emulated VAD timeout before
                #    pushing aggregation (simulating VAD behavior when no actual VAD
                #    detection occurred).
                #
                # 2. When emulating_vad=False: Use aggregation_timeout as a buffer
                #    to wait for potential late-arriving transcription frames after
                #    a real VAD event.
                #
                # For emulated VAD scenarios, the timeout strategy depends on whether
                # a turn analyzer is configured:
                #
                # - WITH turn analyzer: Use turn_emulated_vad_timeout parameter because
                #   the VAD's stop_secs is set very low (e.g. 0.2s) for rapid speech
                #   chunking to feed the turn analyzer. This low value is too fast
                #   for emulated VAD scenarios where we need to allow users time to
                #   finish speaking (e.g. 0.8s).
                #
                # - WITHOUT turn analyzer: Use VAD's stop_secs directly to maintain
                #   consistent user experience between real VAD detection and
                #   emulated VAD scenarios.
                if not self._emulating_vad:
                    timeout = self._params.aggregation_timeout
                elif self._turn_params:
                    timeout = self._params.turn_emulated_vad_timeout
                else:
                    # Use VAD stop_secs when no turn analyzer is present, fallback if no VAD params
                    timeout = (
                        self._vad_params.stop_secs
                        if self._vad_params
                        else self._params.turn_emulated_vad_timeout
                    )
                await asyncio.wait_for(self._aggregation_event.wait(), timeout=timeout)
                await self._maybe_emulate_user_speaking()
            except asyncio.TimeoutError:
                if not self._user_speaking:
                    await self.push_aggregation()

                # If we are emulating VAD we still need to send the user stopped
                # speaking frame.
                if self._emulating_vad:
                    await self.push_frame(
                        EmulateUserStoppedSpeakingFrame(), FrameDirection.UPSTREAM
                    )
                    self._emulating_vad = False
            finally:
                self._aggregation_event.clear()

    async def _maybe_emulate_user_speaking(self):
        """Maybe emulate user speaking based on transcription.

        Emulate user speaking if we got a transcription but it was not
        detected by VAD. Behavior when bot is speaking depends on the
        enable_emulated_vad_interruptions parameter.
        """
        # Check if we received a transcription but VAD was not able to detect
        # voice (e.g. when you whisper a short utterance). In that case, we need
        # to emulate VAD (i.e. user start/stopped speaking), but we do it only
        # if the bot is not speaking. If the bot is speaking and we really have
        # a short utterance we don't really want to interrupt the bot.
        if (
            not self._user_speaking
            and not self._waiting_for_aggregation
            and len(self._aggregation) > 0
        ):
            if self._bot_speaking and not self._params.enable_emulated_vad_interruptions:
                # If emulated VAD interruptions are disabled and bot is speaking, ignore
                logger.debug("Ignoring user speaking emulation, bot is speaking.")
                await self.reset()
            else:
                # Either bot is not speaking, or emulated VAD interruptions are enabled
                # - trigger user speaking emulation.
                await self.push_frame(EmulateUserStartedSpeakingFrame(), FrameDirection.UPSTREAM)
                self._emulating_vad = True


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
            **kwargs: Additional arguments. Supports deprecated 'expect_stripped_words'.
        """
        super().__init__(context=context, role="assistant", **kwargs)
        self._params = params or LLMAssistantAggregatorParams()

        if "expect_stripped_words" in kwargs:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "Parameter 'expect_stripped_words' is deprecated, use 'params' instead.",
                    DeprecationWarning,
                )

            self._params.expect_stripped_words = kwargs["expect_stripped_words"]

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
        elif isinstance(frame, UserImageRawFrame) and frame.request and frame.request.tool_call_id:
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

        aggregation = self._aggregation.strip()
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
        logger.debug(
            f"{self} UserImageRawFrame: [{frame.request.function_name}:{frame.request.tool_call_id}]"
        )

        if frame.request.tool_call_id not in self._function_calls_in_progress:
            logger.warning(
                f"UserImageRawFrame tool_call_id [{frame.request.tool_call_id}] is not running"
            )
            return

        del self._function_calls_in_progress[frame.request.tool_call_id]

        # Update context with the image frame
        self._update_function_call_result(
            frame.request.function_name, frame.request.tool_call_id, "COMPLETED"
        )
        self._context.add_image_frame_message(
            format=frame.format,
            size=frame.size,
            image=frame.image,
            text=frame.request.context,
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

        if self._params.expect_stripped_words:
            self._aggregation += f" {frame.text}" if self._aggregation else frame.text
        else:
            self._aggregation += frame.text

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
