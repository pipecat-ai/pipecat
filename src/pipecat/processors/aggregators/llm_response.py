#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from abc import abstractmethod
from typing import Dict, List

from loguru import logger

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EmulateUserStartedSpeakingFrame,
    EmulateUserStoppedSpeakingFrame,
    EndFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMMessagesFrame,
    LLMMessagesUpdateFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
    OpenAILLMContextAssistantTimestampFrame,
    StartFrame,
    StartInterruptionFrame,
    TextFrame,
    TranscriptionFrame,
    UserImageRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.time import time_now_iso8601


class LLMFullResponseAggregator(FrameProcessor):
    """This is an LLM aggregator that aggregates a full LLM completion. It
    aggregates LLM text frames (tokens) received between
    `LLMFullResponseStartFrame` and `LLMFullResponseEndFrame`. Every full
    completion is returned via the "on_completion" event handler:

       @aggregator.event_handler("on_completion")
       async def on_completion(
           aggregator: LLMFullResponseAggregator,
           completion: str,
           completed: bool,
       )

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._aggregation = ""
        self._started = False

        self._register_event_handler("on_completion")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartInterruptionFrame):
            await self._call_event_handler("on_completion", self._aggregation, False)
            self._aggregation = ""
            self._started = False
        elif isinstance(frame, LLMFullResponseStartFrame):
            await self._handle_llm_start(frame)
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._handle_llm_end(frame)
        elif isinstance(frame, LLMTextFrame):
            await self._handle_llm_text(frame)

        await self.push_frame(frame, direction)

    async def _handle_llm_start(self, _: LLMFullResponseStartFrame):
        self._started = True

    async def _handle_llm_end(self, _: LLMFullResponseEndFrame):
        await self._call_event_handler("on_completion", self._aggregation, True)
        self._started = False
        self._aggregation = ""

    async def _handle_llm_text(self, frame: TextFrame):
        if not self._started:
            return
        self._aggregation += frame.text


class BaseLLMResponseAggregator(FrameProcessor):
    """This is the base class for all LLM response aggregators. These
    aggregators process incoming frames and aggregate content until they are
    ready to push the aggregation. In the case of a user, an aggregation might
    be a full transcription received from the STT service.

    The LLM response aggregators also keep a store (e.g. a message list or an
    LLM context) of the current conversation, that is, it stores the messages
    said by the user or by the bot.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    @abstractmethod
    def messages(self) -> List[dict]:
        """Returns the messages from the current conversation."""
        pass

    @property
    @abstractmethod
    def role(self) -> str:
        """Returns the role (e.g. user, assistant...) for this aggregator."""
        pass

    @abstractmethod
    def add_messages(self, messages):
        """Add the given messages to the conversation."""
        pass

    @abstractmethod
    def set_messages(self, messages):
        """Reset the conversation with the given messages."""
        pass

    @abstractmethod
    def set_tools(self, tools):
        """Set LLM tools to be used in the current conversation."""
        pass

    @abstractmethod
    def reset(self):
        """Reset the internals of this aggregator. This should not modify the
        internal messages."""
        pass

    @abstractmethod
    async def handle_aggregation(self, aggregation: str):
        """Adds the given aggregation to the aggregator. The aggregator can use
        a simple list of message or a context. It doesn't not push any frames.

        """
        pass

    @abstractmethod
    async def push_aggregation(self):
        """Pushes the current aggregation. For example, iN the case of context
        aggregation this might push a new context frame.

        """
        pass


class LLMContextResponseAggregator(BaseLLMResponseAggregator):
    """This is a base LLM aggregator that uses an LLM context to store the
    conversation. It pushes `OpenAILLMContextFrame` as an aggregation frame.

    """

    def __init__(self, *, context: OpenAILLMContext, role: str, **kwargs):
        super().__init__(**kwargs)
        self._context = context
        self._role = role

        self._aggregation = ""

    @property
    def messages(self) -> List[dict]:
        return self._context.get_messages()

    @property
    def role(self) -> str:
        return self._role

    @property
    def context(self):
        return self._context

    def get_context_frame(self) -> OpenAILLMContextFrame:
        return OpenAILLMContextFrame(context=self._context)

    async def push_context_frame(self, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        frame = self.get_context_frame()
        await self.push_frame(frame, direction)

    def add_messages(self, messages):
        self._context.add_messages(messages)

    def set_messages(self, messages):
        self._context.set_messages(messages)

    def set_tools(self, tools: List):
        self._context.set_tools(tools)

    def reset(self):
        self._aggregation = ""


class LLMUserContextAggregator(LLMContextResponseAggregator):
    """This is a user LLM aggregator that uses an LLM context to store the
    conversation. It aggregates transcriptions from the STT service and it has
    logic to handle multiple scenarios where transcriptions are received between
    VAD events (`UserStartedSpeakingFrame` and `UserStoppedSpeakingFrame`) or
    even outside or no VAD events at all.

    """

    def __init__(
        self,
        context: OpenAILLMContext,
        aggregation_timeout: float = 1.0,
        **kwargs,
    ):
        super().__init__(context=context, role="user", **kwargs)
        self._aggregation_timeout = aggregation_timeout

        self._seen_interim_results = False
        self._user_speaking = False
        self._emulating_vad = False
        self._waiting_for_aggregation = False

        self._aggregation_event = asyncio.Event()
        self._aggregation_task = None

    def reset(self):
        super().reset()
        self._seen_interim_results = False
        self._waiting_for_aggregation = False

    async def handle_aggregation(self, aggregation: str):
        self._context.add_message({"role": self.role, "content": self._aggregation})

    async def process_frame(self, frame: Frame, direction: FrameDirection):
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
        elif isinstance(frame, UserStartedSpeakingFrame):
            await self._handle_user_started_speaking(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._handle_user_stopped_speaking(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, TranscriptionFrame):
            await self._handle_transcription(frame)
        elif isinstance(frame, InterimTranscriptionFrame):
            await self._handle_interim_transcription(frame)
        elif isinstance(frame, LLMMessagesAppendFrame):
            self.add_messages(frame.messages)
        elif isinstance(frame, LLMMessagesUpdateFrame):
            self.set_messages(frame.messages)
        elif isinstance(frame, LLMSetToolsFrame):
            self.set_tools(frame.tools)
        else:
            await self.push_frame(frame, direction)

    async def push_aggregation(self):
        if len(self._aggregation) > 0:
            await self.handle_aggregation(self._aggregation)

            # Reset the aggregation. Reset it before pushing it down, otherwise
            # if the tasks gets cancelled we won't be able to clear things up.
            self.reset()

            frame = OpenAILLMContextFrame(self._context)
            await self.push_frame(frame)

    async def _start(self, frame: StartFrame):
        self._create_aggregation_task()

    async def _stop(self, frame: EndFrame):
        await self._cancel_aggregation_task()

    async def _cancel(self, frame: CancelFrame):
        await self._cancel_aggregation_task()

    async def _handle_user_started_speaking(self, _: UserStartedSpeakingFrame):
        self._user_speaking = True
        self._waiting_for_aggregation = True

    async def _handle_user_stopped_speaking(self, _: UserStoppedSpeakingFrame):
        self._user_speaking = False
        # We just stopped speaking. Let's see if there's some aggregation to
        # push. If the last thing we saw is an interim transcription, let's wait
        # pushing the aggregation as we will probably get a final transcription.
        if not self._seen_interim_results:
            await self.push_aggregation()

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
                await asyncio.wait_for(self._aggregation_event.wait(), self._aggregation_timeout)
                await self._maybe_push_bot_interruption()
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

    async def _maybe_push_bot_interruption(self):
        """If the user stopped speaking a while back and we got a transcription
        frame we might want to interrupt the bot.

        """
        if not self._user_speaking and not self._waiting_for_aggregation:
            # If we reach this case we received a transcription but VAD was not
            # able to detect voice (e.g. when you whisper a short
            # utterance). So, we need to emulate VAD (i.e. user start/stopped
            # speaking).
            await self.push_frame(EmulateUserStartedSpeakingFrame(), FrameDirection.UPSTREAM)
            self._emulating_vad = True


class LLMAssistantContextAggregator(LLMContextResponseAggregator):
    """This is an assistant LLM aggregator that uses an LLM context to store the
    conversation. It aggregates text frames received between
    `LLMFullResponseStartFrame` and `LLMFullResponseEndFrame`.

    """

    def __init__(self, context: OpenAILLMContext, *, expect_stripped_words: bool = True, **kwargs):
        super().__init__(context=context, role="assistant", **kwargs)
        self._expect_stripped_words = expect_stripped_words

        self._started = 0
        self._function_calls_in_progress: Dict[str, FunctionCallInProgressFrame] = {}

    async def handle_aggregation(self, aggregation: str):
        self._context.add_message({"role": "assistant", "content": aggregation})

    async def handle_function_call_in_progress(self, frame: FunctionCallInProgressFrame):
        pass

    async def handle_function_call_result(self, frame: FunctionCallResultFrame):
        pass

    async def handle_function_call_cancel(self, frame: FunctionCallCancelFrame):
        pass

    async def handle_user_image_frame(self, frame: UserImageRawFrame):
        pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartInterruptionFrame):
            await self._handle_interruptions(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMFullResponseStartFrame):
            await self._handle_llm_start(frame)
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._handle_llm_end(frame)
        elif isinstance(frame, TextFrame):
            await self._handle_text(frame)
        elif isinstance(frame, LLMMessagesAppendFrame):
            self.add_messages(frame.messages)
        elif isinstance(frame, LLMMessagesUpdateFrame):
            self.set_messages(frame.messages)
        elif isinstance(frame, LLMSetToolsFrame):
            self.set_tools(frame.tools)
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
        else:
            await self.push_frame(frame, direction)

    async def push_aggregation(self):
        if not self._aggregation:
            return

        aggregation = self._aggregation.strip()
        self.reset()

        if aggregation:
            await self.handle_aggregation(aggregation)

        # Push context frame
        await self.push_context_frame()

        # Push timestamp frame with current time
        timestamp_frame = OpenAILLMContextAssistantTimestampFrame(timestamp=time_now_iso8601())
        await self.push_frame(timestamp_frame)

    async def _handle_interruptions(self, frame: StartInterruptionFrame):
        await self.push_aggregation()
        self._started = 0
        self.reset()

    async def _handle_function_call_in_progress(self, frame: FunctionCallInProgressFrame):
        logger.debug(
            f"{self} FunctionCallInProgressFrame: [{frame.function_name}:{frame.tool_call_id}]"
        )
        await self.handle_function_call_in_progress(frame)
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

        await self.handle_function_call_result(frame)

        # Run inference if the function call result requires it.
        if frame.result:
            run_llm = False

            if properties and properties.run_llm is not None:
                # If the tool call result has a run_llm property, use it
                run_llm = properties.run_llm
            else:
                # Default behavior is to run the LLM if there are no function calls in progress
                run_llm = not bool(self._function_calls_in_progress)

            if run_llm:
                await self.push_context_frame(FrameDirection.UPSTREAM)

        # Emit the on_context_updated callback once the function call
        # result is added to the context
        if properties and properties.on_context_updated:
            await properties.on_context_updated()

    async def _handle_function_call_cancel(self, frame: FunctionCallCancelFrame):
        logger.debug(
            f"{self} FunctionCallCancelFrame: [{frame.function_name}:{frame.tool_call_id}]"
        )
        if frame.tool_call_id not in self._function_calls_in_progress:
            return

        if self._function_calls_in_progress[frame.tool_call_id].cancel_on_interruption:
            await self.handle_function_call_cancel(frame)
            del self._function_calls_in_progress[frame.tool_call_id]

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

        await self.handle_user_image_frame(frame)
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

        if self._expect_stripped_words:
            self._aggregation += f" {frame.text}" if self._aggregation else frame.text
        else:
            self._aggregation += frame.text


class LLMUserResponseAggregator(LLMUserContextAggregator):
    def __init__(self, messages: List[dict] = [], **kwargs):
        super().__init__(context=OpenAILLMContext(messages), **kwargs)

    async def push_aggregation(self):
        if len(self._aggregation) > 0:
            await self.handle_aggregation(self._aggregation)

            # Reset the aggregation. Reset it before pushing it down, otherwise
            # if the tasks gets cancelled we won't be able to clear things up.
            self.reset()

            frame = LLMMessagesFrame(self._context.messages)
            await self.push_frame(frame)


class LLMAssistantResponseAggregator(LLMAssistantContextAggregator):
    def __init__(self, messages: List[dict] = [], **kwargs):
        super().__init__(context=OpenAILLMContext(messages), **kwargs)

    async def push_aggregation(self):
        if len(self._aggregation) > 0:
            await self.handle_aggregation(self._aggregation)

            # Reset the aggregation. Reset it before pushing it down, otherwise
            # if the tasks gets cancelled we won't be able to clear things up.
            self.reset()

            frame = LLMMessagesFrame(self._context.messages)
            await self.push_frame(frame)
