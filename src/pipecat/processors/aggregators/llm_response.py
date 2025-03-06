#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import time
from abc import abstractmethod
from typing import List

from pipecat.frames.frames import (
    CancelFrame,
    EmulateUserStartedSpeakingFrame,
    EmulateUserStoppedSpeakingFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMMessagesFrame,
    LLMMessagesUpdateFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
    StartFrame,
    StartInterruptionFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


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
    async def push_aggregation(self):
        pass


class LLMResponseAggregator(BaseLLMResponseAggregator):
    """This is a base LLM aggregator that uses a simple list of messages to
    store the conversation. It pushes `LLMMessagesFrame` as an aggregation
    frame.

    """

    def __init__(
        self,
        *,
        messages: List[dict],
        role: str = "user",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._messages = messages
        self._role = role

        self._aggregation = ""

        self.reset()

    @property
    def messages(self) -> List[dict]:
        return self._messages

    @property
    def role(self) -> str:
        return self._role

    def add_messages(self, messages):
        self._messages.extend(messages)

    def set_messages(self, messages):
        self.reset()
        self._messages.clear()
        self._messages.extend(messages)

    def set_tools(self, tools):
        pass

    def reset(self):
        self._aggregation = ""

    async def push_aggregation(self):
        if len(self._aggregation) > 0:
            self._messages.append({"role": self._role, "content": self._aggregation})

            # Reset the aggregation. Reset it before pushing it down, otherwise
            # if the tasks gets cancelled we won't be able to clear things up.
            self._aggregation = ""

            frame = LLMMessagesFrame(self._messages)
            await self.push_frame(frame)

            # Reset our accumulator state.
            self.reset()


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

    async def push_aggregation(self):
        if len(self._aggregation) > 0:
            self._context.add_message({"role": self.role, "content": self._aggregation})

            # Reset the aggregation. Reset it before pushing it down, otherwise
            # if the tasks gets cancelled we won't be able to clear things up.
            self._aggregation = ""

            frame = OpenAILLMContextFrame(self._context)
            await self.push_frame(frame)

            # Reset our accumulator state.
            self.reset()


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
        bot_interruption_timeout: float = 2.0,
        **kwargs,
    ):
        super().__init__(context=context, role="user", **kwargs)
        self._aggregation_timeout = aggregation_timeout
        self._bot_interruption_timeout = bot_interruption_timeout

        self._seen_interim_results = False
        self._user_speaking = False
        self._last_user_speaking_time = 0
        self._emulating_vad = False

        self._aggregation_event = asyncio.Event()
        self._aggregation_task = None

        self.reset()

    def reset(self):
        super().reset()
        self._seen_interim_results = False

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

    async def _start(self, frame: StartFrame):
        self._create_aggregation_task()

    async def _stop(self, frame: EndFrame):
        await self._cancel_aggregation_task()

    async def _cancel(self, frame: CancelFrame):
        await self._cancel_aggregation_task()

    async def _handle_user_started_speaking(self, _: UserStartedSpeakingFrame):
        self._last_user_speaking_time = time.time()
        self._user_speaking = True

    async def _handle_user_stopped_speaking(self, _: UserStoppedSpeakingFrame):
        self._last_user_speaking_time = time.time()
        self._user_speaking = False
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
        if not self._user_speaking:
            diff_time = time.time() - self._last_user_speaking_time
            if diff_time > self._bot_interruption_timeout:
                # If we reach this case we received a transcription but VAD was
                # not able to detect voice (e.g. when you whisper a short
                # utterance). So, we need to emulate VAD (i.e. user
                # start/stopped speaking).
                await self.push_frame(EmulateUserStartedSpeakingFrame(), FrameDirection.UPSTREAM)
                self._emulating_vad = True

                # Reset time so we don't interrupt again right away.
                self._last_user_speaking_time = time.time()


class LLMAssistantContextAggregator(LLMContextResponseAggregator):
    """This is an assistant LLM aggregator that uses an LLM context to store the
    conversation. It aggregates text frames received between
    `LLMFullResponseStartFrame` and `LLMFullResponseEndFrame`.

    """

    def __init__(self, context: OpenAILLMContext, *, expect_stripped_words: bool = True, **kwargs):
        super().__init__(context=context, role="assistant", **kwargs)
        self._expect_stripped_words = expect_stripped_words

        self._started = False

        self.reset()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartInterruptionFrame):
            await self.push_aggregation()
            # Reset anyways
            self.reset()
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
        else:
            await self.push_frame(frame, direction)

    async def _handle_llm_start(self, _: LLMFullResponseStartFrame):
        self._started = True

    async def _handle_llm_end(self, _: LLMFullResponseEndFrame):
        self._started = False
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
            self._context.add_message({"role": self.role, "content": self._aggregation})

            # Reset the aggregation. Reset it before pushing it down, otherwise
            # if the tasks gets cancelled we won't be able to clear things up.
            self._aggregation = ""

            frame = LLMMessagesFrame(self._context.messages)
            await self.push_frame(frame)

            # Reset our accumulator state.
            self.reset()


class LLMAssistantResponseAggregator(LLMAssistantContextAggregator):
    def __init__(self, messages: List[dict] = [], **kwargs):
        super().__init__(context=OpenAILLMContext(messages), **kwargs)

    async def push_aggregation(self):
        if len(self._aggregation) > 0:
            self._context.add_message({"role": self.role, "content": self._aggregation})

            # Reset the aggregation. Reset it before pushing it down, otherwise
            # if the tasks gets cancelled we won't be able to clear things up.
            self._aggregation = ""

            frame = LLMMessagesFrame(self._context.messages)
            await self.push_frame(frame)

            # Reset our accumulator state.
            self.reset()
