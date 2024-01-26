import asyncio
import copy
import functools
from typing import AsyncGenerator, Awaitable, Callable
from dailyai.queue_aggregators import LLMAssistantContextAggregator, LLMContextAggregator, LLMUserContextAggregator
from dailyai.queue_frame import EndStreamQueueFrame, QueueFrame, TranscriptionQueueFrame


class InterruptibleConversationWrapper:

    def __init__(
        self,
        frame_generator: Callable[[], AsyncGenerator[QueueFrame, None]],
        runner: Callable[
            [str, LLMContextAggregator, LLMContextAggregator], Awaitable[None]
        ],
        interrupt: Callable[[], None],
        my_participant_id: str | None,
        llm_messages: list[dict[str, str]],
        llm_context_aggregator_in=LLMUserContextAggregator,
        llm_context_aggregator_out=LLMAssistantContextAggregator,
        delay_before_speech_seconds: float = 1.0,
    ):
        self._frame_generator: Callable[[], AsyncGenerator[QueueFrame, None]] = frame_generator
        self._runner: Callable[
            [str, LLMContextAggregator, LLMContextAggregator], Awaitable[None]
        ] = runner
        self._interrupt: Callable[[], None] = interrupt
        self._my_participant_id = my_participant_id
        self._messages: list[dict[str, str]] = llm_messages
        self._delay_before_speech_seconds = delay_before_speech_seconds
        self._llm_context_aggregator_in = llm_context_aggregator_in
        self._llm_context_aggregator_out = llm_context_aggregator_out

        self._current_phrase = ""

    def update_messages(self, new_messages: list[dict[str, str]], task: asyncio.Task | None):
        if task:
            if not task.cancelled():
                self._current_phrase = ""
                self._messages = new_messages

    async def speak_after_delay(self, user_speech, messages):
        await asyncio.sleep(self._delay_before_speech_seconds)
        tma_in = self._llm_context_aggregator_in(
            messages, self._my_participant_id, complete_sentences=False
        )
        tma_out = self._llm_context_aggregator_out(
            messages, self._my_participant_id
        )

        await self._runner(user_speech, tma_in, tma_out)

    async def run_conversation(self):
        current_response_task = None

        async for frame in self._frame_generator():
            if isinstance(frame, EndStreamQueueFrame):
                break
            elif not isinstance(frame, TranscriptionQueueFrame):
                continue

            if frame.participantId == self._my_participant_id:
                continue

            if current_response_task:
                current_response_task.cancel()
                self._interrupt()

            self._current_phrase += " " + frame.text
            current_llm_messages = copy.deepcopy(self._messages)
            current_response_task = asyncio.create_task(
                self.speak_after_delay(self._current_phrase, current_llm_messages)
            )
            current_response_task.add_done_callback(
                functools.partial(self.update_messages, current_llm_messages)
            )
