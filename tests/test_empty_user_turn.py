#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    FunctionCallFromLLM,
    FunctionCallsStartedFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TranscriptionFrame,
    TTSTextFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.empty_user_turn import (
    DEFAULT_EMPTY_USER_TURN_INTERRUPTED_PROMPT,
    EmptyUserTurnConfig,
)
from pipecat.turns.user_start import (
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.text.base_text_aggregator import AggregationType

USER_TURN_STOP_TIMEOUT = 0.2
USER_SPEECH_TIMEOUT = 0.1

INTERRUPTED_PROMPT = "interrupted, nothing recognized"
IDLE_PROMPT = "idle, nothing recognized"


class ContextFrameRecorder(FrameProcessor):
    """Counts the inferences requested by the user aggregator."""

    def __init__(self):
        super().__init__()
        self.context_frames = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            self.context_frames += 1
        await self.push_frame(frame, direction)


def _user_params(**kwargs) -> LLMUserAggregatorParams:
    return LLMUserAggregatorParams(
        user_turn_strategies=UserTurnStrategies(
            start=[VADUserTurnStartStrategy(), TranscriptionUserTurnStartStrategy()],
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=USER_SPEECH_TIMEOUT)],
        ),
        user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        **kwargs,
    )


def _empty_turn() -> list[Frame]:
    """A user turn with voice activity and no transcript, closed by the watchdog."""
    return [
        VADUserStartedSpeakingFrame(),
        VADUserStoppedSpeakingFrame(),
        SleepFrame(sleep=USER_TURN_STOP_TIMEOUT + 0.2),
    ]


def _transcribed_turn(text: str) -> list[Frame]:
    return [
        VADUserStartedSpeakingFrame(),
        TranscriptionFrame(text=text, user_id="", timestamp="now"),
        VADUserStoppedSpeakingFrame(),
        SleepFrame(sleep=USER_SPEECH_TIMEOUT + 0.1),
    ]


def _bot_idle() -> list[Frame]:
    """The bot has finished speaking and is waiting for the user."""
    return [BotStoppedSpeakingFrame(), SleepFrame()]


def _bot_speaking() -> list[Frame]:
    """A bot response that is still being spoken."""
    return [
        LLMFullResponseStartFrame(),
        BotStartedSpeakingFrame(),
        TTSTextFrame("Where would", aggregated_by=AggregationType.WORD),
        SleepFrame(),
    ]


def _developer_messages(context: LLMContext) -> list[str]:
    return [m["content"] for m in context.get_messages() if m.get("role") == "developer"]


class TestEmptyUserTurn(unittest.IsolatedAsyncioTestCase):
    async def _run(self, frames: list[Frame], **user_params) -> tuple[LLMContext, int]:
        context = LLMContext()
        user, assistant = LLMContextAggregatorPair(context, user_params=_user_params(**user_params))
        recorder = ContextFrameRecorder()
        await run_test(Pipeline([user, recorder, assistant]), frames_to_send=frames)
        return context, recorder.context_frames

    def _config(self, **kwargs) -> EmptyUserTurnConfig:
        return EmptyUserTurnConfig(interrupted_prompt=INTERRUPTED_PROMPT, **kwargs)

    async def test_interrupted_while_speaking(self):
        context, inferences = await self._run(
            [*_bot_speaking(), *_empty_turn()], empty_user_turn=self._config()
        )
        self.assertEqual(_developer_messages(context), [INTERRUPTED_PROMPT])
        self.assertEqual(inferences, 1)
        # The recovery comes after what the user heard of the interrupted response.
        messages = context.get_messages()
        self.assertEqual(messages[-2], {"role": "assistant", "content": "Where would"})

    async def test_interrupted_before_response_started(self):
        # The previous turn's response hadn't started when the user spoke:
        # nothing was heard, and the interruption cancelled it.
        context, inferences = await self._run(
            [*_bot_idle(), *_transcribed_turn("Tell me a story."), *_empty_turn()],
            empty_user_turn=self._config(),
        )
        self.assertEqual(_developer_messages(context), [INTERRUPTED_PROMPT])
        self.assertEqual(inferences, 2)

    async def test_before_bot_spoke(self):
        # Until the bot first finishes speaking it isn't waiting for the user,
        # e.g. its greeting may still be on the way.
        context, _ = await self._run(_empty_turn(), empty_user_turn=self._config())
        self.assertEqual(_developer_messages(context), [INTERRUPTED_PROMPT])

    async def test_idle_ignored_by_default(self):
        context, inferences = await self._run(
            [*_bot_idle(), *_empty_turn()], empty_user_turn=self._config()
        )
        self.assertEqual(_developer_messages(context), [])
        self.assertEqual(inferences, 0)

    async def test_idle_prompt(self):
        context, inferences = await self._run(
            [*_bot_idle(), *_empty_turn()],
            empty_user_turn=self._config(idle_prompt=IDLE_PROMPT),
        )
        self.assertEqual(_developer_messages(context), [IDLE_PROMPT])
        self.assertEqual(inferences, 1)

    async def test_idle_after_response_finished(self):
        context, _ = await self._run(
            [*_bot_speaking(), LLMFullResponseEndFrame(), *_bot_idle(), *_empty_turn()],
            empty_user_turn=self._config(idle_prompt=IDLE_PROMPT),
        )
        self.assertEqual(_developer_messages(context), [IDLE_PROMPT])

    async def test_enabled_by_default(self):
        context, inferences = await self._run([*_bot_speaking(), *_empty_turn()])
        self.assertEqual(_developer_messages(context), [DEFAULT_EMPTY_USER_TURN_INTERRUPTED_PROMPT])
        self.assertEqual(inferences, 1)

    async def test_disabled(self):
        context, inferences = await self._run(
            [*_bot_speaking(), *_empty_turn()], empty_user_turn=None
        )
        self.assertEqual(_developer_messages(context), [])
        self.assertEqual(inferences, 0)

    async def test_transcribed_turn_not_recovered(self):
        context, inferences = await self._run(
            [*_bot_speaking(), *_transcribed_turn("Hello!")], empty_user_turn=self._config()
        )
        self.assertEqual(_developer_messages(context), [])
        self.assertEqual(inferences, 1)

    async def test_consecutive_recoveries_are_bounded(self):
        # The second empty turn interrupts the recovery's own pending response,
        # but only one recovery in a row is allowed. A transcribed turn resets
        # the count.
        context, _ = await self._run(
            [
                *_bot_speaking(),
                *_empty_turn(),
                *_empty_turn(),
                *_transcribed_turn("Sorry, what?"),
                *_empty_turn(),
            ],
            empty_user_turn=self._config(),
        )
        self.assertEqual(_developer_messages(context), [INTERRUPTED_PROMPT, INTERRUPTED_PROMPT])

    async def test_function_call_in_progress_not_recovered(self):
        # The function call's result will run the LLM on its own.
        context, inferences = await self._run(
            [
                *_bot_idle(),
                FunctionCallsStartedFrame(
                    function_calls=[
                        FunctionCallFromLLM(
                            function_name="get_weather",
                            tool_call_id="1",
                            arguments={},
                            context=None,
                        )
                    ]
                ),
                SleepFrame(),
                *_empty_turn(),
            ],
            empty_user_turn=self._config(),
        )
        self.assertEqual(_developer_messages(context), [])
        self.assertEqual(inferences, 0)

    async def test_without_pair(self):
        context = LLMContext()
        user = LLMUserAggregator(context, params=_user_params(empty_user_turn=self._config()))
        await run_test(Pipeline([user]), frames_to_send=[*_bot_speaking(), *_empty_turn()])
        self.assertEqual(_developer_messages(context), [INTERRUPTED_PROMPT])


class TestEmptyUserTurnIdle(unittest.IsolatedAsyncioTestCase):
    async def _idle_fired(self, frames: list[Frame], **user_params) -> bool:
        context = LLMContext()
        user, assistant = LLMContextAggregatorPair(
            context, user_params=_user_params(user_idle_timeout=0.2, **user_params)
        )
        idle = False

        @user.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(aggregator):
            nonlocal idle
            idle = True

        await run_test(Pipeline([user, assistant]), frames_to_send=[*frames, SleepFrame(0.4)])
        return idle

    async def test_idle_timer_rearmed_after_empty_turn(self):
        # The output transport stops the bot when the user interrupts it, while
        # the user turn is in progress, so that doesn't start the timer.
        self.assertTrue(
            await self._idle_fired(
                [
                    *_bot_speaking(),
                    VADUserStartedSpeakingFrame(),
                    BotStoppedSpeakingFrame(),
                    VADUserStoppedSpeakingFrame(),
                    SleepFrame(sleep=USER_TURN_STOP_TIMEOUT + 0.2),
                ],
                empty_user_turn=None,
            )
        )

    async def test_idle_timer_not_rearmed_after_recovery(self):
        # The recovery response is on its way, so the user isn't idle yet.
        self.assertFalse(await self._idle_fired([*_bot_speaking(), *_empty_turn()]))


if __name__ == "__main__":
    unittest.main()
