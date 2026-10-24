#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for how the user aggregator writes how the user spoke into the context."""

import pytest

from pipecat.expressive.perception import PerceptionSignal, format_perception
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    LLMContextUserTurnFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator,
    LLMUserAggregatorParams,
    UserTurnStoppedMessage,
)
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

EXPRESSIVE = PipelineParams(enable_expressive_conversation=True)
TIMEOUT = 0.1

FRUSTRATED = [PerceptionSignal("tone", labels=[("frustrated", 0.86), ("neutral", 0.10)])]
NEUTRAL = [PerceptionSignal("tone", labels=[("neutral", 0.81), ("positive", 0.12)])]


def _interim(words: str, perception: list[PerceptionSignal]) -> InterimTranscriptionFrame:
    return InterimTranscriptionFrame(text=words, user_id="", timestamp="now", perception=perception)


def _turn(
    words: str,
    perception: list[PerceptionSignal] | None = None,
    *,
    before: list[Frame] | None = None,
) -> list[Frame]:
    return [
        VADUserStartedSpeakingFrame(),
        *(before or []),
        TranscriptionFrame(text=words, user_id="", timestamp="now", perception=perception),
        SleepFrame(),
        VADUserStoppedSpeakingFrame(),
        SleepFrame(sleep=TIMEOUT + 0.1),
    ]


def _aggregator(context: LLMContext) -> LLMUserAggregator:
    return LLMUserAggregator(
        context,
        params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=TIMEOUT)]
            )
        ),
    )


async def _user_messages(*turns: list[Frame], params=EXPRESSIVE):
    context = LLMContext()
    aggregator = _aggregator(context)
    down, _ = await run_test(
        aggregator,
        frames_to_send=[frame for turn in turns for frame in turn],
        pipeline_params=params,
    )
    return [m["content"] for m in context.messages if m["role"] == "user"], down


def test_a_reading_formats_labels_with_confidences_and_values():
    signals = [*FRUSTRATED, PerceptionSignal("speech_rate", value=4.1)]

    assert format_perception(signals) == (
        "[heard: tone frustrated 0.86, neutral 0.10; speech_rate 4.1]"
    )


@pytest.mark.asyncio
async def test_the_reading_is_written_at_the_head_of_the_user_message():
    messages, down = await _user_messages(_turn("Can you tell me where my order is?", FRUSTRATED))

    assert messages == [
        "[heard: tone frustrated 0.86, neutral 0.10] Can you tell me where my order is?"
    ]
    turn_frames = [f for f in down if isinstance(f, LLMContextUserTurnFrame)]
    assert turn_frames[0].text == "Can you tell me where my order is?"


@pytest.mark.asyncio
async def test_nothing_is_written_without_expressive_conversation():
    messages, _ = await _user_messages(
        _turn("Can you tell me where my order is?", FRUSTRATED), params=None
    )

    assert messages == ["Can you tell me where my order is?"]


@pytest.mark.asyncio
async def test_a_provisional_reading_is_not_written():
    messages, _ = await _user_messages(
        _turn(
            "Can you tell me where my order is?",
            before=[_interim("Can you tell me", FRUSTRATED)],
        )
    )

    assert messages == ["Can you tell me where my order is?"]


@pytest.mark.asyncio
async def test_the_turns_latest_final_reading_is_written():
    messages, _ = await _user_messages(
        _turn(
            "where my order is?",
            FRUSTRATED,
            before=[
                TranscriptionFrame(
                    text="Can you tell me", user_id="", timestamp="now", perception=NEUTRAL
                )
            ],
        )
    )

    assert messages == [
        "[heard: tone frustrated 0.86, neutral 0.10] Can you tell me where my order is?"
    ]


@pytest.mark.asyncio
async def test_a_reading_belongs_to_its_own_turn():
    messages, _ = await _user_messages(
        _turn("Can you tell me where my order is?", FRUSTRATED),
        _turn("Thanks."),
    )

    assert messages == [
        "[heard: tone frustrated 0.86, neutral 0.10] Can you tell me where my order is?",
        "Thanks.",
    ]


async def _stopped(*turns: list[Frame], params=EXPRESSIVE) -> list[UserTurnStoppedMessage]:
    aggregator = _aggregator(LLMContext())
    stopped: list[UserTurnStoppedMessage] = []

    @aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message):
        stopped.append(message)

    await run_test(
        aggregator,
        frames_to_send=[frame for turn in turns for frame in turn],
        pipeline_params=params,
    )
    return stopped


@pytest.mark.asyncio
async def test_the_turn_stops_with_its_final_reading():
    stopped = await _stopped(
        _turn("Can you tell me where my order is?", FRUSTRATED),
        _turn("Thanks."),
    )

    assert [m.perception for m in stopped] == [FRUSTRATED, None]


@pytest.mark.asyncio
async def test_a_provisional_reading_is_not_reported_with_the_turn():
    stopped = await _stopped(
        _turn("Can you tell me where my order is?", before=[_interim("Can you tell me", NEUTRAL)])
    )

    assert stopped[0].perception is None


@pytest.mark.asyncio
async def test_nothing_is_reported_without_expressive_conversation():
    stopped = await _stopped(_turn("Can you tell me where my order is?", FRUSTRATED), params=None)

    assert stopped[0].perception is None
