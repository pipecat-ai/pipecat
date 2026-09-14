import pytest

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.frames.frames import (
    InputDTMFFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.dtmf_aggregator import DTMFAggregator
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_start import TranscriptionUserTurnStartStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

TS = "2026-01-01T00:00:00Z"


def _aggregator(use_interim=True):
    return LLMUserAggregator(
        LLMContext(),
        params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[TranscriptionUserTurnStartStrategy(use_interim=use_interim)],
                stop=[SpeechTimeoutUserTurnStopStrategy(timeout=0.05)],
            )
        ),
    )


def _interim(text):
    return InterimTranscriptionFrame(user_id="u", text=text, timestamp=TS)


def _final(text):
    return TranscriptionFrame(user_id="u", text=text, timestamp=TS)


async def _committed(frames, dtmf=False, **kwargs):
    agg = _aggregator(**kwargs)
    processor = Pipeline([DTMFAggregator(), agg]) if dtmf else agg
    await run_test(processor, frames_to_send=frames)
    return [m["content"] for m in agg.context.get_messages() if m.get("role") == "user"]


@pytest.mark.asyncio
async def test_interim_then_final_same_tick():
    committed = await _committed([_interim("what is"), _final("what is my balance")])
    assert committed == ["what is my balance"]


@pytest.mark.asyncio
async def test_interim_gap_then_final():
    committed = await _committed(
        [_interim("what is"), SleepFrame(sleep=0.02), _final("what is my balance")]
    )
    assert committed == ["what is my balance"]


@pytest.mark.asyncio
async def test_final_then_final_same_tick():
    committed = await _committed([_final("one"), _final("two")])
    assert committed == ["one two"]


@pytest.mark.asyncio
async def test_final_then_final_no_interim():
    committed = await _committed([_final("one"), _final("two")], use_interim=False)
    assert committed == ["one two"]


@pytest.mark.asyncio
async def test_speech_final_plus_keypad_same_tick():
    keys = [InputDTMFFrame(KeypadEntry.ONE), InputDTMFFrame(KeypadEntry.POUND)]
    committed = await _committed([_final("what is my balance"), *keys], dtmf=True)
    assert committed == ["DTMF: 1# what is my balance"]


@pytest.mark.asyncio
async def test_speech_final_gap_then_keypad():
    keys = [InputDTMFFrame(KeypadEntry.ONE), InputDTMFFrame(KeypadEntry.POUND)]
    committed = await _committed(
        [_final("what is my balance"), SleepFrame(sleep=0.02), *keys], dtmf=True
    )
    assert committed == ["what is my balance DTMF: 1#"]
