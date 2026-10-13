#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for a TTS service holding a speculative response until its turn is confirmed."""

import unittest
from collections.abc import AsyncGenerator

from pipecat.frames.frames import (
    EagerEndOfTurnCancelFrame,
    EagerTranscriptionFrame,
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_turn_strategies import EagerUserTurnStrategies
from pipecat.utils.time import seconds_to_nanoseconds

_SAMPLE_RATE = 16000


class WordTTS(TTSService):
    """Speaks each sentence at once, timing its words 0.1 s apart."""

    def __init__(self, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=False,
            sample_rate=_SAMPLE_RATE,
            **kwargs,
        )

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        word_times = [(w, i * 0.1) for i, w in enumerate(text.split())]
        await self.add_word_timestamps(word_times, context_id=context_id)
        yield TTSAudioRawFrame(
            audio=b"\x00\x01" * 320,
            sample_rate=_SAMPLE_RATE,
            num_channels=1,
            context_id=context_id,
        )


class EchoLLM(LLMService):
    """Answers with the transcript it was handed, so a stale answer is visible."""

    def __init__(self, **kwargs):
        super().__init__(settings=LLMSettings(model="test-model"), **kwargs)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame(f"You said {frame.context.messages[-1]['content']}"))
        await self.push_frame(LLMFullResponseEndFrame())


def response(*texts: str, speculation: bool, end: bool = True) -> list[Frame]:
    """Build the frames of one LLM response, as the LLM service marks them."""
    start = LLMFullResponseStartFrame()
    start.speculation = speculation
    frames: list[Frame] = [start, *(LLMTextFrame(t) for t in texts)]
    if end:
        frames.append(LLMFullResponseEndFrame())
    return frames


def spoken(frames) -> str:
    return " ".join(f.text for f in frames if isinstance(f, TTSTextFrame))


def index_of(frames, frame_type: type) -> int:
    return next(i for i, f in enumerate(frames) if isinstance(f, frame_type))


class TestTTSServiceHoldsTheSpeculation(unittest.IsolatedAsyncioTestCase):
    async def test_a_speculative_response_plays_once_its_turn_is_confirmed(self):
        down, _ = await run_test(
            WordTTS(enable_speculation_gate=True),
            frames_to_send=[
                UserStartedSpeakingFrame(),
                *response("Booking your flight.", speculation=True),
                SleepFrame(sleep=0.3),
                UserStoppedSpeakingFrame(),
                SleepFrame(),
            ],
        )

        # Synthesized while the turn was open, but none of it left the service
        # before the frame that confirmed it.
        assert index_of(down, LLMFullResponseStartFrame) > index_of(down, UserStoppedSpeakingFrame)
        assert spoken(down) == "Booking your flight."

    async def test_a_withdrawn_speculative_response_is_never_played(self):
        down, _ = await run_test(
            WordTTS(enable_speculation_gate=True),
            frames_to_send=[
                UserStartedSpeakingFrame(),
                *response("Cancelling your booking.", speculation=True),
                SleepFrame(sleep=0.3),
                EagerEndOfTurnCancelFrame(),
                SleepFrame(),
                UserStoppedSpeakingFrame(),
                *response("Rescheduling it.", speculation=False),
                SleepFrame(sleep=0.3),
            ],
        )

        assert spoken(down) == "Rescheduling it."
        assert len([f for f in down if isinstance(f, TTSStartedFrame)]) == 1

    async def test_a_withdrawal_drops_the_text_waiting_to_be_spoken(self):
        # The withdrawn response never finished its sentence, so its words are
        # still waiting in the aggregator for the next one.
        down, _ = await run_test(
            WordTTS(enable_speculation_gate=True),
            frames_to_send=[
                UserStartedSpeakingFrame(),
                *response("Cancelling your", speculation=True, end=False),
                SleepFrame(),
                EagerEndOfTurnCancelFrame(),
                SleepFrame(),
                UserStoppedSpeakingFrame(),
                *response("Rescheduling it.", speculation=False),
                SleepFrame(sleep=0.3),
            ],
        )

        assert spoken(down) == "Rescheduling it."

    async def test_an_ordinary_response_is_not_held(self):
        down, _ = await run_test(
            WordTTS(enable_speculation_gate=True),
            frames_to_send=[
                UserStartedSpeakingFrame(),
                *response("Booking your flight.", speculation=False),
                SleepFrame(sleep=0.3),
                UserStoppedSpeakingFrame(),
                SleepFrame(),
            ],
        )

        assert index_of(down, TTSStartedFrame) < index_of(down, UserStoppedSpeakingFrame)

    async def test_without_the_gate_a_speculative_response_is_not_held(self):
        down, _ = await run_test(
            WordTTS(),
            frames_to_send=[
                UserStartedSpeakingFrame(),
                *response("Booking your flight.", speculation=True),
                SleepFrame(sleep=0.3),
                UserStoppedSpeakingFrame(),
                SleepFrame(),
            ],
        )

        assert index_of(down, TTSStartedFrame) < index_of(down, UserStoppedSpeakingFrame)

    async def test_a_released_response_is_timed_from_its_release(self):
        # Its audio starts playing when it is released, so its words have to be
        # timed from then rather than from when the audio reached the gate.
        hold = 0.5
        down, _ = await run_test(
            WordTTS(enable_speculation_gate=True),
            frames_to_send=[
                UserStartedSpeakingFrame(),
                *response("Booking your flight.", speculation=True),
                SleepFrame(sleep=hold),
                UserStoppedSpeakingFrame(),
                SleepFrame(),
            ],
        )

        words = [f for f in down if isinstance(f, TTSTextFrame)]
        assert seconds_to_nanoseconds(hold) <= words[0].pts < seconds_to_nanoseconds(hold + 0.3)
        assert words[1].pts - words[0].pts == seconds_to_nanoseconds(0.1)
        end = down[index_of(down, LLMFullResponseEndFrame)]
        assert end.pts == words[-1].pts


class TestTTSServiceHoldsTheLLMResponse(unittest.IsolatedAsyncioTestCase):
    """The LLM service lets the speculative response through and the TTS service holds it."""

    @staticmethod
    def pipeline(context: LLMContext) -> Pipeline:
        aggregator = LLMUserAggregator(
            context,
            params=LLMUserAggregatorParams(user_turn_strategies=EagerUserTurnStrategies()),
        )
        return Pipeline(
            [
                aggregator,
                EchoLLM(enable_speculation_gate=False),
                WordTTS(enable_speculation_gate=True),
            ]
        )

    async def test_a_confirmed_speculation_is_spoken_after_the_turn_ends(self):
        context = LLMContext()

        down, _ = await run_test(
            self.pipeline(context),
            frames_to_send=[
                ProposedUserStartedSpeakingFrame(),
                SleepFrame(),
                EagerTranscriptionFrame("book a flight", "user", "t"),
                SleepFrame(sleep=0.3),
                TranscriptionFrame("book a flight", "user", "t"),
                SleepFrame(),
                ProposedUserStoppedSpeakingFrame(),
                SleepFrame(sleep=0.5),
            ],
        )

        assert spoken(down) == "You said book a flight"
        assert index_of(down, TTSStartedFrame) > index_of(down, UserStoppedSpeakingFrame)
        assert context.messages[0] == {"role": "user", "content": "book a flight"}

    async def test_a_mismatched_transcript_never_speaks_the_prediction(self):
        context = LLMContext()

        down, _ = await run_test(
            self.pipeline(context),
            frames_to_send=[
                ProposedUserStartedSpeakingFrame(),
                SleepFrame(),
                EagerTranscriptionFrame("i want to cancel", "user", "t"),
                SleepFrame(sleep=0.3),
                # The user carried on, so the committed transcript differs.
                TranscriptionFrame("i want to cancel, actually reschedule it", "user", "t"),
                SleepFrame(),
                ProposedUserStoppedSpeakingFrame(),
                SleepFrame(sleep=0.5),
            ],
        )

        assert spoken(down) == "You said i want to cancel, actually reschedule it"


if __name__ == "__main__":
    unittest.main()
