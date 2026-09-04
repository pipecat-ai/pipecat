#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from unittest.mock import AsyncMock, patch

from loguru import logger

from pipecat.frames.frames import (
    EagerEndOfTurnCancelFrame,
    EagerEndOfTurnTranscriptionFrame,
    Frame,
    FunctionCallResultFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.filters.user_turn_speculation_gate import (
    SpeculationState,
    UserTurnSpeculationGate,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.turns.user_turn_strategies import EagerUserTurnStrategies


def response(speculation_id: str | None, *texts: str, end: bool = True):
    """Build the frames of one LLM response, stamped as the service would."""
    start = LLMFullResponseStartFrame()
    start.speculation_id = speculation_id
    frames = [start, *(LLMTextFrame(t) for t in texts)]
    if end:
        stop = LLMFullResponseEndFrame()
        stop.speculation_id = speculation_id
        frames.append(stop)
    return frames


class TestUserTurnSpeculationGate(unittest.IsolatedAsyncioTestCase):
    async def test_non_speculative_response_passes_through(self):
        gate = UserTurnSpeculationGate()

        await run_test(
            gate,
            frames_to_send=response(None, "Hello."),
            expected_down_frames=[
                LLMFullResponseStartFrame,
                LLMTextFrame,
                LLMFullResponseEndFrame,
            ],
        )

    async def test_speculative_response_is_held_until_the_turn_ends(self):
        gate = UserTurnSpeculationGate()

        down, _ = await run_test(
            gate,
            frames_to_send=[
                *response("abc", "Booking ", "your flight."),
                SleepFrame(),
                UserStoppedSpeakingFrame(speculation_id="abc"),
            ],
            expected_down_frames=[
                # Nothing until the turn is confirmed, then the whole response
                # in the order it was generated.
                UserStoppedSpeakingFrame,
                LLMFullResponseStartFrame,
                LLMTextFrame,
                LLMTextFrame,
                LLMFullResponseEndFrame,
            ],
        )
        assert [f.text for f in down if isinstance(f, LLMTextFrame)] == [
            "Booking ",
            "your flight.",
        ]
        assert gate.state == SpeculationState.OPEN

    async def test_withdrawn_speculation_is_discarded(self):
        gate = UserTurnSpeculationGate()

        await run_test(
            gate,
            frames_to_send=[
                *response("abc", "Cancelling ", "your booking.", end=False),
                SleepFrame(),
                EagerEndOfTurnCancelFrame(speculation_id="abc"),
                SleepFrame(),
                # Straggling frames of the withdrawn response, still queued
                # behind the cancellation, which overtook them.
                LLMTextFrame(" Done."),
                *response(None, "Rescheduling instead."),
            ],
            expected_down_frames=[
                EagerEndOfTurnCancelFrame,
                LLMFullResponseStartFrame,
                LLMTextFrame,
                LLMFullResponseEndFrame,
            ],
        )

    async def test_withdrawal_arriving_before_the_response_it_cancels(self):
        # A cancellation is a system frame, so it can overtake the response
        # frames it withdraws.
        gate = UserTurnSpeculationGate()

        await run_test(
            gate,
            frames_to_send=[
                EagerEndOfTurnCancelFrame(speculation_id="abc"),
                SleepFrame(),
                *response("abc", "Cancelling."),
                *response(None, "Rescheduling instead."),
            ],
            expected_down_frames=[
                EagerEndOfTurnCancelFrame,
                LLMFullResponseStartFrame,
                LLMTextFrame,
                LLMFullResponseEndFrame,
            ],
        )

    async def test_withdrawal_leaves_a_different_speculation_alone(self):
        gate = UserTurnSpeculationGate()

        await run_test(
            gate,
            frames_to_send=[
                *response("abc", "Booking."),
                SleepFrame(),
                EagerEndOfTurnCancelFrame(speculation_id="other"),
                SleepFrame(),
                UserStoppedSpeakingFrame(speculation_id="abc"),
            ],
            expected_down_frames=[
                EagerEndOfTurnCancelFrame,
                UserStoppedSpeakingFrame,
                LLMFullResponseStartFrame,
                LLMTextFrame,
                LLMFullResponseEndFrame,
            ],
        )

    async def test_tts_audio_is_held_with_the_response(self):
        # The gate placed after the TTS service holds synthesized audio too.
        gate = UserTurnSpeculationGate()
        audio = TTSAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1)

        await run_test(
            gate,
            frames_to_send=[
                *response("abc", "Hi.", end=False),
                audio,
                SleepFrame(),
                EagerEndOfTurnCancelFrame(speculation_id="abc"),
            ],
            expected_down_frames=[EagerEndOfTurnCancelFrame],
        )

    async def test_interruption_discards_the_speculation(self):
        gate = UserTurnSpeculationGate()

        await run_test(
            gate,
            frames_to_send=[
                *response("abc", "Booking.", end=False),
                SleepFrame(),
                InterruptionFrame(),
            ],
            expected_down_frames=[InterruptionFrame],
        )

    async def test_unresolved_speculation_is_discarded_after_the_buffer_timeout(self):
        gate = UserTurnSpeculationGate(max_buffer_duration=0.2)

        await run_test(
            gate,
            frames_to_send=[
                *response("abc", "Booking."),
                SleepFrame(sleep=0.5),
                # Confirmation arrives after the gate gave up on it.
                UserStoppedSpeakingFrame(),
            ],
            expected_down_frames=[UserStoppedSpeakingFrame],
        )
        assert gate.state == SpeculationState.OPEN


class TestEagerMatchPolicies(unittest.IsolatedAsyncioTestCase):
    def test_exact_match(self):
        from pipecat.turns.user_stop import ExactMatch

        policy = ExactMatch()
        assert policy.matches("book a flight", "book a flight")
        assert not policy.matches("book a flight", "Book a flight.")
        assert not policy.matches("book a flight", "book a flight tomorrow")

    def test_normalized_match(self):
        from pipecat.turns.user_stop import NormalizedMatch

        policy = NormalizedMatch()
        assert policy.matches("book a flight", "Book a flight.")
        assert policy.matches("book  a flight", "book a flight")
        assert policy.matches("its ready", "It's ready!")
        assert not policy.matches("book a flight", "book a flight tomorrow")
        assert not policy.matches("i want to cancel", "I want to reschedule.")


class TestUserTurnSpeculationGateOrdering(unittest.IsolatedAsyncioTestCase):
    async def test_turn_ending_without_confirming_the_speculation_releases_nothing(self):
        # The mismatch path ends the turn and withdraws the speculation, and the
        # two signals can arrive in either order.
        gate = UserTurnSpeculationGate()

        await run_test(
            gate,
            frames_to_send=[
                *response("abc", "Cancelling.", end=False),
                SleepFrame(),
                UserStoppedSpeakingFrame(),
                SleepFrame(),
                EagerEndOfTurnCancelFrame(speculation_id="abc"),
            ],
            expected_down_frames=[UserStoppedSpeakingFrame, EagerEndOfTurnCancelFrame],
        )

    async def test_confirmation_arriving_before_the_response(self):
        gate = UserTurnSpeculationGate()

        await run_test(
            gate,
            frames_to_send=[
                UserStoppedSpeakingFrame(speculation_id="abc"),
                SleepFrame(),
                *response("abc", "Booking."),
            ],
            expected_down_frames=[
                UserStoppedSpeakingFrame,
                LLMFullResponseStartFrame,
                LLMTextFrame,
                LLMFullResponseEndFrame,
            ],
        )


if __name__ == "__main__":
    unittest.main()


class TestSupersededSpeculation(unittest.IsolatedAsyncioTestCase):
    async def test_a_new_response_supersedes_a_held_one(self):
        # A withdrawal that arrives before the response it voids needs no
        # memory: the response is held on arrival, and whatever answers the
        # turn instead supersedes it.
        gate = UserTurnSpeculationGate()

        await run_test(
            gate,
            frames_to_send=[
                # Withdrawn before its response reached us.
                EagerEndOfTurnCancelFrame("abc"),
                SleepFrame(),
                *response("abc", "Cancelling.", end=False),
                SleepFrame(),
                *response(None, "Rescheduling instead."),
            ],
            expected_down_frames=[
                EagerEndOfTurnCancelFrame,
                LLMFullResponseStartFrame,
                LLMTextFrame,
                LLMFullResponseEndFrame,
            ],
        )
        assert gate.state == SpeculationState.OPEN

    async def test_a_held_response_does_not_swallow_the_one_that_supersedes_it(self):
        # The held response never ends: its generation was cancelled mid-flight,
        # so no end frame is coming and only a new response resolves it. Its
        # frames are dropped, but the response that supersedes it has to pass
        # through whole — nothing of the held one can still be queued behind a
        # frame that arrived after it.
        gate = UserTurnSpeculationGate()

        down, _ = await run_test(
            gate,
            frames_to_send=[
                *response("abc", "Booking.", end=False),
                SleepFrame(),
                *response(None, "Something else entirely."),
            ],
            expected_down_frames=[
                LLMFullResponseStartFrame,
                LLMTextFrame,
                LLMFullResponseEndFrame,
            ],
        )
        assert [f.text for f in down if isinstance(f, LLMTextFrame)] == ["Something else entirely."]


def tool_result(value: str = "booked") -> FunctionCallResultFrame:
    return FunctionCallResultFrame(
        function_name="book_flight",
        tool_call_id="call-1",
        arguments={},
        result=value,
    )


class TestUninterruptibleFrames(unittest.IsolatedAsyncioTestCase):
    async def test_a_tool_result_survives_a_discarded_speculation(self):
        # An async tool started in an earlier turn can return while a
        # speculation is held. Its result belongs to that earlier work and is
        # guaranteed delivery, so discarding the speculation around it keeps it.
        gate = UserTurnSpeculationGate()

        down, _ = await run_test(
            gate,
            frames_to_send=[
                *response("abc", "Booking.", end=False),
                tool_result(),
                SleepFrame(),
                EagerEndOfTurnCancelFrame("abc"),
            ],
            expected_down_frames=[
                EagerEndOfTurnCancelFrame,
                FunctionCallResultFrame,
            ],
        )
        assert [f.result for f in down if isinstance(f, FunctionCallResultFrame)] == ["booked"]

    async def test_a_tool_result_is_held_in_order_with_the_response(self):
        # Uninterruptible frames are ordered like any other, so one that arrives
        # mid-response is released in the position it arrived in.
        gate = UserTurnSpeculationGate()

        down, _ = await run_test(
            gate,
            frames_to_send=[
                *response("abc", "Booking.", end=False),
                tool_result(),
                SleepFrame(),
                UserStoppedSpeakingFrame(speculation_id="abc"),
            ],
            expected_down_frames=[
                UserStoppedSpeakingFrame,
                LLMFullResponseStartFrame,
                LLMTextFrame,
                FunctionCallResultFrame,
            ],
        )

    async def test_a_tool_result_is_not_dropped_with_a_response_being_dropped(self):
        # Nothing is held back while dropping a withdrawn response's tail, so an
        # uninterruptible frame passes on in order rather than being dropped.
        gate = UserTurnSpeculationGate()

        await run_test(
            gate,
            frames_to_send=[
                *response("abc", "Booking.", end=False),
                SleepFrame(),
                EagerEndOfTurnCancelFrame("abc"),
                SleepFrame(),
                # Still queued behind the withdrawal, which overtook it.
                LLMTextFrame(" Done."),
                tool_result(),
            ],
            expected_down_frames=[
                EagerEndOfTurnCancelFrame,
                FunctionCallResultFrame,
            ],
        )


class TestUnheldSpeculationWarning(unittest.IsolatedAsyncioTestCase):
    async def test_the_output_transport_reports_a_speculation_nothing_held(self):
        # Speculating without a gate speaks unconfirmed responses, which is the
        # outcome the feature exists to avoid, so it is reported where it does
        # the harm rather than failing silently.
        transport = BaseOutputTransport(TransportParams())
        transport._handle_frame = AsyncMock()

        start = LLMFullResponseStartFrame()
        start.speculation_id = "abc"

        with patch.object(logger, "error") as error:
            await transport.process_frame(start, FrameDirection.DOWNSTREAM)
            await transport.process_frame(start, FrameDirection.DOWNSTREAM)

        # Reported once: a line per turn would not tell anyone anything new.
        assert error.call_count == 1
        assert "UserTurnSpeculationGate" in error.call_args[0][0]
        assert transport._handle_frame.await_count == 2

    async def test_an_ordinary_response_is_not_reported(self):
        transport = BaseOutputTransport(TransportParams())
        transport._handle_frame = AsyncMock()

        with patch.object(logger, "error") as error:
            await transport.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)

        assert error.call_count == 0


class TestResolvedResponsesCarryNoSpeculationId(unittest.IsolatedAsyncioTestCase):
    async def test_a_released_response_is_no_longer_marked_speculative(self):
        # Past the gate the response is confirmed, so the id is cleared: an id
        # downstream means nothing held the response back.
        gate = UserTurnSpeculationGate()

        down, _ = await run_test(
            gate,
            frames_to_send=[
                *response("abc", "Booking."),
                SleepFrame(),
                UserStoppedSpeakingFrame(speculation_id="abc"),
            ],
            expected_down_frames=[
                UserStoppedSpeakingFrame,
                LLMFullResponseStartFrame,
                LLMTextFrame,
                LLMFullResponseEndFrame,
            ],
        )

        assert [
            f.speculation_id
            for f in down
            if isinstance(f, (LLMFullResponseStartFrame, LLMFullResponseEndFrame))
        ] == [None, None]

    async def test_a_response_confirmed_before_it_arrives_is_unmarked_too(self):
        gate = UserTurnSpeculationGate()

        down, _ = await run_test(
            gate,
            frames_to_send=[
                UserStoppedSpeakingFrame(speculation_id="abc"),
                SleepFrame(),
                *response("abc", "Booking."),
            ],
            expected_down_frames=[
                UserStoppedSpeakingFrame,
                LLMFullResponseStartFrame,
                LLMTextFrame,
                LLMFullResponseEndFrame,
            ],
        )

        assert [
            f.speculation_id
            for f in down
            if isinstance(f, (LLMFullResponseStartFrame, LLMFullResponseEndFrame))
        ] == [None, None]


class GatedLLM(FrameProcessor):
    """Answers every context frame, stamping ids the way `LLMService` does."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        start = LLMFullResponseStartFrame()
        start.speculation_id = frame.speculation_id
        await self.push_frame(start)
        await self.push_frame(LLMTextFrame("Booking your flight."))
        end = LLMFullResponseEndFrame()
        end.speculation_id = frame.speculation_id
        await self.push_frame(end)


class TestGatedPipeline(unittest.IsolatedAsyncioTestCase):
    async def test_a_gated_speculation_is_not_reported_to_the_output_transport(self):
        # The whole path, which is where a released response and an ungated one
        # have to look different: aggregator, LLM, gate, output transport.
        context = LLMContext()
        aggregator = LLMUserAggregator(
            context,
            params=LLMUserAggregatorParams(user_turn_strategies=EagerUserTurnStrategies()),
        )
        transport = BaseOutputTransport(TransportParams())
        transport._handle_frame = AsyncMock()

        with patch.object(logger, "error") as error:
            await run_test(
                Pipeline([aggregator, GatedLLM(), UserTurnSpeculationGate(), transport]),
                frames_to_send=[
                    ProposedUserStartedSpeakingFrame(),
                    SleepFrame(),
                    EagerEndOfTurnTranscriptionFrame("book a flight", "user", "t", "abc"),
                    SleepFrame(),
                    TranscriptionFrame("Book a flight.", "user", "t"),
                    SleepFrame(),
                    ProposedUserStoppedSpeakingFrame(),
                    SleepFrame(sleep=1.0),
                ],
            )

        assert error.call_count == 0
        assert context.messages == [{"role": "user", "content": "Book a flight."}]
