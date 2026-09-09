#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from loguru import logger

from pipecat.frames.frames import (
    EagerEndOfTurnCancelFrame,
    EagerEndOfTurnTranscriptionFrame,
    EndFrame,
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
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.turns.speculation_gate import GatedFrame, SpeculationGate, SpeculationState
from pipecat.turns.user_turn_strategies import EagerUserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager

DOWN = FrameDirection.DOWNSTREAM


def response(speculation_id: str | None, *texts: str, end: bool = True) -> list[Frame]:
    """Build the frames of one LLM response, stamped as the service would."""
    start = LLMFullResponseStartFrame()
    start.speculation_id = speculation_id
    frames: list[Frame] = [start, *(LLMTextFrame(t) for t in texts)]
    if end:
        stop = LLMFullResponseEndFrame()
        stop.speculation_id = speculation_id
        frames.append(stop)
    return frames


def tool_result(value: str = "booked") -> FunctionCallResultFrame:
    return FunctionCallResultFrame(
        function_name="book_flight",
        tool_call_id="call-1",
        arguments={},
        result=value,
    )


async def make_gate(**kwargs) -> SpeculationGate:
    """Build a gate wired up the way a host wires one."""
    gate = SpeculationGate(push_expired=kwargs.pop("push_expired", None) or _ignore, **kwargs)
    await gate.setup(TaskManager())
    return gate


async def _ignore(frames: list[GatedFrame]):
    pass


async def _collect(into: list, frames: list[GatedFrame]):
    into.append([frame for frame, _ in frames])


def emit(gate: SpeculationGate, *frames: Frame) -> list[Frame]:
    """Send frames through the gate, collecting everything it lets out."""
    emitted = []
    for frame in frames:
        emitted += [f for f, _ in gate.process(frame, DOWN)]
    return emitted


def types(frames: list[Frame]) -> list[type]:
    return [type(f) for f in frames]


class TestSpeculationGate(unittest.IsolatedAsyncioTestCase):
    async def test_non_speculative_response_passes_through(self):
        gate = await make_gate()

        assert types(emit(gate, *response(None, "Hello."))) == [
            LLMFullResponseStartFrame,
            LLMTextFrame,
            LLMFullResponseEndFrame,
        ]

    async def test_speculative_response_is_held_until_the_turn_ends(self):
        gate = await make_gate()

        assert emit(gate, *response("abc", "Booking ", "your flight.")) == []
        assert gate.state == SpeculationState.HOLDING

        # Confirmed: the whole response follows, in the order it was generated.
        released = emit(gate, UserStoppedSpeakingFrame(speculation_id="abc"))
        assert types(released) == [
            UserStoppedSpeakingFrame,
            LLMFullResponseStartFrame,
            LLMTextFrame,
            LLMTextFrame,
            LLMFullResponseEndFrame,
        ]
        assert [f.text for f in released if isinstance(f, LLMTextFrame)] == [
            "Booking ",
            "your flight.",
        ]
        assert gate.state == SpeculationState.OPEN

    async def test_withdrawn_speculation_is_discarded(self):
        gate = await make_gate()

        assert emit(gate, *response("abc", "Cancelling ", "your booking.", end=False)) == []
        assert types(emit(gate, EagerEndOfTurnCancelFrame(speculation_id="abc"))) == [
            EagerEndOfTurnCancelFrame
        ]

        # Straggling frames of the withdrawn response, still queued behind the
        # cancellation, which overtook them.
        assert emit(gate, LLMTextFrame(" Done.")) == []
        assert types(emit(gate, *response(None, "Rescheduling instead."))) == [
            LLMFullResponseStartFrame,
            LLMTextFrame,
            LLMFullResponseEndFrame,
        ]

    async def test_withdrawal_arriving_before_the_response_it_cancels(self):
        # A cancellation is a system frame, so it can overtake the response
        # frames it withdraws.
        gate = await make_gate()

        assert types(emit(gate, EagerEndOfTurnCancelFrame(speculation_id="abc"))) == [
            EagerEndOfTurnCancelFrame
        ]
        assert emit(gate, *response("abc", "Cancelling.")) == []
        assert types(emit(gate, *response(None, "Rescheduling instead."))) == [
            LLMFullResponseStartFrame,
            LLMTextFrame,
            LLMFullResponseEndFrame,
        ]

    async def test_withdrawal_leaves_a_different_speculation_alone(self):
        gate = await make_gate()

        emit(gate, *response("abc", "Booking."))
        assert types(emit(gate, EagerEndOfTurnCancelFrame(speculation_id="other"))) == [
            EagerEndOfTurnCancelFrame
        ]
        assert types(emit(gate, UserStoppedSpeakingFrame(speculation_id="abc"))) == [
            UserStoppedSpeakingFrame,
            LLMFullResponseStartFrame,
            LLMTextFrame,
            LLMFullResponseEndFrame,
        ]

    async def test_synthesized_audio_is_held_with_the_response(self):
        # A host that gates after synthesis holds the audio too.
        gate = await make_gate()
        audio = TTSAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1)

        assert emit(gate, *response("abc", "Hi.", end=False), audio) == []
        assert types(emit(gate, EagerEndOfTurnCancelFrame(speculation_id="abc"))) == [
            EagerEndOfTurnCancelFrame
        ]

    async def test_interruption_discards_the_speculation(self):
        gate = await make_gate()

        emit(gate, *response("abc", "Booking.", end=False))
        assert types(emit(gate, InterruptionFrame())) == [InterruptionFrame]

    async def test_upstream_frames_are_never_held(self):
        gate = await make_gate()

        emit(gate, *response("abc", "Booking.", end=False))
        upstream = LLMTextFrame("upstream")
        assert gate.process(upstream, FrameDirection.UPSTREAM) == [
            (upstream, FrameDirection.UPSTREAM)
        ]

    async def test_shutdown_delivers_what_has_to_outlive_the_speculation(self):
        # EndFrame is uninterruptible and awaited by the runner, so holding it
        # would hang shutdown.
        gate = await make_gate()

        emit(gate, *response("abc", "Booking.", end=False), tool_result())
        assert types(emit(gate, EndFrame())) == [FunctionCallResultFrame, EndFrame]

    async def test_a_hold_that_outlasts_its_bound_is_discarded(self):
        expired = []
        gate = await make_gate(max_hold_duration=0.05, push_expired=lambda f: _collect(expired, f))

        emit(gate, *response("abc", "Booking."))
        await asyncio.sleep(0.2)

        assert gate.state == SpeculationState.OPEN
        assert expired == [[]]
        # Confirmation arriving after the gate gave up releases nothing.
        assert types(emit(gate, UserStoppedSpeakingFrame(speculation_id="abc"))) == [
            UserStoppedSpeakingFrame
        ]

    async def test_a_hold_resolved_in_time_never_expires(self):
        expired = []
        gate = await make_gate(max_hold_duration=0.05, push_expired=lambda f: _collect(expired, f))

        emit(gate, *response("abc", "Booking."))
        emit(gate, UserStoppedSpeakingFrame(speculation_id="abc"))
        await asyncio.sleep(0.2)

        assert expired == []

    async def test_an_expired_hold_still_delivers_what_must_outlive_it(self):
        expired = []
        gate = await make_gate(max_hold_duration=0.05, push_expired=lambda f: _collect(expired, f))

        emit(gate, *response("abc", "Booking.", end=False), tool_result())
        await asyncio.sleep(0.2)

        assert [types(batch) for batch in expired] == [[FunctionCallResultFrame]]


class TestSpeculationHoldSignal(unittest.IsolatedAsyncioTestCase):
    """`speculation_id` is what a host arms its hold timer on."""

    async def test_it_names_the_held_speculation_only_while_holding(self):
        gate = await make_gate()
        assert gate.speculation_id is None

        emit(gate, *response("abc", "Booking."))
        assert gate.speculation_id == "abc"

        emit(gate, UserStoppedSpeakingFrame(speculation_id="abc"))
        assert gate.speculation_id is None

    async def test_a_superseding_speculation_renames_the_hold(self):
        # The timer has to start over rather than inherit the remainder of the
        # hold it supersedes.
        gate = await make_gate()

        emit(gate, *response("abc", "Booking.", end=False))
        assert gate.speculation_id == "abc"

        emit(gate, *response("def", "Rescheduling.", end=False))
        assert gate.speculation_id == "def"


class TestEagerMatchPolicies(unittest.IsolatedAsyncioTestCase):
    async def test_exact_match(self):
        from pipecat.turns.user_stop import ExactMatch

        policy = ExactMatch()
        assert policy.matches("book a flight", "book a flight")
        assert not policy.matches("book a flight", "Book a flight.")
        assert not policy.matches("book a flight", "book a flight tomorrow")

    async def test_normalized_match(self):
        from pipecat.turns.user_stop import NormalizedMatch

        policy = NormalizedMatch()
        assert policy.matches("book a flight", "Book a flight.")
        assert policy.matches("book  a flight", "book a flight")
        assert policy.matches("its ready", "It's ready!")
        assert not policy.matches("book a flight", "book a flight tomorrow")
        assert not policy.matches("i want to cancel", "I want to reschedule.")


class TestSpeculationGateOrdering(unittest.IsolatedAsyncioTestCase):
    async def test_turn_ending_without_confirming_the_speculation_releases_nothing(self):
        # The mismatch path ends the turn and withdraws the speculation, and the
        # two signals can arrive in either order.
        gate = await make_gate()

        emit(gate, *response("abc", "Cancelling.", end=False))
        assert types(emit(gate, UserStoppedSpeakingFrame())) == [UserStoppedSpeakingFrame]
        assert types(emit(gate, EagerEndOfTurnCancelFrame(speculation_id="abc"))) == [
            EagerEndOfTurnCancelFrame
        ]

    async def test_confirmation_arriving_before_the_response(self):
        gate = await make_gate()

        assert types(emit(gate, UserStoppedSpeakingFrame(speculation_id="abc"))) == [
            UserStoppedSpeakingFrame
        ]
        assert types(emit(gate, *response("abc", "Booking."))) == [
            LLMFullResponseStartFrame,
            LLMTextFrame,
            LLMFullResponseEndFrame,
        ]


class TestSupersededSpeculation(unittest.IsolatedAsyncioTestCase):
    async def test_a_new_response_supersedes_a_held_one(self):
        # A withdrawal that arrives before the response it voids needs no
        # memory: the response is held on arrival, and whatever answers the
        # turn instead supersedes it.
        gate = await make_gate()

        emit(gate, EagerEndOfTurnCancelFrame("abc"))
        assert emit(gate, *response("abc", "Cancelling.", end=False)) == []
        assert types(emit(gate, *response(None, "Rescheduling instead."))) == [
            LLMFullResponseStartFrame,
            LLMTextFrame,
            LLMFullResponseEndFrame,
        ]
        assert gate.state == SpeculationState.OPEN

    async def test_a_held_response_does_not_swallow_the_one_that_supersedes_it(self):
        # The held response never ends: its generation was cancelled mid-flight,
        # so no end frame is coming and only a new response resolves it. Its
        # frames are dropped, but the response that supersedes it has to pass
        # through whole — nothing of the held one can still be queued behind a
        # frame that arrived after it.
        gate = await make_gate()

        emit(gate, *response("abc", "Booking.", end=False))
        emitted = emit(gate, *response(None, "Something else entirely."))

        assert types(emitted) == [
            LLMFullResponseStartFrame,
            LLMTextFrame,
            LLMFullResponseEndFrame,
        ]
        assert [f.text for f in emitted if isinstance(f, LLMTextFrame)] == [
            "Something else entirely."
        ]


class TestUninterruptibleFrames(unittest.IsolatedAsyncioTestCase):
    async def test_a_tool_result_survives_a_discarded_speculation(self):
        # An async tool started in an earlier turn can return while a
        # speculation is held. Its result belongs to that earlier work and is
        # guaranteed delivery, so discarding the speculation around it keeps it.
        gate = await make_gate()

        emit(gate, *response("abc", "Booking.", end=False), tool_result())
        emitted = emit(gate, EagerEndOfTurnCancelFrame("abc"))

        assert types(emitted) == [EagerEndOfTurnCancelFrame, FunctionCallResultFrame]
        assert [f.result for f in emitted if isinstance(f, FunctionCallResultFrame)] == ["booked"]

    async def test_a_tool_result_is_held_in_order_with_the_response(self):
        # Uninterruptible frames are ordered like any other, so one that arrives
        # mid-response is released in the position it arrived in.
        gate = await make_gate()

        emit(gate, *response("abc", "Booking.", end=False), tool_result())
        assert types(emit(gate, UserStoppedSpeakingFrame(speculation_id="abc"))) == [
            UserStoppedSpeakingFrame,
            LLMFullResponseStartFrame,
            LLMTextFrame,
            FunctionCallResultFrame,
        ]

    async def test_a_tool_result_is_not_dropped_with_a_response_being_dropped(self):
        # Nothing is held back while dropping a withdrawn response's tail, so an
        # uninterruptible frame passes on in order rather than being dropped.
        gate = await make_gate()

        emit(gate, *response("abc", "Booking.", end=False))
        assert types(emit(gate, EagerEndOfTurnCancelFrame("abc"))) == [EagerEndOfTurnCancelFrame]
        # Still queued behind the withdrawal, which overtook it.
        assert types(emit(gate, LLMTextFrame(" Done."), tool_result())) == [FunctionCallResultFrame]


class TestResolvedResponsesCarryNoSpeculationId(unittest.IsolatedAsyncioTestCase):
    async def test_a_released_response_is_no_longer_marked_speculative(self):
        # Past the gate the response is confirmed, so the id is cleared: an id
        # downstream means nothing held the response back.
        gate = await make_gate()

        emit(gate, *response("abc", "Booking."))
        emitted = emit(gate, UserStoppedSpeakingFrame(speculation_id="abc"))

        assert [
            f.speculation_id
            for f in emitted
            if isinstance(f, (LLMFullResponseStartFrame, LLMFullResponseEndFrame))
        ] == [None, None]

    async def test_a_response_confirmed_before_it_arrives_is_unmarked_too(self):
        gate = await make_gate()

        emit(gate, UserStoppedSpeakingFrame(speculation_id="abc"))
        emitted = emit(gate, *response("abc", "Booking."))

        assert [
            f.speculation_id
            for f in emitted
            if isinstance(f, (LLMFullResponseStartFrame, LLMFullResponseEndFrame))
        ] == [None, None]


class TestUnheldSpeculationWarning(unittest.IsolatedAsyncioTestCase):
    async def test_the_output_transport_reports_a_speculation_nothing_held(self):
        # Speculating without anything holding the response speaks unconfirmed
        # replies, which is the outcome the feature exists to avoid, so it is
        # reported where it does the harm rather than failing silently.
        transport = BaseOutputTransport(TransportParams())
        transport._handle_frame = AsyncMock()

        start = LLMFullResponseStartFrame()
        start.speculation_id = "abc"

        with patch.object(logger, "error") as error:
            await transport.process_frame(start, DOWN)
            await transport.process_frame(start, DOWN)

        # Reported once: a line per turn would not tell anyone anything new.
        assert error.call_count == 1
        assert "speculative response reached" in str(error.call_args[0][0])
        assert transport._handle_frame.await_count == 2

    async def test_an_ordinary_response_is_not_reported(self):
        transport = BaseOutputTransport(TransportParams())
        transport._handle_frame = AsyncMock()

        with patch.object(logger, "error") as error:
            await transport.process_frame(LLMFullResponseStartFrame(), DOWN)

        assert error.call_count == 0


class GatedLLM(LLMService):
    """Answers every context frame, gating what it pushes as `LLMService` does."""

    def __init__(self, **kwargs):
        super().__init__(settings=LLMSettings(model="test-model"), **kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame("Booking your flight."))
        await self.push_frame(LLMFullResponseEndFrame())


class TestGatedPipeline(unittest.IsolatedAsyncioTestCase):
    async def test_a_gated_speculation_is_not_reported_to_the_output_transport(self):
        # The whole path, which is where a released response and an ungated one
        # have to look different: aggregator, gating LLM, output transport.
        context = LLMContext()
        aggregator = LLMUserAggregator(
            context,
            params=LLMUserAggregatorParams(user_turn_strategies=EagerUserTurnStrategies()),
        )
        transport = BaseOutputTransport(TransportParams())
        transport._handle_frame = AsyncMock()

        with patch.object(logger, "error") as error:
            await run_test(
                Pipeline([aggregator, GatedLLM(), transport]),
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

        reported = [str(call.args[0]) for call in error.call_args_list]
        assert not any("speculative response reached" in message for message in reported)
        assert context.messages == [{"role": "user", "content": "Book a flight."}]


if __name__ == "__main__":
    unittest.main()
