#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from unittest.mock import AsyncMock

from pipecat.frames.frames import (
    EagerEndOfTurnCancelFrame,
    EagerTranscriptionFrame,
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
    UserStartedSpeakingFrame,
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
from pipecat.turns.speculation_gate import SpeculationGate, SpeculationState
from pipecat.turns.user_turn_strategies import EagerUserTurnStrategies

DOWN = FrameDirection.DOWNSTREAM


def response(*texts: str, end: bool = True) -> list[Frame]:
    """Build the frames of one LLM response."""
    frames: list[Frame] = [LLMFullResponseStartFrame(), *(LLMTextFrame(t) for t in texts)]
    if end:
        frames.append(LLMFullResponseEndFrame())
    return frames


def tool_result(value: str = "booked") -> FunctionCallResultFrame:
    return FunctionCallResultFrame(
        function_name="book_flight",
        tool_call_id="call-1",
        arguments={},
        result=value,
    )


def speculate(
    gate: SpeculationGate, speculative: bool, *texts: str, end: bool = True, then=()
) -> list[Frame]:
    """Run an inference through the gate the way a host does.

    The gate is told whether the inference is speculative before its frames
    arrive, which is what decides whether the response is held.
    """
    gate.begin_speculation(speculative)
    return emit(gate, *response(*texts, end=end), *then)


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
        gate = SpeculationGate()

        assert types(speculate(gate, False, "Hello.")) == [
            LLMFullResponseStartFrame,
            LLMTextFrame,
            LLMFullResponseEndFrame,
        ]

    async def test_speculative_response_is_held_until_the_turn_ends(self):
        gate = SpeculationGate()

        assert speculate(gate, True, "Booking ", "your flight.") == []
        assert gate.state == SpeculationState.HOLDING

        # The turn ended: the whole response follows, in the order it was generated.
        released = emit(gate, UserStoppedSpeakingFrame())
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
        gate = SpeculationGate()

        assert speculate(gate, True, "Cancelling ", "your booking.", end=False) == []
        assert types(emit(gate, EagerEndOfTurnCancelFrame())) == [EagerEndOfTurnCancelFrame]

        # Straggling frames of the withdrawn response, still queued behind the
        # cancellation, which overtook them.
        assert emit(gate, LLMTextFrame(" Done.")) == []
        assert types(speculate(gate, False, "Rescheduling instead.")) == [
            LLMFullResponseStartFrame,
            LLMTextFrame,
            LLMFullResponseEndFrame,
        ]

    async def test_withdrawal_arriving_before_the_response_it_cancels(self):
        # A cancellation is a system frame, so it can overtake the response
        # frames it withdraws.
        gate = SpeculationGate()

        assert types(emit(gate, EagerEndOfTurnCancelFrame())) == [EagerEndOfTurnCancelFrame]
        assert speculate(gate, True, "Cancelling.") == []
        assert types(speculate(gate, False, "Rescheduling instead.")) == [
            LLMFullResponseStartFrame,
            LLMTextFrame,
            LLMFullResponseEndFrame,
        ]

    async def test_synthesized_audio_is_held_with_the_response(self):
        # A host that gates after synthesis holds the audio too.
        gate = SpeculationGate()
        audio = TTSAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1)

        assert speculate(gate, True, "Hi.", end=False, then=(audio,)) == []
        assert types(emit(gate, EagerEndOfTurnCancelFrame())) == [EagerEndOfTurnCancelFrame]

    async def test_interruption_discards_the_speculation(self):
        gate = SpeculationGate()

        speculate(gate, True, "Booking.", end=False)
        assert types(emit(gate, InterruptionFrame())) == [InterruptionFrame]

    async def test_upstream_frames_are_never_held(self):
        gate = SpeculationGate()

        speculate(gate, True, "Booking.", end=False)
        upstream = LLMTextFrame("upstream")
        assert gate.process(upstream, FrameDirection.UPSTREAM) == [
            (upstream, FrameDirection.UPSTREAM)
        ]

    async def test_shutdown_delivers_what_has_to_outlive_the_speculation(self):
        # EndFrame is uninterruptible and awaited by the runner, so holding it
        # would hang shutdown.
        gate = SpeculationGate()

        speculate(gate, True, "Booking.", end=False, then=(tool_result(),))
        assert types(emit(gate, EndFrame())) == [FunctionCallResultFrame, EndFrame]


class TestSpeculating(unittest.IsolatedAsyncioTestCase):
    """What the gate reports about the inference in flight.

    A host asks this rather than tracking it alongside, so it has to hold up
    however the turn and the inference are ordered.
    """

    async def test_an_ordinary_inference_is_never_speculating(self):
        gate = SpeculationGate()
        gate.begin_speculation(False)

        assert not gate.speculating

    async def test_it_lasts_until_the_turn_ends(self):
        gate = SpeculationGate()
        assert not gate.speculating

        speculate(gate, True, "Booking.", end=False)
        assert gate.speculating

        emit(gate, UserStoppedSpeakingFrame())
        assert not gate.speculating

    async def test_a_turn_ending_mid_inference_clears_it_before_the_response_ends(self):
        # The window the feature exists to exploit: the turn ends while the
        # inference is still generating, so what it does next is committed.
        gate = SpeculationGate()

        gate.begin_speculation(True)
        emit(gate, response("Let me check.", end=False)[0])
        emit(gate, UserStoppedSpeakingFrame())

        assert not gate.speculating

    async def test_a_turn_that_ended_before_the_inference_is_never_speculating(self):
        # The turn end is a system frame and can pass the context frame that
        # starts the inference it confirms.
        gate = SpeculationGate()

        emit(gate, UserStoppedSpeakingFrame())
        gate.begin_speculation(True)

        assert not gate.speculating
        # And nothing is held back, since the turn it answers is already over.
        assert types(emit(gate, *response("Booking."))) == [
            LLMFullResponseStartFrame,
            LLMTextFrame,
            LLMFullResponseEndFrame,
        ]

    async def test_a_new_turn_makes_a_speculative_inference_speculative_again(self):
        gate = SpeculationGate()

        emit(gate, UserStoppedSpeakingFrame())
        emit(gate, UserStartedSpeakingFrame())
        assert speculate(gate, True, "Booking.") == []
        assert gate.speculating
        assert gate.state == SpeculationState.HOLDING

    async def test_a_withdrawal_clears_it(self):
        gate = SpeculationGate()

        speculate(gate, True, "Booking.", end=False)
        emit(gate, EagerEndOfTurnCancelFrame())

        assert not gate.speculating

    async def test_an_interruption_clears_it(self):
        gate = SpeculationGate()

        speculate(gate, True, "Booking.", end=False)
        emit(gate, InterruptionFrame())

        assert not gate.speculating

    async def test_a_superseding_inference_keeps_its_own(self):
        # Dropping the held response ends that hold, but the inference that
        # replaced it is speculative and must stay so.
        gate = SpeculationGate()

        speculate(gate, True, "Booking.", end=False)
        speculate(gate, True, "Rescheduling.", end=False)

        assert gate.speculating


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
    async def test_a_withdrawal_then_the_turn_end_releases_nothing(self):
        # The mismatch path withdraws the speculation and then ends the turn.
        # Both are system frames, so they arrive in that order.
        gate = SpeculationGate()

        speculate(gate, True, "Cancelling.", end=False)
        assert types(emit(gate, EagerEndOfTurnCancelFrame())) == [EagerEndOfTurnCancelFrame]
        assert types(emit(gate, UserStoppedSpeakingFrame())) == [UserStoppedSpeakingFrame]

    async def test_the_turn_end_arriving_before_the_response(self):
        gate = SpeculationGate()

        assert types(emit(gate, UserStoppedSpeakingFrame())) == [UserStoppedSpeakingFrame]
        assert types(speculate(gate, True, "Booking.")) == [
            LLMFullResponseStartFrame,
            LLMTextFrame,
            LLMFullResponseEndFrame,
        ]


class TestSupersededSpeculation(unittest.IsolatedAsyncioTestCase):
    async def test_a_new_response_supersedes_a_held_one(self):
        # A withdrawal that arrives before the response it voids needs no
        # memory: the response is held on arrival, and whatever answers the
        # turn instead supersedes it.
        gate = SpeculationGate()

        emit(gate, EagerEndOfTurnCancelFrame())
        assert speculate(gate, True, "Cancelling.", end=False) == []
        assert types(speculate(gate, False, "Rescheduling instead.")) == [
            LLMFullResponseStartFrame,
            LLMTextFrame,
            LLMFullResponseEndFrame,
        ]
        assert gate.state == SpeculationState.OPEN

    async def test_a_held_response_does_not_swallow_the_one_that_supersedes_it(self):
        # The held response never ends: its generation was cancelled mid-flight,
        # so no end frame is coming and only a new response resolves it. Its
        # frames are dropped, but the response that supersedes it has to pass
        # through whole.
        gate = SpeculationGate()

        speculate(gate, True, "Booking.", end=False)
        emitted = speculate(gate, False, "Something else entirely.")

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
        gate = SpeculationGate()

        speculate(gate, True, "Booking.", end=False, then=(tool_result(),))
        emitted = emit(gate, EagerEndOfTurnCancelFrame())

        assert types(emitted) == [EagerEndOfTurnCancelFrame, FunctionCallResultFrame]
        assert [f.result for f in emitted if isinstance(f, FunctionCallResultFrame)] == ["booked"]

    async def test_a_tool_result_is_held_in_order_with_the_response(self):
        # Uninterruptible frames are ordered like any other, so one that arrives
        # mid-response is released in the position it arrived in.
        gate = SpeculationGate()

        speculate(gate, True, "Booking.", end=False, then=(tool_result(),))
        assert types(emit(gate, UserStoppedSpeakingFrame())) == [
            UserStoppedSpeakingFrame,
            LLMFullResponseStartFrame,
            LLMTextFrame,
            FunctionCallResultFrame,
        ]

    async def test_a_tool_result_is_not_dropped_with_a_response_being_dropped(self):
        # Nothing is held back while dropping a withdrawn response's tail, so an
        # uninterruptible frame passes on in order rather than being dropped.
        gate = SpeculationGate()

        speculate(gate, True, "Booking.", end=False)
        assert types(emit(gate, EagerEndOfTurnCancelFrame())) == [EagerEndOfTurnCancelFrame]
        # Still queued behind the withdrawal, which overtook it.
        assert types(emit(gate, LLMTextFrame(" Done."), tool_result())) == [FunctionCallResultFrame]


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
    async def test_the_transport_sees_a_response_only_once_the_turn_ends(self):
        # The whole path: aggregator, gating LLM, output transport. Nothing the
        # speculation produced reaches the transport before the turn ends, and
        # the context records the committed transcript rather than the eager one.
        context = LLMContext()
        aggregator = LLMUserAggregator(
            context,
            params=LLMUserAggregatorParams(user_turn_strategies=EagerUserTurnStrategies()),
        )
        transport = BaseOutputTransport(TransportParams())
        transport._handle_frame = AsyncMock()

        await run_test(
            Pipeline([aggregator, GatedLLM(), transport]),
            frames_to_send=[
                ProposedUserStartedSpeakingFrame(),
                SleepFrame(),
                EagerTranscriptionFrame("book a flight", "user", "t"),
                SleepFrame(),
                TranscriptionFrame("Book a flight.", "user", "t"),
                SleepFrame(),
                ProposedUserStoppedSpeakingFrame(),
                SleepFrame(sleep=1.0),
            ],
        )

        spoken = [call.args[0] for call in transport._handle_frame.await_args_list]
        assert [f.text for f in spoken if isinstance(f, LLMTextFrame)] == ["Booking your flight."]
        assert context.messages == [{"role": "user", "content": "Book a flight."}]


if __name__ == "__main__":
    unittest.main()
