#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import (
    EagerEndOfTurnCancelFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.filters.speculative_response_gate import (
    SpeculationState,
    SpeculativeResponseGate,
)
from pipecat.tests.utils import SleepFrame, run_test


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


class TestSpeculativeResponseGate(unittest.IsolatedAsyncioTestCase):
    async def test_non_speculative_response_passes_through(self):
        gate = SpeculativeResponseGate()

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
        gate = SpeculativeResponseGate()

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
        gate = SpeculativeResponseGate()

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
        gate = SpeculativeResponseGate()

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
        gate = SpeculativeResponseGate()

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
        gate = SpeculativeResponseGate()
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
        gate = SpeculativeResponseGate()

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
        gate = SpeculativeResponseGate(max_buffer_duration=0.2)

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


class TestSpeculativeResponseGateOrdering(unittest.IsolatedAsyncioTestCase):
    async def test_turn_ending_without_confirming_the_speculation_releases_nothing(self):
        # The mismatch path ends the turn and withdraws the speculation, and the
        # two signals can arrive in either order.
        gate = SpeculativeResponseGate()

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
        gate = SpeculativeResponseGate()

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
        gate = SpeculativeResponseGate()

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
        gate = SpeculativeResponseGate()

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
