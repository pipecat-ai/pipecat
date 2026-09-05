#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import (
    EagerEndOfTurnCancelFrame,
    EagerEndOfTurnTranscriptionFrame,
    FunctionCallFromLLM,
    InterruptionFrame,
    LLMContextFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
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
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_stop import EagerUserTurnStopStrategy, ExactMatch, deferred
from pipecat.turns.user_turn_strategies import EagerUserTurnStrategies


def aggregator(
    context: LLMContext, *, user_turn_stop_timeout: float = 5.0, **kwargs
) -> LLMUserAggregator:
    return LLMUserAggregator(
        context,
        params=LLMUserAggregatorParams(
            user_turn_strategies=EagerUserTurnStrategies(**kwargs),
            user_turn_stop_timeout=user_turn_stop_timeout,
        ),
    )


def eager(text: str, speculation_id: str = "abc") -> EagerEndOfTurnTranscriptionFrame:
    return EagerEndOfTurnTranscriptionFrame(text, "user", "2026-09-03T00:00:00Z", speculation_id)


def final(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text, "user", "2026-09-03T00:00:00Z")


class TestEagerUserTurnStrategies(unittest.IsolatedAsyncioTestCase):
    async def test_eager_end_of_turn_runs_inference_without_touching_the_context(self):
        context = LLMContext()
        context.set_messages([{"role": "system", "content": "Be brief."}])

        down, _ = await run_test(
            aggregator(context),
            frames_to_send=[
                ProposedUserStartedSpeakingFrame(),
                SleepFrame(),
                eager("book a flight"),
                SleepFrame(),
            ],
        )

        # Inference ran against a provisional copy carrying the eager transcript.
        provisional = next(f for f in down if isinstance(f, LLMContextFrame))
        assert provisional.speculation_id is not None
        assert provisional.context is not context
        assert provisional.context.messages[-1] == {"role": "user", "content": "book a flight"}

        # The real context is untouched: the turn hasn't ended.
        assert context.messages == [{"role": "system", "content": "Be brief."}]

    async def test_matching_transcript_ends_the_turn_and_keeps_the_response(self):
        context = LLMContext()

        down, _ = await run_test(
            aggregator(context),
            frames_to_send=[
                ProposedUserStartedSpeakingFrame(),
                SleepFrame(),
                eager("book a flight"),
                SleepFrame(),
                final("book a flight"),
                SleepFrame(),
                ProposedUserStoppedSpeakingFrame(),
                SleepFrame(sleep=1.0),
            ],
        )

        contexts = [f for f in down if isinstance(f, LLMContextFrame)]
        stops = [f for f in down if isinstance(f, UserStoppedSpeakingFrame)]

        assert not any(isinstance(f, EagerEndOfTurnCancelFrame) for f in down)
        # One inference, the speculative one: the confirmed turn is written to
        # the context without answering it a second time.
        assert len(contexts) == 1
        assert contexts[0].speculation_id is not None
        # The turn end names the speculation, which is what releases its response.
        assert [f.speculation_id for f in stops] == [contexts[0].speculation_id]
        assert context.messages == [{"role": "user", "content": "book a flight"}]

    async def test_differing_transcript_withdraws_the_speculation(self):
        context = LLMContext()

        down, _ = await run_test(
            aggregator(context),
            frames_to_send=[
                ProposedUserStartedSpeakingFrame(),
                SleepFrame(),
                eager("i want to cancel"),
                SleepFrame(),
                final("i want to cancel, actually reschedule it"),
                SleepFrame(),
                ProposedUserStoppedSpeakingFrame(),
                SleepFrame(sleep=1.0),
            ],
        )

        contexts = [f for f in down if isinstance(f, LLMContextFrame)]
        cancel = next(f for f in down if isinstance(f, EagerEndOfTurnCancelFrame))

        # The speculative inference, then a second one on the committed transcript.
        assert [c.speculation_id is not None for c in contexts] == [True, False]
        assert cancel.speculation_id == contexts[0].speculation_id
        assert contexts[1].context.messages[-1] == {
            "role": "user",
            "content": "i want to cancel, actually reschedule it",
        }
        # The turn end confirms nothing, so it releases nothing.
        assert all(
            f.speculation_id is None for f in down if isinstance(f, UserStoppedSpeakingFrame)
        )

        # Only the committed transcript reaches the context.
        assert context.messages == [
            {"role": "user", "content": "i want to cancel, actually reschedule it"}
        ]

    async def test_resuming_the_turn_withdraws_the_speculation(self):
        context = LLMContext()

        down, _ = await run_test(
            aggregator(context),
            frames_to_send=[
                ProposedUserStartedSpeakingFrame(),
                SleepFrame(),
                eager("i think"),
                SleepFrame(),
                # The service withdraws the prediction it made.
                EagerEndOfTurnCancelFrame("abc"),
                SleepFrame(),
                final("i think i'll book it tomorrow"),
                SleepFrame(),
                ProposedUserStoppedSpeakingFrame(),
                SleepFrame(sleep=1.0),
            ],
        )

        contexts = [f for f in down if isinstance(f, LLMContextFrame)]
        cancels = [f for f in down if isinstance(f, EagerEndOfTurnCancelFrame)]

        # The service's withdrawal travels on its own, naming the prediction the
        # speculative inference was run from. The strategy adds none of its own.
        assert [c.speculation_id for c in cancels] == ["abc"]
        assert contexts[0].speculation_id == "abc"
        assert contexts[1].speculation_id is None
        assert context.messages == [{"role": "user", "content": "i think i'll book it tomorrow"}]

    async def test_formatting_differences_are_tolerated_by_default(self):
        context = LLMContext()

        down, _ = await run_test(
            aggregator(context),
            frames_to_send=[
                ProposedUserStartedSpeakingFrame(),
                SleepFrame(),
                eager("book a flight to tokyo"),
                SleepFrame(),
                # The service formats the transcript it commits; the response
                # still answers the same turn.
                final("Book a flight to Tokyo."),
                SleepFrame(),
                ProposedUserStoppedSpeakingFrame(),
                SleepFrame(sleep=1.0),
            ],
        )

        contexts = [f for f in down if isinstance(f, LLMContextFrame)]

        assert not any(isinstance(f, EagerEndOfTurnCancelFrame) for f in down)
        assert len(contexts) == 1
        # The eager transcript drove the response; the committed one is what the
        # context records.
        assert contexts[0].context.messages[-1] == {
            "role": "user",
            "content": "book a flight to tokyo",
        }
        assert context.messages == [{"role": "user", "content": "Book a flight to Tokyo."}]

    async def test_turn_without_an_eager_prediction_behaves_normally(self):
        context = LLMContext()

        down, _ = await run_test(
            aggregator(context),
            frames_to_send=[
                ProposedUserStartedSpeakingFrame(),
                SleepFrame(),
                final("hello there"),
                SleepFrame(),
                ProposedUserStoppedSpeakingFrame(),
                SleepFrame(sleep=1.0),
            ],
        )

        contexts = [f for f in down if isinstance(f, LLMContextFrame)]

        assert not any(isinstance(f, EagerEndOfTurnCancelFrame) for f in down)
        assert len(contexts) == 1
        assert contexts[0].speculation_id is None
        assert all(
            f.speculation_id is None for f in down if isinstance(f, UserStoppedSpeakingFrame)
        )
        assert context.messages == [{"role": "user", "content": "hello there"}]

    async def test_eager_transcript_is_not_pushed_downstream(self):
        # It is a TextFrame, so anything downstream that speaks or aggregates
        # text must never see it.
        context = LLMContext()

        down, _ = await run_test(
            Pipeline([aggregator(context)]),
            frames_to_send=[
                ProposedUserStartedSpeakingFrame(),
                SleepFrame(),
                eager("book a flight"),
                SleepFrame(),
            ],
        )

        assert not any(isinstance(f, EagerEndOfTurnTranscriptionFrame) for f in down)


class SpeculativeToolCallLLM(LLMService):
    """LLM service that answers every context frame with a tool call."""

    def __init__(self, **kwargs):
        super().__init__(settings=LLMSettings(model="test-model"), **kwargs)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            await self.run_function_calls(
                [
                    FunctionCallFromLLM(
                        function_name="book_flight",
                        tool_call_id="call-1",
                        arguments={},
                        context=frame.context,
                    )
                ]
            )
        else:
            await self.push_frame(frame, direction)


class TestSpeculativeToolCalls(unittest.IsolatedAsyncioTestCase):
    async def test_speculative_inference_does_not_execute_tools(self):
        # Tools run inside the service, so a discarded speculation could not
        # undo them. They must not run until the turn is committed.
        calls = []

        llm = SpeculativeToolCallLLM()
        llm.register_function("book_flight", lambda params: calls.append(params))

        context = LLMContext(messages=[{"role": "user", "content": "book a flight"}])
        speculative = LLMContextFrame(context=context, speculation_id="abc")

        down, up = await run_test(llm, frames_to_send=[speculative, SleepFrame()])

        assert calls == []
        withdrawals = [f for f in [*down, *up] if isinstance(f, EagerEndOfTurnCancelFrame)]
        assert [f.speculation_id for f in withdrawals] == ["abc", "abc"]

    async def test_committed_inference_executes_tools(self):
        calls = []

        llm = SpeculativeToolCallLLM()
        llm.register_function("book_flight", lambda params: calls.append(params))

        context = LLMContext(messages=[{"role": "user", "content": "book a flight"}])

        down, up = await run_test(
            llm,
            frames_to_send=[LLMContextFrame(context=context), SleepFrame(sleep=0.5)],
        )

        assert len(calls) == 1
        assert not any(isinstance(f, EagerEndOfTurnCancelFrame) for f in [*down, *up])


if __name__ == "__main__":
    unittest.main()


class TestUnresolvedSpeculation(unittest.IsolatedAsyncioTestCase):
    async def test_new_turn_withdraws_a_speculation_left_in_flight(self):
        # A turn boundary the service didn't resolve — a fresh turn starting
        # over a live prediction — still has to withdraw it, or the gate would
        # hold the response until its buffer times out.
        pushed = []

        strategy = EagerUserTurnStopStrategy()
        strategy.add_event_handler(
            "on_push_frame", lambda s, frame, direction: pushed.append(frame)
        )

        await strategy.process_frame(eager("book a flight"))
        await strategy.handle_user_turn_started()

        assert [f.speculation_id for f in pushed if isinstance(f, EagerEndOfTurnCancelFrame)] == [
            "abc"
        ]

        # The withdrawal happens once: a second boundary has nothing left to
        # withdraw.
        await strategy.handle_user_turn_stopped()
        assert len([f for f in pushed if isinstance(f, EagerEndOfTurnCancelFrame)]) == 1


class TestTurnCommittedWithoutATranscript(unittest.IsolatedAsyncioTestCase):
    async def test_speculation_is_withdrawn_when_there_is_nothing_to_compare(self):
        # A service can commit an end of turn without a transcript — Flux drops
        # one below `min_confidence`, Cartesia sends `turn.end` with an empty
        # transcript for a turn that captured only noise. The eager prediction
        # had a transcript, but there is nothing to check it against, so the
        # response it produced is discarded rather than spoken.
        context = LLMContext()

        down, _ = await run_test(
            aggregator(context, user_turn_stop_timeout=0.3),
            frames_to_send=[
                ProposedUserStartedSpeakingFrame(),
                SleepFrame(),
                eager("book a flight"),
                SleepFrame(),
                SleepFrame(),
                ProposedUserStoppedSpeakingFrame(),
                SleepFrame(sleep=0.8),
            ],
        )

        contexts = [f for f in down if isinstance(f, LLMContextFrame)]
        cancels = [f for f in down if isinstance(f, EagerEndOfTurnCancelFrame)]

        # Only the speculative inference ran, and it was withdrawn.
        assert [c.speculation_id for c in contexts] == ["abc"]
        assert [c.speculation_id for c in cancels] == ["abc"]
        # The turn end confirms nothing, so the gate would release nothing even
        # if it saw the turn end before the withdrawal.
        assert all(
            f.speculation_id is None for f in down if isinstance(f, UserStoppedSpeakingFrame)
        )
        # The eager transcript is never written: only a committed one is.
        assert context.messages == []


class TestDeferredEagerStrategy(unittest.IsolatedAsyncioTestCase):
    async def test_deferring_carries_the_speculation_to_the_subscriber(self):
        # The wrapper is transparent apart from the event it suppresses, so the
        # speculation reaches the subscriber on the event that starts it.
        triggered = []

        inner = EagerUserTurnStopStrategy()
        wrapper = deferred(inner)
        wrapper.add_event_handler(
            "on_user_turn_inference_triggered",
            lambda strategy, speculation: triggered.append(speculation),
        )

        await wrapper.process_frame(eager("book a flight"))

        assert [s.text for s in triggered] == ["book a flight"]
        assert triggered[0].id == "abc"


class TestExactMatchPolicy(unittest.IsolatedAsyncioTestCase):
    async def test_formatting_differences_withdraw_the_speculation(self):
        # Opting into ExactMatch requires the committed transcript to be
        # identical, so a service that formats what it commits discards the
        # response it had already generated.
        context = LLMContext()

        down, _ = await run_test(
            aggregator(context, match_policy=ExactMatch()),
            frames_to_send=[
                ProposedUserStartedSpeakingFrame(),
                SleepFrame(),
                eager("book a flight to tokyo"),
                SleepFrame(),
                final("Book a flight to Tokyo."),
                SleepFrame(),
                ProposedUserStoppedSpeakingFrame(),
                SleepFrame(sleep=1.0),
            ],
        )

        cancels = [f for f in down if isinstance(f, EagerEndOfTurnCancelFrame)]
        assert [c.speculation_id for c in cancels] == ["abc"]
        assert context.messages == [{"role": "user", "content": "Book a flight to Tokyo."}]
