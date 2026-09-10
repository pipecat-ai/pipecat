#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.frames.frames import (
    EagerEndOfTurnCancelFrame,
    EagerTranscriptionFrame,
    FunctionCallFromLLM,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
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
from pipecat.utils.asyncio.task_manager import TaskManager

from .frame_processor_helpers import frame_processor_setup


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


def eager(text: str) -> EagerTranscriptionFrame:
    return EagerTranscriptionFrame(text, "user", "2026-09-03T00:00:00Z")


async def strategy_alone(**kwargs) -> EagerUserTurnStopStrategy:
    """A strategy set up outside a pipeline, since it runs a task of its own."""
    strategy = EagerUserTurnStopStrategy(**kwargs)
    await strategy.setup(frame_processor_setup(TaskManager()))
    return strategy


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
        assert provisional.speculative
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
        assert contexts[0].speculative
        # The turn end is what releases its response.
        assert len(stops) == 1
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
        assert [c.speculative for c in contexts] == [True, False]
        # The withdrawal precedes the turn end, so the gate has nothing to release.
        assert down.index(cancel) < next(
            i for i, f in enumerate(down) if isinstance(f, UserStoppedSpeakingFrame)
        )
        assert contexts[1].context.messages[-1] == {
            "role": "user",
            "content": "i want to cancel, actually reschedule it",
        }

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
                EagerEndOfTurnCancelFrame(),
                SleepFrame(),
                final("i think i'll book it tomorrow"),
                SleepFrame(),
                ProposedUserStoppedSpeakingFrame(),
                SleepFrame(sleep=1.0),
            ],
        )

        contexts = [f for f in down if isinstance(f, LLMContextFrame)]
        cancels = [f for f in down if isinstance(f, EagerEndOfTurnCancelFrame)]

        # The service's withdrawal travels on its own. The strategy adds none of
        # its own.
        assert len(cancels) == 1
        assert [c.speculative for c in contexts] == [True, False]
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
        assert not contexts[0].speculative
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

        assert not any(isinstance(f, EagerTranscriptionFrame) for f in down)


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
        speculative = LLMContextFrame(context=context, speculative=True)

        down, up = await run_test(llm, frames_to_send=[speculative, SleepFrame()])

        assert calls == []
        # Broadcast both ways: downstream to the gate, upstream to the strategy.
        withdrawals = [f for f in [*down, *up] if isinstance(f, EagerEndOfTurnCancelFrame)]
        assert len(withdrawals) == 2

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


class TestUnresolvedSpeculation(unittest.IsolatedAsyncioTestCase):
    async def test_new_turn_withdraws_a_speculation_left_in_flight(self):
        # A fresh turn starting over a live prediction has to withdraw it, or
        # the gate would hold the response for good: no turn end withdraws it,
        # since the previous turn never ended.
        pushed = []

        strategy = await strategy_alone()
        strategy.add_event_handler(
            "on_push_frame", lambda s, frame, direction: pushed.append(frame)
        )

        await strategy.process_frame(eager("book a flight"))
        await strategy.handle_user_turn_started()

        assert len([f for f in pushed if isinstance(f, EagerEndOfTurnCancelFrame)]) == 1

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
        assert [c.speculative for c in contexts] == [True]
        assert len(cancels) == 1
        # The eager transcript is never written: only a committed one is.
        assert context.messages == []


class TestDeferredEagerStrategy(unittest.IsolatedAsyncioTestCase):
    async def test_deferring_carries_the_speculation_to_the_subscriber(self):
        # The wrapper is transparent apart from the event it suppresses, so the
        # speculation reaches the subscriber on the event that starts it.
        triggered = []

        inner = EagerUserTurnStopStrategy()
        wrapper = deferred(inner)
        await wrapper.setup(frame_processor_setup(TaskManager()))
        wrapper.add_event_handler(
            "on_user_turn_inference_triggered",
            lambda strategy, speculation: triggered.append(speculation),
        )

        await wrapper.process_frame(eager("book a flight"))

        assert [s.text for s in triggered] == ["book a flight"]


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
        assert len(cancels) == 1
        assert context.messages == [{"role": "user", "content": "Book a flight to Tokyo."}]


class EchoingStreamLLM(LLMService):
    """LLM service that answers every context frame slowly, naming the user message it answers."""

    def __init__(self, *, stream_for: float = 0.3, **kwargs):
        super().__init__(settings=LLMSettings(model="test-model"), **kwargs)
        self._stream_for = stream_for

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        user = frame.context.messages[-1]["content"]
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame(f"Answer to: {user}"))
        await asyncio.sleep(self._stream_for)
        await self.push_frame(LLMFullResponseEndFrame())


class TestMismatchEndToEnd(unittest.IsolatedAsyncioTestCase):
    """A prediction that misses is withdrawn ahead of the turn end, and the turn is still answered."""

    async def test_only_the_answer_to_the_committed_transcript_is_spoken(self):
        context = LLMContext()

        down, _ = await run_test(
            Pipeline([aggregator(context), EchoingStreamLLM()]),
            frames_to_send=[
                ProposedUserStartedSpeakingFrame(),
                SleepFrame(),
                eager("i want to cancel"),
                # The speculative response is still streaming when the commit lands.
                SleepFrame(sleep=0.1),
                final("i want to cancel, actually reschedule it"),
                SleepFrame(),
                ProposedUserStoppedSpeakingFrame(),
                SleepFrame(sleep=1.5),
            ],
        )

        # The withdrawal reaches the LLM before the turn end, so the held
        # speculative response is discarded rather than released.
        cancel_at = next(i for i, f in enumerate(down) if isinstance(f, EagerEndOfTurnCancelFrame))
        stop_at = next(i for i, f in enumerate(down) if isinstance(f, UserStoppedSpeakingFrame))
        assert cancel_at < stop_at
        assert [f.text for f in down if isinstance(f, LLMTextFrame)] == [
            "Answer to: i want to cancel, actually reschedule it"
        ]
        assert context.messages == [
            {"role": "user", "content": "i want to cancel, actually reschedule it"}
        ]

    async def test_a_prediction_that_holds_is_spoken_once(self):
        context = LLMContext()

        down, _ = await run_test(
            Pipeline([aggregator(context), EchoingStreamLLM()]),
            frames_to_send=[
                ProposedUserStartedSpeakingFrame(),
                SleepFrame(),
                eager("book a flight"),
                SleepFrame(sleep=0.1),
                final("Book a flight."),
                SleepFrame(),
                ProposedUserStoppedSpeakingFrame(),
                SleepFrame(sleep=1.5),
            ],
        )

        assert [f.text for f in down if isinstance(f, LLMTextFrame)] == ["Answer to: book a flight"]
        assert not any(isinstance(f, EagerEndOfTurnCancelFrame) for f in down)
        assert context.messages == [{"role": "user", "content": "Book a flight."}]


class SpeculativeLLM(LLMService):
    """LLM service that answers every context frame with one text response."""

    def __init__(self, **kwargs):
        super().__init__(settings=LLMSettings(model="test-model"), **kwargs)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame("Booking your flight."))
        await self.push_frame(LLMFullResponseEndFrame())


class TestLLMServiceHoldsTheSpeculation(unittest.IsolatedAsyncioTestCase):
    """The LLM service holds a speculative response until its turn is confirmed."""

    @staticmethod
    def response_frames(frames):
        """The response frames among everything the service pushed."""
        return [
            f
            for f in frames
            if isinstance(f, (LLMFullResponseStartFrame, LLMTextFrame, LLMFullResponseEndFrame))
        ]

    @staticmethod
    def context_frame(speculative=False):
        context = LLMContext(messages=[{"role": "user", "content": "book a flight"}])
        return LLMContextFrame(context=context, speculative=speculative)

    async def test_a_speculative_response_leaves_the_service_only_once_the_turn_ends(self):
        down, _ = await run_test(
            SpeculativeLLM(),
            frames_to_send=[
                self.context_frame(True),
                SleepFrame(),
                UserStoppedSpeakingFrame(),
                SleepFrame(),
            ],
        )

        # The whole response, and only after the frame that confirmed it.
        assert [type(f) for f in self.response_frames(down)] == [
            LLMFullResponseStartFrame,
            LLMTextFrame,
            LLMFullResponseEndFrame,
        ]
        confirmed_at = next(
            i for i, f in enumerate(down) if isinstance(f, UserStoppedSpeakingFrame)
        )
        response_at = next(
            i for i, f in enumerate(down) if isinstance(f, LLMFullResponseStartFrame)
        )
        assert response_at > confirmed_at

    async def test_a_withdrawn_speculative_response_never_leaves_the_service(self):
        down, _ = await run_test(
            SpeculativeLLM(),
            frames_to_send=[
                self.context_frame(True),
                SleepFrame(),
                EagerEndOfTurnCancelFrame(),
                SleepFrame(),
            ],
        )

        assert self.response_frames(down) == []

    async def test_an_ordinary_response_is_not_held(self):
        down, _ = await run_test(
            SpeculativeLLM(),
            frames_to_send=[self.context_frame(), SleepFrame()],
        )

        assert [type(f) for f in self.response_frames(down)] == [
            LLMFullResponseStartFrame,
            LLMTextFrame,
            LLMFullResponseEndFrame,
        ]


class TestSpeculationTimeout(unittest.IsolatedAsyncioTestCase):
    """The strategy withdraws a speculation the service neither commits nor withdraws."""

    async def test_an_unresolved_speculation_is_withdrawn_after_the_timeout(self):
        pushed = []

        strategy = await strategy_alone(speculation_timeout=0.1)
        strategy.add_event_handler(
            "on_push_frame", lambda s, frame, direction: pushed.append(frame)
        )

        await strategy.process_frame(eager("book a flight"))
        await asyncio.sleep(0.4)

        assert [type(f) for f in pushed] == [EagerEndOfTurnCancelFrame]

    async def test_a_speculation_resolved_in_time_is_not_withdrawn(self):
        pushed = []

        strategy = await strategy_alone(speculation_timeout=0.1)
        strategy.add_event_handler(
            "on_push_frame", lambda s, frame, direction: pushed.append(frame)
        )

        await strategy.process_frame(eager("book a flight"))
        # The service withdrew it itself, so the timer has nothing left to do.
        await strategy.process_frame(EagerEndOfTurnCancelFrame())
        await asyncio.sleep(0.4)

        assert pushed == []


class TestATurnOutlastingTheHoldIsStillAnswered(unittest.IsolatedAsyncioTestCase):
    """A speculation that times out must not cost the turn its reply.

    The bound exists so the bot is never left silent; withdrawing the held
    response has to put the turn back on the ordinary path.
    """

    @staticmethod
    def aggregator_and_llm(speculation_timeout: float):
        context = LLMContext()
        return (
            context,
            aggregator(context, speculation_timeout=speculation_timeout),
            SpeculativeLLM(),
        )

    async def test_a_matching_commit_after_the_hold_expires_runs_a_fresh_inference(self):
        context, user_aggregator, llm = self.aggregator_and_llm(0.2)

        down, _ = await run_test(
            Pipeline([user_aggregator, llm]),
            frames_to_send=[
                ProposedUserStartedSpeakingFrame(),
                SleepFrame(),
                eager("book a flight"),
                # Longer than the timeout, so the strategy withdraws it before the commit.
                SleepFrame(sleep=0.8),
                final("Book a flight."),
                SleepFrame(),
                ProposedUserStoppedSpeakingFrame(),
                SleepFrame(sleep=0.8),
            ],
        )

        # The turn is answered, and the context records the committed
        # transcript once rather than an unanswered user message.
        assert [f.text for f in down if isinstance(f, LLMTextFrame)] == ["Booking your flight."]
        assert context.messages == [{"role": "user", "content": "Book a flight."}]

    async def test_a_hold_that_does_not_expire_is_released_as_before(self):
        context, user_aggregator, llm = self.aggregator_and_llm(5.0)

        down, _ = await run_test(
            Pipeline([user_aggregator, llm]),
            frames_to_send=[
                ProposedUserStartedSpeakingFrame(),
                SleepFrame(),
                eager("book a flight"),
                SleepFrame(),
                final("Book a flight."),
                SleepFrame(),
                ProposedUserStoppedSpeakingFrame(),
                SleepFrame(sleep=0.8),
            ],
        )

        assert [f.text for f in down if isinstance(f, LLMTextFrame)] == ["Booking your flight."]


class StreamingToolCallLLM(LLMService):
    """LLM service whose completion streams for a while before calling a tool.

    Models the window the feature exists to exploit: the committed end of turn
    can land while the response is still being generated, before the call the
    response ends up making.
    """

    def __init__(self, *, stream_for: float, **kwargs):
        super().__init__(settings=LLMSettings(model="test-model"), **kwargs)
        self._stream_for = stream_for
        self.calls: list[str] = []
        self.register_function("book_flight", lambda params: self.calls.append(params.tool_call_id))

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame("Let me check that."))
        # The turn can be confirmed anywhere in here.
        await asyncio.sleep(self._stream_for)
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
        await self.push_frame(LLMFullResponseEndFrame())


class TestToolCallsAcrossTheConfirmation(unittest.IsolatedAsyncioTestCase):
    """A tool call is speculative only until the turn it answers is confirmed."""

    @staticmethod
    def context_frame(speculative=False):
        context = LLMContext(messages=[{"role": "user", "content": "book a flight"}])
        return LLMContextFrame(context=context, speculative=speculative)

    @staticmethod
    def withdrawals(down, up):
        return [f for f in [*down, *up] if isinstance(f, EagerEndOfTurnCancelFrame)]

    async def test_a_turn_confirmed_mid_stream_lets_the_call_run(self):
        # The committed end of turn arrives while the response is still
        # generating, so by the time it reaches its tool call the response is
        # confirmed and the call is an ordinary one.
        llm = StreamingToolCallLLM(stream_for=0.4)

        down, up = await run_test(
            llm,
            frames_to_send=[
                self.context_frame(True),
                SleepFrame(sleep=0.15),
                UserStoppedSpeakingFrame(),
                SleepFrame(sleep=1.0),
            ],
        )

        assert llm.calls == ["call-1"]
        # The released response survives: nothing withdraws it, which is what
        # used to leave the bot speaking a preamble and then falling silent.
        assert [f.text for f in down if isinstance(f, LLMTextFrame)] == ["Let me check that."]
        assert self.withdrawals(down, up) == []

    async def test_a_turn_end_that_overtakes_the_inference_lets_the_call_run(self):
        # The turn end is a system frame, so it can pass the context frame that
        # starts the inference it confirms.
        llm = StreamingToolCallLLM(stream_for=0.1)

        down, up = await run_test(
            llm,
            frames_to_send=[
                UserStoppedSpeakingFrame(),
                SleepFrame(),
                self.context_frame(True),
                SleepFrame(sleep=0.5),
            ],
        )

        assert llm.calls == ["call-1"]
        assert self.withdrawals(down, up) == []

    async def test_a_call_reached_before_the_turn_ends_is_withdrawn(self):
        # Still speculative: the tool must not run, and the withdrawal reaches
        # the strategy so it re-runs on the committed transcript.
        llm = StreamingToolCallLLM(stream_for=0.05)

        down, up = await run_test(
            llm,
            frames_to_send=[self.context_frame(True), SleepFrame(sleep=0.8)],
        )

        assert llm.calls == []
        assert len(self.withdrawals(down, up)) == 2

    async def test_an_ordinary_inference_calls_its_tool(self):
        llm = StreamingToolCallLLM(stream_for=0.05)

        down, up = await run_test(
            llm,
            frames_to_send=[self.context_frame(), SleepFrame(sleep=0.8)],
        )

        assert llm.calls == ["call-1"]
        assert self.withdrawals(down, up) == []


if __name__ == "__main__":
    unittest.main()
