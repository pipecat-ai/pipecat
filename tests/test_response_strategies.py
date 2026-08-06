#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
import unittest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallResultProperties,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    ResponseFrame,
    TextFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMAssistantAggregatorParams,
)
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.response import (
    AnnouncementConfig,
    AnnouncementStyle,
    CompletedToolResult,
    DelayedResponseStrategy,
)


def make_aggregator(context: LLMContext, settle_secs: float = 0.1) -> LLMAssistantAggregator:
    return LLMAssistantAggregator(
        context,
        params=LLMAssistantAggregatorParams(
            response_strategy=DelayedResponseStrategy(settle_secs=settle_secs)
        ),
    )


def append_response(text: str, run_llm: bool = True) -> ResponseFrame:
    return ResponseFrame(
        frame=LLMMessagesAppendFrame(
            messages=[{"role": "developer", "content": text}],
            run_llm=run_llm,
        )
    )


class TestNoStrategy(unittest.IsolatedAsyncioTestCase):
    async def test_response_frame_released_immediately_with_warning(self):
        context = LLMContext()
        aggregator = LLMAssistantAggregator(context)

        released = []

        @aggregator.event_handler("on_response_released")
        async def on_response_released(aggregator, frames):
            released.append(frames)

        await run_test(
            aggregator,
            frames_to_send=[append_response("task finished")],
            expected_up_frames=[LLMContextFrame],
        )
        self.assertEqual(context.messages[-1]["content"], "task finished")
        self.assertEqual(len(released), 1)

    async def test_response_frame_released_even_while_bot_speaking(self):
        # Without a strategy there is no deferral machinery at all: release
        # is immediate regardless of activity.
        context = LLMContext()
        aggregator = LLMAssistantAggregator(context)

        await run_test(
            aggregator,
            frames_to_send=[
                BotStartedSpeakingFrame(),
                append_response("task finished"),
            ],
            expected_up_frames=[LLMContextFrame],
        )
        self.assertEqual(context.messages[-1]["content"], "task finished")


class TestDelayedResponseStrategy(unittest.IsolatedAsyncioTestCase):
    async def test_release_when_idle_after_settle(self):
        # No conversational activity observed at all (the text-mode/evals
        # case): every wait condition is open, so the response releases as
        # soon as its settle window elapses.
        context = LLMContext()
        aggregator = make_aggregator(context)

        await run_test(
            aggregator,
            frames_to_send=[append_response("task finished"), SleepFrame(0.3)],
            expected_up_frames=[LLMContextFrame],
        )
        self.assertEqual(context.messages[-1]["content"], "task finished")

    async def test_deferred_while_bot_speaking_released_after_stop(self):
        context = LLMContext()
        aggregator = make_aggregator(context)

        deferred = []
        released = []

        @aggregator.event_handler("on_response_deferred")
        async def on_response_deferred(aggregator, frame):
            deferred.append(frame)

        @aggregator.event_handler("on_response_released")
        async def on_response_released(aggregator, frames):
            released.append(frames)

        response = append_response("task finished")
        await run_test(
            aggregator,
            frames_to_send=[
                BotStartedSpeakingFrame(),
                response,
                SleepFrame(0.1),
                BotStoppedSpeakingFrame(),
                SleepFrame(0.4),
            ],
            expected_up_frames=[LLMContextFrame],
        )
        self.assertEqual(context.messages[-1]["content"], "task finished")
        self.assertEqual(deferred, [response])
        self.assertEqual(released, [[response]])

    async def test_deferred_while_user_speaking(self):
        context = LLMContext()
        aggregator = make_aggregator(context)

        await run_test(
            aggregator,
            frames_to_send=[
                UserStartedSpeakingFrame(),
                append_response("task finished"),
                SleepFrame(0.1),
                UserStoppedSpeakingFrame(),
                SleepFrame(0.4),
            ],
            expected_up_frames=[LLMContextFrame],
        )
        self.assertEqual(context.messages[-1]["content"], "task finished")

    async def test_deferred_while_answer_owed(self):
        # An LLM response is streaming: the assistant-initiated response must
        # not jump in ahead of the answer the user is waiting for.
        context = LLMContext()
        aggregator = make_aggregator(context)

        await run_test(
            aggregator,
            frames_to_send=[
                LLMFullResponseStartFrame(),
                append_response("task finished"),
                SleepFrame(0.1),
                LLMFullResponseEndFrame(),
                SleepFrame(0.4),
            ],
            expected_up_frames=[LLMContextFrame],
        )
        self.assertEqual(context.messages[-1]["content"], "task finished")

    async def test_async_function_call_does_not_hold_response(self):
        # A long-running async tool (cancel_on_interruption=False) is the
        # typical *producer* of assistant-initiated responses — an unrelated
        # response must not be held until it finishes.
        context = LLMContext()
        aggregator = make_aggregator(context)

        await run_test(
            aggregator,
            frames_to_send=[
                FunctionCallInProgressFrame(
                    function_name="slow_research",
                    tool_call_id="1",
                    arguments={},
                    cancel_on_interruption=False,
                ),
                SleepFrame(0.1),
                # The async call never completes within the test: the
                # response must release anyway.
                append_response("task finished"),
                SleepFrame(0.3),
            ],
            expected_up_frames=[LLMContextFrame],
        )
        self.assertEqual(context.messages[-1]["content"], "task finished")

    async def test_reactive_function_call_holds_response(self):
        # A reactive call (cancel_on_interruption=True) means the user is
        # waiting on an answer: hold the assistant-initiated response until
        # the result lands.
        context = LLMContext()
        strategy = DelayedResponseStrategy(settle_secs=0.1)
        aggregator = LLMAssistantAggregator(
            context,
            params=LLMAssistantAggregatorParams(response_strategy=strategy),
        )

        released = []
        tool_content_at_release = []

        @aggregator.event_handler("on_response_released")
        async def on_response_released(aggregator, frames):
            released.append(frames)

        # The strategy's release event is synchronous, so this captures the
        # tool message's content at the actual moment of release: if the
        # response were wrongly released before the result landed, this
        # would still read "IN_PROGRESS".
        @strategy.event_handler("on_response_released")
        async def on_strategy_released(strategy, items):
            tool_messages = [m for m in context.messages if m.get("role") == "tool"]
            tool_content_at_release.append(tool_messages[-1]["content"])

        await run_test(
            aggregator,
            frames_to_send=[
                FunctionCallInProgressFrame(
                    function_name="get_weather",
                    tool_call_id="1",
                    arguments={},
                    cancel_on_interruption=True,
                ),
                append_response("task finished"),
                SleepFrame(0.1),
                FunctionCallResultFrame(
                    function_name="get_weather",
                    tool_call_id="1",
                    arguments={},
                    result={"conditions": "Sunny"},
                    # Keep the result from triggering its own inference so the
                    # single expected LLMContextFrame is the release's.
                    properties=FunctionCallResultProperties(run_llm=False),
                ),
                SleepFrame(0.4),
            ],
            expected_up_frames=[LLMContextFrame],
        )
        self.assertEqual(context.messages[-1]["content"], "task finished")
        self.assertEqual(len(released), 1)
        self.assertEqual(json.loads(tool_content_at_release[0]), {"conditions": "Sunny"})

    async def test_batching_merges_appends_into_single_run(self):
        context = LLMContext()
        aggregator = make_aggregator(context)

        released = []

        @aggregator.event_handler("on_response_released")
        async def on_response_released(aggregator, frames):
            released.append(frames)

        await run_test(
            aggregator,
            frames_to_send=[
                BotStartedSpeakingFrame(),
                append_response("first result"),
                append_response("second result"),
                SleepFrame(0.1),
                BotStoppedSpeakingFrame(),
                SleepFrame(0.4),
            ],
            # Two pending responses, ONE inference trigger.
            expected_up_frames=[LLMContextFrame],
        )
        # FIFO order preserved in context.
        self.assertEqual(context.messages[-2]["content"], "first result")
        self.assertEqual(context.messages[-1]["content"], "second result")
        # Released together as one batch.
        self.assertEqual(len(released), 1)
        self.assertEqual(len(released[0]), 2)

    async def test_settle_window_restarted_by_activity(self):
        # Bot stops, but the user starts speaking inside the settle window:
        # the batch must hold and release only after the user is done.
        context = LLMContext()
        aggregator = make_aggregator(context, settle_secs=0.3)

        await run_test(
            aggregator,
            frames_to_send=[
                BotStartedSpeakingFrame(),
                append_response("task finished"),
                SleepFrame(0.1),
                BotStoppedSpeakingFrame(),
                # User barges into the settle window.
                SleepFrame(0.1),
                UserStartedSpeakingFrame(),
                SleepFrame(0.4),  # longer than settle — must NOT release while speaking
                UserStoppedSpeakingFrame(),
                SleepFrame(0.6),
            ],
            expected_up_frames=[LLMContextFrame],
        )
        self.assertEqual(context.messages[-1]["content"], "task finished")

    async def test_interruption_does_not_discard_pending_response(self):
        context = LLMContext()
        aggregator = make_aggregator(context)

        await run_test(
            aggregator,
            frames_to_send=[
                BotStartedSpeakingFrame(),
                append_response("task finished"),
                SleepFrame(0.1),
                InterruptionFrame(),
                BotStoppedSpeakingFrame(),
                SleepFrame(0.4),
            ],
            expected_up_frames=[LLMContextFrame],
        )
        self.assertEqual(context.messages[-1]["content"], "task finished")

    async def test_tts_speak_payload_released_upstream(self):
        context = LLMContext()
        aggregator = make_aggregator(context)

        await run_test(
            aggregator,
            frames_to_send=[
                ResponseFrame(frame=TTSSpeakFrame("Your results are ready.")),
                SleepFrame(0.3),
            ],
            # Verbatim speech: no inference trigger, just the speak frame
            # heading back upstream toward the TTS service.
            expected_up_frames=[TTSSpeakFrame],
        )

    async def test_other_payload_pushed_upstream_with_warning(self):
        # Payloads without special handling are not dropped: they replay
        # upstream as-is (with a warning), reaching whatever processor
        # handles them.
        context = LLMContext()
        aggregator = make_aggregator(context)

        await run_test(
            aggregator,
            frames_to_send=[
                ResponseFrame(frame=TextFrame("no special handling")),
                SleepFrame(0.3),
            ],
            expected_up_frames=[TextFrame],
        )

    async def test_pending_responses_dropped_at_shutdown(self):
        context = LLMContext()
        aggregator = make_aggregator(context)

        released = []

        @aggregator.event_handler("on_response_released")
        async def on_response_released(aggregator, frames):
            released.append(frames)

        # The bot never stops speaking, so the response is still pending when
        # run_test sends the EndFrame — it must be dropped, not blurted out
        # mid-teardown.
        await run_test(
            aggregator,
            frames_to_send=[
                BotStartedSpeakingFrame(),
                append_response("task finished"),
                SleepFrame(0.1),
            ],
            expected_up_frames=[],
        )
        self.assertEqual(released, [])
        # Never released: the message never reached the context.
        self.assertEqual(len(context.messages), 0)

    async def test_announced_call_stall_resolves_on_in_progress(self):
        # FunctionCallsStartedFrame seeds the in-progress map with None,
        # which counts as reactive (pending) until the call's
        # FunctionCallInProgressFrame reveals it's async. That reveal must
        # push a fresh activity snapshot, or a deferred response would stall
        # for the async tool's entire duration.
        context = LLMContext()
        aggregator = make_aggregator(context)

        await run_test(
            aggregator,
            frames_to_send=[
                FunctionCallsStartedFrame(
                    function_calls=[
                        FunctionCallFromLLM(
                            function_name="slow_research",
                            tool_call_id="1",
                            arguments={},
                            context=context,
                        )
                    ]
                ),
                # Deferred: the announced call is still None in the map.
                append_response("task finished"),
                SleepFrame(0.1),
                # The call reveals itself async: response_pending flips False
                # and the pending response must release (after settle).
                FunctionCallInProgressFrame(
                    function_name="slow_research",
                    tool_call_id="1",
                    arguments={},
                    cancel_on_interruption=False,
                ),
                SleepFrame(0.4),
            ],
            expected_up_frames=[LLMContextFrame],
        )
        self.assertEqual(context.messages[-1]["content"], "task finished")

    async def test_no_release_in_inference_request_window(self):
        # A deferred post-function-result push requests an inference, but
        # LLMFullResponseStartFrame only arrives one time-to-first-token
        # later. A pending response must not release inside that window —
        # that would push a second context frame and double-trigger
        # inference. Here the LLM never responds, so the response must stay
        # held (and be dropped at shutdown): exactly ONE context frame.
        context = LLMContext()
        aggregator = make_aggregator(context)

        released = []

        @aggregator.event_handler("on_response_released")
        async def on_response_released(aggregator, items):
            released.append(items)

        await run_test(
            aggregator,
            frames_to_send=[
                FunctionCallInProgressFrame(
                    function_name="get_weather",
                    tool_call_id="1",
                    arguments={},
                    cancel_on_interruption=True,
                ),
                BotStartedSpeakingFrame(),
                SleepFrame(0.1),
                # Reactive result while the bot speaks: its inference push is
                # deferred to BotStoppedSpeakingFrame.
                FunctionCallResultFrame(
                    function_name="get_weather",
                    tool_call_id="1",
                    arguments={},
                    result={"conditions": "Sunny"},
                ),
                append_response("task finished"),
                SleepFrame(0.1),
                # The deferred push fires here — inference requested, no
                # response started yet.
                BotStoppedSpeakingFrame(),
                SleepFrame(0.5),
            ],
            expected_up_frames=[LLMContextFrame],
        )
        self.assertEqual(released, [])
        self.assertNotEqual(context.messages[-1]["content"], "task finished")

    async def test_release_after_requested_inference_completes(self):
        # Companion to the request-window test: once the requested inference
        # actually runs (LLMFullResponseStart/End), the pending response
        # releases — two context pushes total, sequential and legitimate.
        context = LLMContext()
        aggregator = make_aggregator(context)

        released = []

        @aggregator.event_handler("on_response_released")
        async def on_response_released(aggregator, items):
            released.append(items)

        await run_test(
            aggregator,
            frames_to_send=[
                FunctionCallInProgressFrame(
                    function_name="get_weather",
                    tool_call_id="1",
                    arguments={},
                    cancel_on_interruption=True,
                ),
                BotStartedSpeakingFrame(),
                SleepFrame(0.1),
                FunctionCallResultFrame(
                    function_name="get_weather",
                    tool_call_id="1",
                    arguments={},
                    result={"conditions": "Sunny"},
                ),
                append_response("task finished"),
                SleepFrame(0.1),
                BotStoppedSpeakingFrame(),
                SleepFrame(0.2),
                # The response to the deferred push arrives...
                LLMFullResponseStartFrame(),
                LLMFullResponseEndFrame(),
                # ...and only now may the pending response release.
                SleepFrame(0.4),
            ],
            expected_up_frames=[LLMContextFrame, LLMContextFrame],
        )
        self.assertEqual(len(released), 1)
        self.assertEqual(context.messages[-1]["content"], "task finished")

    async def test_async_final_routes_through_strategy(self):
        # An async tool's final result with run_llm unset: the result lands
        # in context immediately, and the announcement is scheduled by the
        # strategy (here: quiet conversation, so released right away) with a
        # composed instruction message and a single inference trigger.
        context = LLMContext()
        aggregator = make_aggregator(context)

        released = []

        @aggregator.event_handler("on_response_released")
        async def on_response_released(aggregator, items):
            released.append(items)

        await run_test(
            aggregator,
            frames_to_send=[
                FunctionCallInProgressFrame(
                    function_name="slow_research",
                    tool_call_id="1",
                    arguments={},
                    cancel_on_interruption=False,
                ),
                SleepFrame(0.1),
                FunctionCallResultFrame(
                    function_name="slow_research",
                    tool_call_id="1",
                    arguments={},
                    result={"finding": "interesting"},
                ),
                SleepFrame(0.3),
            ],
            expected_up_frames=[LLMContextFrame],
        )
        # The composed announcement is the last message, referencing the tool.
        self.assertIn("slow_research", context.messages[-1]["content"])
        self.assertEqual(context.messages[-1]["role"], "developer")
        self.assertEqual(len(released), 1)
        self.assertIsInstance(released[0][0], CompletedToolResult)
        self.assertEqual(released[0][0].function_name, "slow_research")

    async def test_async_final_explicit_run_llm_true_also_routes(self):
        # run_llm decides *whether* to respond; the strategy decides *when*.
        # There is no per-call immediacy override: an explicit run_llm=True
        # on an async final routes through the strategy like the default.
        context = LLMContext()
        aggregator = make_aggregator(context)

        released = []

        @aggregator.event_handler("on_response_released")
        async def on_response_released(aggregator, items):
            released.append(items)

        await run_test(
            aggregator,
            frames_to_send=[
                FunctionCallInProgressFrame(
                    function_name="slow_research",
                    tool_call_id="1",
                    arguments={},
                    cancel_on_interruption=False,
                ),
                SleepFrame(0.1),
                FunctionCallResultFrame(
                    function_name="slow_research",
                    tool_call_id="1",
                    arguments={},
                    result={"finding": "interesting"},
                    properties=FunctionCallResultProperties(run_llm=True),
                ),
                SleepFrame(0.3),
            ],
            expected_up_frames=[LLMContextFrame],
        )
        self.assertEqual(len(released), 1)
        self.assertIsInstance(released[0][0], CompletedToolResult)

    async def test_async_final_run_llm_false_stays_silent(self):
        context = LLMContext()
        aggregator = make_aggregator(context)

        await run_test(
            aggregator,
            frames_to_send=[
                FunctionCallInProgressFrame(
                    function_name="slow_research",
                    tool_call_id="1",
                    arguments={},
                    cancel_on_interruption=False,
                ),
                SleepFrame(0.1),
                FunctionCallResultFrame(
                    function_name="slow_research",
                    tool_call_id="1",
                    arguments={},
                    result={"finding": "interesting"},
                    properties=FunctionCallResultProperties(run_llm=False),
                ),
                SleepFrame(0.1),
            ],
            expected_up_frames=[],
        )

    async def test_reactive_final_does_not_route_through_strategy(self):
        # A reactive call's result is an answer the user is waiting for:
        # inference runs immediately even with a strategy configured.
        context = LLMContext()
        aggregator = make_aggregator(context)

        released = []

        @aggregator.event_handler("on_response_released")
        async def on_response_released(aggregator, items):
            released.append(items)

        await run_test(
            aggregator,
            frames_to_send=[
                FunctionCallInProgressFrame(
                    function_name="get_weather",
                    tool_call_id="1",
                    arguments={},
                    cancel_on_interruption=True,
                ),
                SleepFrame(0.1),
                FunctionCallResultFrame(
                    function_name="get_weather",
                    tool_call_id="1",
                    arguments={},
                    result={"conditions": "Sunny"},
                ),
                SleepFrame(0.1),
            ],
            expected_up_frames=[LLMContextFrame],
        )
        self.assertEqual(released, [])

    async def test_async_finals_batch_in_a_quiet_conversation(self):
        # Two async tools complete back to back with nothing else happening.
        # Queuing restarts the settle window, so the first result waits for
        # the second instead of releasing alone and leaving it to trail as a
        # second spoken turn.
        context = LLMContext()
        aggregator = make_aggregator(context)

        released = []

        @aggregator.event_handler("on_response_released")
        async def on_response_released(aggregator, items):
            released.append(items)

        await run_test(
            aggregator,
            frames_to_send=[
                FunctionCallInProgressFrame(
                    function_name="research_a",
                    tool_call_id="1",
                    arguments={},
                    cancel_on_interruption=False,
                ),
                FunctionCallInProgressFrame(
                    function_name="research_b",
                    tool_call_id="2",
                    arguments={},
                    cancel_on_interruption=False,
                ),
                # Long enough that the window opened by the calls starting
                # has expired: the results arrive into a settled silence.
                SleepFrame(0.3),
                FunctionCallResultFrame(
                    function_name="research_a",
                    tool_call_id="1",
                    arguments={},
                    result={"finding": "A"},
                ),
                FunctionCallResultFrame(
                    function_name="research_b",
                    tool_call_id="2",
                    arguments={},
                    result={"finding": "B"},
                ),
                SleepFrame(0.3),
            ],
            expected_up_frames=[LLMContextFrame],
        )
        self.assertEqual(len(released), 1)
        self.assertEqual(len(released[0]), 2)
        announcement = context.messages[-1]["content"]
        self.assertIn("research_a", announcement)
        self.assertIn("research_b", announcement)

    async def test_async_finals_batch_into_one_announcement(self):
        # Two async tools complete while the bot is speaking: one release,
        # one composed message naming both, one inference trigger.
        context = LLMContext()
        aggregator = make_aggregator(context)

        released = []

        @aggregator.event_handler("on_response_released")
        async def on_response_released(aggregator, items):
            released.append(items)

        await run_test(
            aggregator,
            frames_to_send=[
                FunctionCallInProgressFrame(
                    function_name="research_a",
                    tool_call_id="1",
                    arguments={},
                    cancel_on_interruption=False,
                ),
                FunctionCallInProgressFrame(
                    function_name="research_b",
                    tool_call_id="2",
                    arguments={},
                    cancel_on_interruption=False,
                ),
                BotStartedSpeakingFrame(),
                SleepFrame(0.1),
                FunctionCallResultFrame(
                    function_name="research_a",
                    tool_call_id="1",
                    arguments={},
                    result={"finding": "A"},
                ),
                FunctionCallResultFrame(
                    function_name="research_b",
                    tool_call_id="2",
                    arguments={},
                    result={"finding": "B"},
                ),
                SleepFrame(0.1),
                BotStoppedSpeakingFrame(),
                SleepFrame(0.4),
            ],
            expected_up_frames=[LLMContextFrame],
        )
        announcement = context.messages[-1]["content"]
        self.assertIn("research_a", announcement)
        self.assertIn("research_b", announcement)
        self.assertEqual(len(released), 1)
        self.assertEqual(len(released[0]), 2)

    async def _run_one_completion(self, aggregator, expected_up_frames):
        """Drive a single async tool completion through a quiet conversation."""
        await run_test(
            aggregator,
            frames_to_send=[
                FunctionCallInProgressFrame(
                    function_name="slow_research",
                    tool_call_id="1",
                    arguments={},
                    cancel_on_interruption=False,
                ),
                SleepFrame(0.1),
                FunctionCallResultFrame(
                    function_name="slow_research",
                    tool_call_id="1",
                    arguments={},
                    result={"finding": "interesting"},
                ),
                SleepFrame(0.3),
            ],
            expected_up_frames=expected_up_frames,
        )

    def _aggregator_with(self, context, announcement):
        return LLMAssistantAggregator(
            context,
            params=LLMAssistantAggregatorParams(
                response_strategy=DelayedResponseStrategy(
                    settle_secs=0.1, announcement=announcement
                )
            ),
        )

    async def test_single_result_style_states_the_result(self):
        # The default for one completion: announce it outright.
        context = LLMContext()
        aggregator = self._aggregator_with(context, AnnouncementConfig())

        await self._run_one_completion(aggregator, [LLMContextFrame])
        announcement = context.messages[-1]["content"]
        self.assertIn("what it found", announcement)
        self.assertIn("slow_research", announcement)

    async def test_single_notify_style_offers_without_stating(self):
        context = LLMContext()
        aggregator = self._aggregator_with(
            context, AnnouncementConfig(single_style=AnnouncementStyle.NOTIFY)
        )

        await self._run_one_completion(aggregator, [LLMContextFrame])
        announcement = context.messages[-1]["content"]
        self.assertIn("ask whether they'd like to hear it", announcement)
        self.assertIn("Do not state the result", announcement)

    async def test_custom_prompt_replaces_the_default(self):
        context = LLMContext()
        aggregator = self._aggregator_with(
            context, AnnouncementConfig(single_prompt="CUSTOM about {name}")
        )

        await self._run_one_completion(aggregator, [LLMContextFrame])
        self.assertEqual(context.messages[-1]["content"], "CUSTOM about slow_research")

    def test_default_styles_split_by_cardinality(self):
        # One result is stated; a batch is offered rather than delivered.
        config = AnnouncementConfig()
        one = config.compose([CompletedToolResult("research_a", "1", {}, {"finding": "a"})])
        many = config.compose(
            [
                CompletedToolResult("research_a", "1", {}, {"finding": "a"}),
                CompletedToolResult("research_b", "2", {}, {"finding": "b"}),
            ]
        )
        self.assertIn("what it found", one[0]["content"])
        self.assertNotIn("Do not state", one[0]["content"])
        self.assertIn("Do not state the results", many[0]["content"])
        self.assertIn("'research_a', 'research_b'", many[0]["content"])
        self.assertIn("2 background tasks", many[0]["content"])

    def test_unknown_prompt_placeholder_is_rejected(self):
        # A typo'd placeholder fails at construction rather than mid-call.
        with self.assertRaises(ValueError):
            AnnouncementConfig(single_prompt="about {nmae}")
        with self.assertRaises(ValueError):
            AnnouncementConfig(multiple_prompt="{count} tasks: {name}")

    async def test_append_without_run_llm_updates_context_silently(self):
        # Replay preserves the inner frame's semantics: an append with
        # run_llm=False lands in context at the opening without triggering
        # inference.
        context = LLMContext()
        aggregator = make_aggregator(context)

        await run_test(
            aggregator,
            frames_to_send=[append_response("background fact", run_llm=False), SleepFrame(0.3)],
            expected_up_frames=[],
        )
        self.assertEqual(context.messages[-1]["content"], "background fact")


if __name__ == "__main__":
    unittest.main()
