#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the eval harness.

Two layers:

- :class:`TestTranslate` unit-tests the RTVI-server-message → friendly-event
  translation in isolation (pure, fast).
- :class:`TestEvalsHarnessIntegration` runs scenarios via :meth:`EvalScriptSession.from_scenario` against a fake
  RTVI WebSocket server that replies to ``client-ready``/``send-text`` with
  scripted RTVI server messages — exercising the handshake, send/receive, event
  matching, and context paths without a real bot pipeline.
- :class:`TestProgressEvent` covers the ``on_progress`` event and the deprecated
  callback that feeds it.
"""

import asyncio
import base64
import json
import socket
import tempfile
import time
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import websockets

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.evals.audio import load_user_audio
from pipecat.evals.client import EvalClient, _BotFrameSink, _PersonaTurnRelay
from pipecat.evals.events import EvalEventStream
from pipecat.evals.matcher import ExpectationMatcher
from pipecat.evals.persona import EvalPersona
from pipecat.evals.results import EvalTrace
from pipecat.evals.scenario import (
    EvalExpectation,
    EvalFunctionCall,
    EvalScriptScenario,
    EvalScriptTurn,
    EvalSendAfter,
)
from pipecat.evals.script_session import EvalScriptSession
from pipecat.frames.frames import (
    AggregationType,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InputTransportMessageFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


def _rtvi(msg_type: str, data: dict | None = None) -> str:
    return json.dumps({"label": RTVI.MESSAGE_LABEL, "type": msg_type, "data": data})


def _session(bot_audio: bool = False) -> EvalScriptSession:
    return EvalScriptSession(
        EvalScriptScenario(name="t", turns=[], bot_audio=bot_audio), "ws://localhost:0"
    )


def _stream(bot_audio: bool = False) -> EvalEventStream:
    return EvalEventStream(bot_audio=bot_audio, trace=EvalTrace())


def _matcher(judge=None, bot_audio: bool = False) -> ExpectationMatcher:
    return ExpectationMatcher(stream=_stream(bot_audio), judge=judge, trace=EvalTrace())


def _client(
    scenario: EvalScriptScenario | None = None,
    bot_audio: bool = False,
    bot_url: str = "ws://localhost:0",
) -> EvalClient:
    scenario = scenario or EvalScriptScenario(name="t", turns=[], bot_audio=bot_audio)
    return EvalScriptSession(scenario, bot_url)._client


def _capture_injected(client: EvalClient) -> list:
    """Stand in for the client's sink, collecting the user-turn frames it is handed."""
    injected: list = []

    class _FakeSink:
        async def inject(self, frame):
            injected.append(frame)

    client._sink = _FakeSink()
    return injected


class TestFramesToEvents(unittest.TestCase):
    """The bot's frames and reported messages map to the scenario's events."""

    def _bare(self, event):
        self.assertIsNotNone(event)
        # The timing a reply carries for the measures is not what these tests map.
        return {k: v for k, v in event.items() if k != "started_at"}

    def test_a_reply_carries_the_time_of_its_first_token(self):
        s = _stream(bot_audio=False)
        s.frame_to_event(LLMFullResponseStartFrame())
        s.frame_to_event(LLMTextFrame(text="Hello "))
        s.frame_to_event(LLMTextFrame(text="world"))
        event = s.frame_to_event(LLMFullResponseEndFrame())
        self.assertIn("started_at", event)
        self.assertLessEqual(event["started_at"], s.elapsed())
        # A response with no text carries no time.
        s.frame_to_event(LLMFullResponseStartFrame())
        event = s.frame_to_event(LLMFullResponseEndFrame())
        self.assertNotIn("started_at", event)

    def test_llm_lifecycle_aggregates_text(self):
        s = _stream(bot_audio=False)
        self.assertEqual(s.frame_to_event(LLMFullResponseStartFrame()), {"type": "llm_started"})
        self.assertIsNone(s.frame_to_event(LLMTextFrame(text="Hello ")))
        self.assertIsNone(s.frame_to_event(LLMTextFrame(text="world")))
        self.assertEqual(
            self._bare(s.frame_to_event(LLMFullResponseEndFrame())),
            {"type": "llm_response", "text": "Hello world"},
        )

    def test_interruption_suppresses_straggler(self):
        s = _stream(bot_audio=False)
        interrupt = InputTransportMessageFrame(
            message={"label": RTVI.MESSAGE_LABEL, "type": "bot-interrupted"}
        )
        s.frame_to_event(LLMFullResponseStartFrame())
        s.frame_to_event(LLMTextFrame(text="Tell me about Paris"))
        self.assertEqual(s.frame_to_event(interrupt), {"type": "bot_interrupted"})
        # Straggler from the interrupted response is dropped.
        self.assertIsNone(s.frame_to_event(LLMTextFrame(text=" what would")))
        self.assertIsNone(s.frame_to_event(LLMFullResponseEndFrame()))
        # The genuinely new response.
        s.frame_to_event(LLMFullResponseStartFrame())
        s.frame_to_event(LLMTextFrame(text="Tokyo"))
        self.assertEqual(
            self._bare(s.frame_to_event(LLMFullResponseEndFrame())),
            {"type": "llm_response", "text": "Tokyo"},
        )

    def test_user_transcription_from_message(self):
        # The bot's reported user-transcription arrives as a raw RTVI message frame,
        # not a TranscriptionFrame (which the eval reserves for the bot's response).
        s = _stream()

        def msg(data):
            return InputTransportMessageFrame(
                message={"label": RTVI.MESSAGE_LABEL, "type": "user-transcription", "data": data}
            )

        self.assertEqual(
            s.frame_to_event(msg({"text": "hello", "final": True})),
            {"type": "user_transcription", "transcript": "hello"},
        )
        # Interim transcriptions are ignored.
        self.assertIsNone(s.frame_to_event(msg({"text": "hel", "final": False})))

    def test_transcription_frame_is_ignored_by_the_sink(self):
        # The user aggregator consumes the STT's TranscriptionFrames to build the
        # bot's turn; the response is emitted from on_user_turn_stopped, not here.
        frame = TranscriptionFrame(text="Paris", user_id="bot", timestamp="t")
        self.assertIsNone(_stream(bot_audio=True).frame_to_event(frame))

    def test_reported_speaking_and_vad_events_from_messages(self):
        # The bot's reports about the harness (its raw VAD and turn-level speaking)
        # arrive as raw messages and map to the matching scenario events.
        s = _stream(bot_audio=True)

        def msg(msg_type):
            return InputTransportMessageFrame(
                message={"label": RTVI.MESSAGE_LABEL, "type": msg_type}
            )

        self.assertEqual(
            s.frame_to_event(msg("user-started-speaking")),
            {"type": "user_started_speaking"},
        )
        self.assertEqual(
            s.frame_to_event(msg("user-stopped-speaking")),
            {"type": "user_stopped_speaking"},
        )
        self.assertEqual(
            s.frame_to_event(msg("vad-user-started-speaking")),
            {"type": "vad_user_started_speaking"},
        )
        self.assertEqual(
            s.frame_to_event(msg("vad-user-stopped-speaking")),
            {"type": "vad_user_stopped_speaking"},
        )
        self.assertEqual(
            s.frame_to_event(msg("bot-interrupted")),
            {"type": "bot_interrupted"},
        )

    def test_computed_vad_and_speaking_frames_are_ignored(self):
        # The user aggregator's own VAD/speaking/interruption frames (computed from
        # the bot's audio) are internal plumbing, not scenario events.
        s = _stream(bot_audio=True)
        for frame in (
            VADUserStartedSpeakingFrame(),
            VADUserStoppedSpeakingFrame(),
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            InterruptionFrame(),
        ):
            self.assertIsNone(s.frame_to_event(frame))

    def test_user_started_speaking_message_discards_interrupted_output(self):
        # A new user turn drops the bot's leftover output (but keeps a queued
        # user_transcription, which is the turn's input).
        s = _stream(bot_audio=True)
        s._text_buffer = ["greeting"]
        s._queue.put_nowait({"type": "response", "text": "greeting"})
        s._queue.put_nowait({"type": "user_transcription", "transcript": "hi"})
        msg = InputTransportMessageFrame(
            message={"label": RTVI.MESSAGE_LABEL, "type": "user-started-speaking"}
        )
        self.assertEqual(s.frame_to_event(msg), {"type": "user_started_speaking"})
        self.assertEqual(s._text_buffer, [])
        self.assertEqual(s._queue.get_nowait(), {"type": "user_transcription", "transcript": "hi"})
        self.assertTrue(s._queue.empty())

    def test_function_call(self):
        s = _stream()
        event = self._bare(
            s.frame_to_event(
                FunctionCallInProgressFrame(
                    function_name="get_weather", tool_call_id="c", arguments={"city": "Paris"}
                )
            )
        )
        self.assertEqual(
            event, {"type": "function_call", "name": "get_weather", "args": {"city": "Paris"}}
        )

    def test_tts_text_only_in_audio_mode(self):
        def tts(text):
            return TTSTextFrame(text=text, aggregated_by=AggregationType.SENTENCE)

        self.assertIsNone(_stream(bot_audio=False).frame_to_event(tts("x")))
        self.assertEqual(
            _stream(bot_audio=True).frame_to_event(tts("spoken")),
            {"type": "tts_response", "text": "spoken"},
        )

    def test_empty_response_still_emitted(self):
        # An interrupted response (no text) emits an empty llm_response; the
        # matcher's aggregation decides whether that should pass or fail.
        s = _stream(bot_audio=False)
        self.assertEqual(s.frame_to_event(LLMFullResponseStartFrame()), {"type": "llm_started"})
        self.assertEqual(
            self._bare(s.frame_to_event(LLMFullResponseEndFrame())),
            {"type": "llm_response", "text": ""},
        )

    def test_unmapped_message_ignored(self):
        msg = InputTransportMessageFrame(
            message={"label": RTVI.MESSAGE_LABEL, "type": "metrics", "data": {}}
        )
        self.assertIsNone(_stream().frame_to_event(msg))


class _FakeJudge:
    """Returns queued verdicts without calling a real LLM."""

    def __init__(self, verdicts: list[str]):
        self._verdicts = list(verdicts)
        self.calls: list[str] = []
        self.segments: list[str] = []
        # The tool calls added to the conversation, and the calls judged.
        self.tool_calls: list[tuple[str, dict | None]] = []
        self.call_asks: list[tuple[str, dict | None, str]] = []

    def add_user_message(self, text):
        pass

    def add_assistant_message(self, text):
        self.segments.append(text)

    def add_tool_call(self, name, args):
        self.tool_calls.append((name, args))

    async def evaluate(self, criterion: str):
        self.calls.append(criterion)
        return self._next_verdict()

    async def evaluate_call(self, name, args, criterion):
        self.call_asks.append((name, args, criterion))
        return self._next_verdict()

    def _next_verdict(self):
        from pipecat.evals.judge import JudgeVerdict

        v = self._verdicts.pop(0)
        return JudgeVerdict(verdict=v, reason=f"({v})", raw_response="")


class TestBotSpeaking(unittest.IsolatedAsyncioTestCase):
    """The stream follows the bot's own report of whether it is speaking."""

    async def test_the_bots_speaking_frames_become_events_and_a_wait(self):
        stream = _stream()
        self.assertFalse(stream.bot_speaking)
        self.assertTrue(await stream.wait_bot_quiet(0.01))

        self.assertEqual(
            stream.frame_to_event(BotStartedSpeakingFrame()), {"type": "bot_started_speaking"}
        )
        self.assertTrue(stream.bot_speaking)
        self.assertFalse(await stream.wait_bot_quiet(0.05))

        self.assertEqual(
            stream.frame_to_event(BotStoppedSpeakingFrame()), {"type": "bot_stopped_speaking"}
        )
        self.assertFalse(stream.bot_speaking)
        self.assertTrue(await stream.wait_bot_quiet(0.01))

    async def test_an_interruption_counts_as_the_bot_going_quiet(self):
        stream = _stream()
        stream.frame_to_event(BotStartedSpeakingFrame())
        stream._interrupted()
        self.assertFalse(stream.bot_speaking)


class TestSendWaitsForTheBot(unittest.IsolatedAsyncioTestCase):
    """A turn is not sent over a speaking bot."""

    async def test_the_send_waits_until_the_bot_stops_speaking(self):
        session = _session()
        driver, stream = session._driver, session._stream
        stream.frame_to_event(BotStartedSpeakingFrame())

        async def stop_soon():
            await asyncio.sleep(0.15)
            stream.frame_to_event(BotStoppedSpeakingFrame())

        task = asyncio.create_task(stop_soon())
        started = asyncio.get_running_loop().time()
        await driver._await_bot_quiet()
        await task
        self.assertGreaterEqual(asyncio.get_running_loop().time() - started, 0.15)

    async def test_an_observing_turn_lets_the_previous_reply_end_first(self):
        # The bot is still speaking turn 1's reply when turn 2, which only
        # listens for the bot's next move, begins: the rest of that reply,
        # however late its transcription lands, is not turn 2's response.
        session = _session(bot_audio=True)
        driver, stream = session._driver, session._stream
        stream.frame_to_event(BotStartedSpeakingFrame())
        stream.bot_turn_started()
        await stream.append({"type": "response", "text": "Spring in Japan is gorgeous."})

        async def finish_reply():
            await asyncio.sleep(0.1)
            stream.frame_to_event(BotStoppedSpeakingFrame())

        task = asyncio.create_task(finish_reply())
        await driver._await_previous_reply()
        await task
        # The queued tail is gone, and one transcribed after the wait is stale.
        with self.assertRaises(TimeoutError):
            await stream.next_event("response", time.monotonic() + 0.05)
        await stream.bot_turn_stopped("Do you have cities in mind?")
        with self.assertRaises(TimeoutError):
            await stream.next_event("response", time.monotonic() + 0.05)

    async def test_a_quiet_bot_holds_nothing_up(self):
        session = _session()
        started = asyncio.get_running_loop().time()
        await session._driver._await_bot_quiet()
        self.assertLess(asyncio.get_running_loop().time() - started, 0.05)


class TestJudgeNoIsProvisional(unittest.IsolatedAsyncioTestCase):
    """A judge's "no" fails the reply only once the reply is over."""

    def setUp(self):
        self.exp = EvalExpectation(event="response", eval="gives the weather")

    async def _match(self, s: ExpectationMatcher, budget_ms: int = 5000):
        return await s.match(self.exp, time.monotonic(), budget_ms, 0, 0)

    async def test_a_no_while_the_bot_speaks_waits_for_the_rest(self):
        judge = _FakeJudge(["no", "yes"])
        s = _matcher(judge, bot_audio=True)
        s._stream.frame_to_event(BotStartedSpeakingFrame())
        await s._stream.append({"type": "response", "text": "Sure. Why don't skeletons fight?"})

        async def rest():
            await asyncio.sleep(0.1)
            await s._stream.append({"type": "response", "text": "The weather is 75 degrees."})

        task = asyncio.create_task(rest())
        self.assertIsNone(await self._match(s))
        await task
        self.assertEqual(judge.calls, ["gives the weather"] * 2)

    async def test_a_no_after_the_bot_stopped_waits_out_the_transcription(self):
        judge = _FakeJudge(["no", "yes"])
        s = _matcher(judge, bot_audio=True)
        await s._stream.append({"type": "response", "text": "Transferring you now."})

        async def late_sentence():
            await asyncio.sleep(0.1)
            await s._stream.append({"type": "response", "text": "The boots cost $299."})

        task = asyncio.create_task(late_sentence())
        self.assertIsNone(await self._match(s))
        await task

    async def test_a_no_on_a_finished_reply_fails(self):
        from pipecat.evals import matcher as matcher_module

        judge = _FakeJudge(["no"])
        s = _matcher(judge, bot_audio=True)
        await s._stream.append({"type": "response", "text": "I like turtles."})
        with patch.object(matcher_module, "JUDGE_NO_GRACE_S", 0.1):
            failure = await self._match(s)
        assert failure is not None
        self.assertEqual(failure.kind, "judge_no")


class TestBotTurn(unittest.IsolatedAsyncioTestCase):
    """The bot's finished spoken turn is the reply only if it began after the input."""

    async def _responses(self, s: EvalEventStream) -> list[str]:
        out = []
        while not s._queue.empty():
            out.append(s._queue.get_nowait()["text"])
        return out

    async def test_turn_after_input_and_llm_restart_is_the_reply(self):
        s = _stream(bot_audio=True)
        s.input_sent()
        s.frame_to_event(LLMFullResponseStartFrame())  # the bot's new response
        s.bot_turn_started()
        await s.bot_turn_stopped("Tokyo.")
        self.assertEqual(await self._responses(s), ["Tokyo."])

    async def test_turn_begun_before_input_is_dropped_even_after_llm_restart(self):
        # The interrupted turn finalizes late, after the bot has already started
        # its real reply: still not the reply.
        s = _stream(bot_audio=True)
        s.bot_turn_started()
        s.input_sent()
        s.frame_to_event(LLMFullResponseStartFrame())
        await s.bot_turn_stopped("Let's take a journey")
        self.assertEqual(await self._responses(s), [])

    async def test_turn_before_llm_restart_is_dropped(self):
        s = _stream(bot_audio=True)
        s.input_sent()
        s.bot_turn_started()
        await s.bot_turn_stopped("straggler")
        self.assertEqual(await self._responses(s), [])

    async def test_bot_first_turn_needs_no_input(self):
        s = _stream(bot_audio=True)
        s.bot_turn_started()
        await s.bot_turn_stopped("Hello there!")
        self.assertEqual(await self._responses(s), ["Hello there!"])

    async def test_empty_turn_is_not_a_response(self):
        s = _stream(bot_audio=True)
        s.bot_turn_started()
        await s.bot_turn_stopped("")
        self.assertEqual(await self._responses(s), [])


class TestMatchAbsent(unittest.IsolatedAsyncioTestCase):
    """An ``absent: true`` expectation passes on a quiet window, fails on arrival."""

    async def test_absent_passes_when_no_event_arrives(self):
        import time

        s = _matcher()
        expectation = EvalExpectation(event="llm_response", absent=True)
        failure = await s.match(expectation, time.monotonic(), 100, 0, 0)
        self.assertIsNone(failure)
        self.assertIn("no 'llm_response'", s.last_match_text)

    async def test_absent_fails_when_event_arrives(self):
        import time

        s = _matcher()
        s._stream._queue.put_nowait({"type": "llm_response", "text": "I repeat myself"})
        expectation = EvalExpectation(event="llm_response", absent=True)
        failure = await s.match(expectation, time.monotonic(), 100, 3, 2)
        self.assertIsNotNone(failure)
        self.assertEqual(failure.turn_index, 3)
        self.assertEqual(failure.expectation_index, 2)
        self.assertIn("I repeat myself", failure.reason)

    async def test_absent_ignores_other_event_types(self):
        import time

        s = _matcher()
        # Unrelated events in the window must not trip the absence check.
        s._stream._queue.put_nowait({"type": "user_stopped_speaking"})
        s._stream._queue.put_nowait({"type": "tts_response", "text": "spoken"})
        expectation = EvalExpectation(event="llm_response", absent=True)
        failure = await s.match(expectation, time.monotonic(), 100, 0, 0)
        self.assertIsNone(failure)


def _call(name: str, args: dict | None = None, stopped: bool = False) -> dict:
    return {
        "type": "function_call_stopped" if stopped else "function_call",
        "name": name,
        "args": args,
    }


class TestFunctionCallEval(unittest.IsolatedAsyncioTestCase):
    """``eval:`` on a function call puts each matched call to the judge."""

    CRITERION = "the suggestion is about tracing, for Jennifer Smith"
    ARGS = {"title": "OpenTelemetry tracing", "speaker": "Jennifer Smith"}

    def _exp(self, *names: str, event: str = "function_call") -> EvalExpectation:
        calls = [EvalFunctionCall(name=n) for n in names] or None
        return EvalExpectation(event=event, calls=calls, eval=self.CRITERION)

    async def _match(self, s: ExpectationMatcher, exp: EvalExpectation, budget_ms: int = 1000):
        return await s.match(exp, time.monotonic(), budget_ms, 0, 0)

    async def test_yes_passes_and_the_ask_carries_the_args(self):
        judge = _FakeJudge(["yes"])
        s = _matcher(judge)
        await s._stream.append(_call("submit_session_suggestion", self.ARGS))
        self.assertIsNone(await self._match(s, self._exp("submit_session_suggestion")))
        self.assertEqual(
            judge.call_asks, [("submit_session_suggestion", self.ARGS, self.CRITERION)]
        )
        self.assertEqual(s.last_match_text, "submit_session_suggestion")

    async def test_no_fails_with_the_judges_reason(self):
        judge = _FakeJudge(["no"])
        s = _matcher(judge)
        await s._stream.append(_call("submit_session_suggestion", self.ARGS))
        failure = await self._match(s, self._exp("submit_session_suggestion"))
        assert failure is not None
        self.assertEqual(failure.kind, "judge_no")
        self.assertIn("judge said no — (no)", failure.reason)
        self.assertIn("submit_session_suggestion(", failure.reason)

    async def test_continue_counts_as_no(self):
        judge = _FakeJudge(["continue"])
        s = _matcher(judge)
        await s._stream.append(_call("submit_session_suggestion", self.ARGS))
        failure = await self._match(s, self._exp("submit_session_suggestion"))
        assert failure is not None
        self.assertEqual(failure.kind, "judge_no")
        self.assertIn("judge said continue", failure.reason)

    async def test_every_listed_call_is_judged(self):
        judge = _FakeJudge(["yes", "no"])
        s = _matcher(judge)
        await s._stream.append(_call("lookup", {"q": "a"}))
        await s._stream.append(_call("submit", {"q": "b"}))
        failure = await self._match(s, self._exp("lookup", "submit"))
        assert failure is not None
        self.assertEqual(failure.kind, "judge_no")
        self.assertEqual([ask[0] for ask in judge.call_asks], ["lookup", "submit"])

    async def test_verbatim_args_are_checked_before_the_judge(self):
        judge = _FakeJudge([])  # would IndexError if the judge were asked
        s = _matcher(judge)
        await s._stream.append(_call("submit", {"speaker": "someone else"}))
        exp = EvalExpectation(
            event="function_call",
            calls=[EvalFunctionCall(name="submit", args={"speaker": "Jennifer Smith"})],
            eval=self.CRITERION,
        )
        failure = await self._match(s, exp, budget_ms=200)
        assert failure is not None
        self.assertEqual(failure.kind, "function_args_mismatch")
        self.assertEqual(judge.call_asks, [])

    async def test_stopped_calls_are_judged_too(self):
        judge = _FakeJudge(["yes"])
        s = _matcher(judge)
        await s._stream.append(_call("write_report", {"cancelled": True}, stopped=True))
        exp = self._exp("write_report", event="function_call_stopped")
        self.assertIsNone(await self._match(s, exp))
        self.assertEqual(judge.call_asks, [("write_report", {"cancelled": True}, self.CRITERION)])

    async def test_no_judge_fails_before_matching(self):
        s = _matcher(judge=None)
        await s._stream.append(_call("submit", self.ARGS))
        failure = await self._match(s, self._exp("submit"))
        assert failure is not None
        self.assertEqual(failure.kind, "no_judge")


class TestToolCallsInJudgeContext(unittest.IsolatedAsyncioTestCase):
    """Every function call the matcher pops reaches the judge's conversation once, in order."""

    async def _match(self, s: ExpectationMatcher, exp: EvalExpectation, budget_ms: int = 1000):
        return await s.match(exp, time.monotonic(), budget_ms, 0, 0)

    async def test_matched_and_buffered_calls_are_added_once_in_arrival_order(self):
        judge = _FakeJudge([])
        s = _matcher(judge)
        await s._stream.append(_call("lookup", {"q": "a"}))
        await s._stream.append(_call("submit", {"q": "b"}))
        await s._stream.append(_call("lookup", {"q": "a"}, stopped=True))
        # Waiting for `submit` buffers `lookup`; then `lookup` comes from the buffer.
        exp = EvalExpectation(
            event="function_call",
            calls=[EvalFunctionCall(name="submit"), EvalFunctionCall(name="lookup")],
        )
        self.assertIsNone(await self._match(s, exp))
        self.assertEqual(judge.tool_calls, [("lookup", {"q": "a"}), ("submit", {"q": "b"})])

    async def test_a_reply_eval_sees_the_call(self):
        """A criterion about what the bot submitted passes only with the call in context."""
        from pipecat.evals import matcher as matcher_module
        from pipecat.evals.judge import EvalJudge

        class _ToolAwareLLM:
            """Says yes only if the conversation shows the submission for Jennifer Smith."""

            async def run_inference(self, context, max_tokens=None, system_instruction=None):
                submitted = any(
                    m["role"] == "assistant"
                    and m["content"].startswith("[tool call] submit_session_suggestion(")
                    and '"speaker":"Jennifer Smith"' in m["content"]
                    for m in context.get_messages()
                )
                if submitted:
                    return '{"verdict": "yes", "reason": "the call shows it"}'
                return '{"verdict": "no", "reason": "no such submission in the conversation"}'

        call = _call("submit_session_suggestion", {"title": "Tracing", "speaker": "Jennifer Smith"})
        reply = {"type": "llm_response", "text": "Submitted your tracing session, Jennifer."}
        reply_exp = EvalExpectation(
            event="llm_response", eval="confirms the session it actually submitted"
        )

        # With the call expectation matched first, the judge sees the call line
        # and then the confirmation, as one reply.
        judge = EvalJudge(_ToolAwareLLM())
        s = _matcher(judge)
        await s._stream.append(dict(call))
        await s._stream.append(dict(reply))
        call_exp = EvalExpectation(
            event="function_call", calls=[EvalFunctionCall(name="submit_session_suggestion")]
        )
        self.assertIsNone(await self._match(s, call_exp))
        self.assertIsNone(await self._match(s, reply_exp))
        self.assertEqual(
            [m["content"] for m in judge._context.get_messages()],
            [
                '[tool call] submit_session_suggestion({"title":"Tracing","speaker":"Jennifer Smith"})',
                "Submitted your tracing session, Jennifer.",
            ],
        )

        # Without it (the call skipped past, never popped as a call) the same
        # judge, criterion and reply fail: the words alone are not evidence.
        judge = EvalJudge(_ToolAwareLLM())
        s = _matcher(judge)
        await s._stream.append(dict(call))
        await s._stream.append(dict(reply))
        with patch.object(matcher_module, "JUDGE_NO_GRACE_S", 0.1):
            failure = await self._match(s, reply_exp)
        assert failure is not None
        self.assertEqual(failure.kind, "judge_no")


class TestEvaluateAggregate(unittest.IsolatedAsyncioTestCase):
    """The pass/fail/continue decision over accumulated response text."""

    async def test_text_contains_present_passes(self):
        s = _matcher(bot_audio=False)
        exp = EvalExpectation(event="llm_response", text_contains="Paris")
        self.assertEqual(await s._evaluate_aggregate("The capital is Paris.", exp), ("pass", ""))

    async def test_text_contains_absent_continues(self):
        s = _matcher()
        exp = EvalExpectation(event="llm_response", text_contains="Paris")
        status, _ = await s._evaluate_aggregate("Let me check on that.", exp)
        self.assertEqual(status, "continue")

    async def test_eval_yes_passes(self):
        s = _matcher()
        s._judge = _FakeJudge(["yes"])
        exp = EvalExpectation(event="llm_response", eval="describes the weather")
        status, reason = await s._evaluate_aggregate("It's 75 and sunny.", exp)
        self.assertEqual(status, "pass")
        self.assertIn("judge said yes", reason)

    async def test_eval_no_fails(self):
        s = _matcher()
        s._judge = _FakeJudge(["no"])
        exp = EvalExpectation(event="llm_response", eval="describes the weather")
        status, reason = await s._evaluate_aggregate("I like turtles.", exp)
        self.assertEqual(status, "fail")
        self.assertIn("judge said no", reason)

    async def test_eval_continue_waits_for_more(self):
        s = _matcher()
        s._judge = _FakeJudge(["continue"])
        exp = EvalExpectation(event="llm_response", eval="describes the weather")
        status, _ = await s._evaluate_aggregate("Let me check on that.", exp)
        self.assertEqual(status, "continue")

    async def test_eval_empty_aggregate_skips_judge(self):
        s = _matcher()
        judge = _FakeJudge([])  # would IndexError if the judge were called
        s._judge = judge
        exp = EvalExpectation(event="llm_response", eval="describes the weather")
        status, _ = await s._evaluate_aggregate("   ", exp)
        self.assertEqual(status, "continue")
        self.assertEqual(judge.calls, [])


class TestRequiredReportLevel(unittest.TestCase):
    """The minimal function-call report level the harness asks the bot for."""

    def _level(self, *expects) -> str | None:
        scenario = EvalScriptScenario(
            name="t", bot_audio=False, turns=[EvalScriptTurn(user="x", expect=list(expects))]
        )
        return scenario.required_report_level()

    def test_none_without_function_call(self):
        self.assertIsNone(self._level(EvalExpectation(event="llm_response")))

    def test_none_for_bare_function_call(self):
        # Just asserting the call happened needs no name/args, so no elevation.
        self.assertIsNone(self._level(EvalExpectation(event="function_call")))

    def test_name_when_only_name_asserted(self):
        self.assertEqual(
            self._level(
                EvalExpectation(event="function_call", calls=[EvalFunctionCall(name="get_weather")])
            ),
            "name",
        )

    def test_full_when_args_asserted(self):
        self.assertEqual(
            self._level(
                EvalExpectation(
                    event="function_call",
                    calls=[EvalFunctionCall(name="get_weather", args={"city": "P"})],
                )
            ),
            "full",
        )

    def test_full_when_a_call_is_judged(self):
        # The judge reads the call's arguments, so names alone are not enough.
        self.assertEqual(
            self._level(
                EvalExpectation(
                    event="function_call",
                    calls=[EvalFunctionCall(name="submit")],
                    eval="submitted for the right person",
                )
            ),
            "full",
        )

    def test_full_when_a_reply_after_a_call_is_judged(self):
        # The call reaches the judge's conversation, with its arguments.
        self.assertEqual(
            self._level(
                EvalExpectation(event="function_call"),
                EvalExpectation(event="llm_response", eval="confirms what it submitted"),
            ),
            "full",
        )


class TestNeedsVadEvents(unittest.TestCase):
    """The harness enables raw VAD events only when a scenario references them."""

    def _needs(self, turn: EvalScriptTurn) -> bool:
        scenario = EvalScriptScenario(name="t", turns=[turn])
        return scenario.needs_vad_events()

    def test_false_without_vad_events(self):
        self.assertFalse(
            self._needs(EvalScriptTurn(user="x", expect=[EvalExpectation(event="response")]))
        )

    def test_true_when_expected(self):
        self.assertTrue(
            self._needs(
                EvalScriptTurn(
                    user="x", expect=[EvalExpectation(event="vad_user_started_speaking")]
                )
            )
        )

    def test_true_when_used_as_send_after_anchor(self):
        self.assertTrue(
            self._needs(
                EvalScriptTurn(
                    user="x",
                    expect=[EvalExpectation(event="response")],
                    send_after=EvalSendAfter(event="vad_user_stopped_speaking", delay_ms=2000),
                )
            )
        )


class TestConnectURL(unittest.TestCase):
    """The harness signals skip-TTS via the connect URL in text mode."""

    def _url(self, bot_audio: bool, base: str = "ws://localhost:7860") -> str:
        scenario = EvalScriptScenario(name="t", turns=[], bot_audio=bot_audio)
        return _client(scenario, bot_url=base)._connect_url()

    def test_text_mode_adds_skip_tts(self):
        self.assertEqual(self._url(bot_audio=False), "ws://localhost:7860?skip_tts=true")

    def test_audio_mode_is_plain(self):
        self.assertEqual(self._url(bot_audio=True), "ws://localhost:7860")

    def test_appends_to_existing_query(self):
        self.assertEqual(
            self._url(bot_audio=False, base="ws://localhost:7860?x=1"),
            "ws://localhost:7860?x=1&skip_tts=true",
        )

    def test_response_adds_capture_audio(self):
        scenario = EvalScriptScenario(
            name="t",
            bot_audio=True,
            turns=[EvalScriptTurn(user="x", expect=[EvalExpectation(event="response", eval="ok")])],
        )
        url = _client(scenario, bot_url="ws://localhost:7860")._connect_url()
        self.assertIn("capture_bot_audio=true", url)
        self.assertNotIn("skip_tts", url)  # audio mode, so no skip

    def test_skip_tts_flag_in_text_mode(self):
        # Text-mode scenarios silence the bot (skip_tts before any greeting);
        # audio-mode scenarios let it speak.
        text = EvalScriptScenario(name="t", turns=[], bot_audio=False)
        self.assertIn("skip_tts=true", _client(text, bot_url="ws://localhost:7860")._connect_url())
        audio = EvalScriptScenario(name="t", turns=[], bot_audio=True)
        self.assertNotIn("skip_tts", _client(audio, bot_url="ws://localhost:7860")._connect_url())


class TestResponseTranscriptionSkip(unittest.IsolatedAsyncioTestCase):
    async def test_skipped_without_audio_mode(self):
        # The `response` transcription needs the bot's audio; without audio mode,
        # skip (don't run a guaranteed failure).
        scenario = EvalScriptScenario(
            name="t",
            bot_audio=False,
            turns=[EvalScriptTurn(user="x", expect=[EvalExpectation(event="response", eval="ok")])],
        )
        result = await EvalScriptSession(scenario, "ws://localhost:0").run()
        self.assertIsNotNone(result.skipped)
        self.assertFalse(result.passed)
        self.assertIn("response", result.skipped)
        self.assertEqual([t.status for t in result.turns], ["not_run"])


class TestTextContainsResolution(unittest.TestCase):
    """text_contains resolves against whichever event carries the text."""

    def _check(self, event: dict, exp: EvalExpectation):
        return _matcher()._check_payload(event, exp, 0, 0)

    def test_on_one_event_text(self):
        exp = EvalExpectation(event="llm_response", text_contains="Paris")
        self.assertIsNone(self._check({"type": "llm_response", "text": "It's Paris."}, exp))
        self.assertIsNotNone(self._check({"type": "llm_response", "text": "London."}, exp))

    def test_on_user_transcription_ignores_spacing(self):
        exp = EvalExpectation(event="user_transcription", text_contains="the capital of")
        ok = {"type": "user_transcription", "transcript": " What  is the  capital  of Germany?"}
        self.assertIsNone(self._check(ok, exp))

    def test_on_user_transcription_transcript(self):
        exp = EvalExpectation(event="user_transcription", text_contains="hello")
        ok = {"type": "user_transcription", "transcript": "hello world"}
        self.assertIsNone(self._check(ok, exp))
        failure = self._check({"type": "user_transcription", "transcript": "bye"}, exp)
        self.assertIsNotNone(failure)
        self.assertIn("does not contain", failure.reason)


class _Collector(FrameProcessor):
    """Records what the processor before it pushes, without a running pipeline."""

    def __init__(self):
        super().__init__()
        self.frames: list = []

    async def queue_frame(self, frame, direction=FrameDirection.DOWNSTREAM, callback=None):
        self.frames.append(frame)


def _bot_response(*texts: str, skip_tts: bool = False) -> list:
    """The frames of one LLM response, as the bot's (or the persona's) LLM emits them."""
    frames = [LLMFullResponseStartFrame(), *[LLMTextFrame(text=t) for t in texts]]
    frames.append(LLMFullResponseEndFrame())
    for frame in frames:
        frame.skip_tts = skip_tts
    return frames


def _persona() -> EvalPersona:
    """A persona whose LLM is a stub: the sink tests only need its context and its hearing."""
    return EvalPersona("A caller.", "A goal.", SimpleNamespace(register_function=lambda *a: None))  # type: ignore[arg-type]


class TestBotFrameSink(unittest.IsolatedAsyncioTestCase):
    """The sink is the boundary between the bot's side of the pipeline and the user's."""

    def _sink(self, **kwargs) -> tuple[_BotFrameSink, _Collector]:
        sink = _BotFrameSink(_stream(bot_audio=True), **kwargs)
        nxt = _Collector()
        sink.link(nxt)
        return sink, nxt

    @staticmethod
    def _events(sink: _BotFrameSink) -> list[str]:
        return [e["type"] for e in sink._stream.events_seen]

    async def _push(self, sink, *frames):
        for frame in frames:
            await sink.process_frame(frame, FrameDirection.DOWNSTREAM)

    async def test_inject_pushes_downstream(self):
        sink, nxt = self._sink()
        frame = TTSSpeakFrame("hello")
        await sink.inject(frame)
        self.assertEqual(nxt.frames, [frame])

    async def test_the_bots_frames_stop_at_the_sink(self):
        sink, nxt = self._sink()
        await self._push(
            sink,
            *_bot_response("Hello"),
            TTSTextFrame(text="Hello", aggregated_by=AggregationType.SENTENCE),
            BotStartedSpeakingFrame(),
            FunctionCallInProgressFrame(function_name="f", tool_call_id="c", arguments={}),
            InputTransportMessageFrame(message={"label": RTVI.MESSAGE_LABEL, "type": "x"}),
        )
        self.assertEqual(nxt.frames, [])

    async def test_a_scripted_run_swallows_the_computed_interruption(self):
        sink, nxt = self._sink()
        await self._push(sink, InterruptionFrame(), LLMContextFrame(LLMContext()))
        self.assertEqual(nxt.frames, [])  # no persona: nothing to interrupt or run

    async def test_a_persona_is_interrupted_and_runs_per_context_frame(self):
        persona = _persona()
        context = persona.context
        sink, nxt = self._sink(persona=persona)
        interruption = InterruptionFrame()
        run = LLMContextFrame(context)
        await self._push(sink, interruption, run)
        self.assertEqual(nxt.frames, [interruption, run])

    async def test_hang_up_silences_the_persona(self):
        persona = _persona()
        context = persona.context
        sink, nxt = self._sink(persona=persona, persona_hears=True)
        persona.hang_up()
        await self._push(sink, LLMContextFrame(context), *_bot_response("Anything else?"))
        self.assertEqual(nxt.frames, [])
        self.assertEqual(context.get_messages(), [])

    async def test_text_feed_hands_the_bots_response_to_the_persona(self):
        persona = _persona()
        context = persona.context
        sink, nxt = self._sink(persona=persona, persona_hears=True)
        await self._push(sink, *_bot_response("Hello ", "there"))
        self.assertEqual(context.get_messages(), [{"role": "user", "content": "Hello there"}])
        self.assertEqual(len(nxt.frames), 1)
        self.assertIsInstance(nxt.frames[0], LLMContextFrame)
        self.assertIs(nxt.frames[0].context, context)

    async def test_text_feed_waits_for_the_bots_function_call(self):
        persona = _persona()
        context = persona.context
        sink, nxt = self._sink(persona=persona, persona_hears=True)
        await self._push(
            sink,
            LLMFullResponseStartFrame(),
            LLMTextFrame(text="Let me check."),
            FunctionCallInProgressFrame(function_name="f", tool_call_id="c", arguments={}),
            LLMFullResponseEndFrame(),
        )
        self.assertEqual(context.get_messages(), [])  # held: the call is still running
        await self._push(
            sink,
            FunctionCallResultFrame(function_name="f", tool_call_id="c", arguments={}, result=1),
            *_bot_response("It's sunny."),
        )
        self.assertEqual(
            context.get_messages(), [{"role": "user", "content": "Let me check. It's sunny."}]
        )
        self.assertEqual(len(nxt.frames), 1)

    async def test_upstream_frames_are_not_the_bots(self):
        # The persona's own function call travels upstream through the sink; it
        # is neither an event nor held back.
        sink, _ = self._sink()
        prev = _Collector()
        prev.link(sink)
        call = FunctionCallInProgressFrame(function_name="end_call", tool_call_id="c", arguments={})
        await sink.process_frame(call, FrameDirection.UPSTREAM)
        self.assertEqual(prev.frames, [call])
        self.assertEqual(sink._stream.events_seen, [])

    async def test_text_feed_ignores_an_empty_response(self):
        persona = _persona()
        context = persona.context
        sink, nxt = self._sink(persona=persona, persona_hears=True)
        await self._push(sink, *_bot_response())
        self.assertEqual(context.get_messages(), [])
        self.assertEqual(nxt.frames, [])


def _untimed(events: list[dict]) -> list[dict]:
    """The events without the arrival time the stream stamps on them."""
    return [{k: v for k, v in e.items() if k != "at"} for e in events]


class TestPersonaTurnRelay(unittest.IsolatedAsyncioTestCase):
    """The persona's text-mode responses become one send-text each."""

    def _relay(self) -> tuple[_PersonaTurnRelay, _Collector, EvalTrace]:
        trace = EvalTrace()
        stream = _stream(bot_audio=True)
        relay = _PersonaTurnRelay(lambda text: {"type": "send-text", "text": text}, trace, stream)
        nxt = _Collector()
        relay.link(nxt)
        return relay, nxt, trace

    async def test_skip_tts_response_is_sent_as_one_text_turn(self):
        relay, nxt, trace = self._relay()
        frames = _bot_response("Hi ", "there.", skip_tts=True)
        for frame in frames:
            await relay.process_frame(frame, FrameDirection.DOWNSTREAM)
        messages = [f for f in nxt.frames if isinstance(f, OutputTransportMessageUrgentFrame)]
        self.assertEqual(
            [m.message for m in messages], [{"type": "send-text", "text": "Hi there."}]
        )
        # The response's own frames go on, for the assistant aggregator.
        self.assertEqual([f for f in nxt.frames if f in frames], frames)
        self.assertTrue(any("'Hi there.' (persona, text)" in line for line in trace.lines))
        self.assertEqual(
            _untimed(relay._stream.events_seen), [{"type": "persona_turn", "text": "Hi there."}]
        )

    async def test_spoken_response_is_only_traced(self):
        relay, nxt, trace = self._relay()
        # In audio mode the TTS speaks the text and forwards the end frame after it.
        for frame in (
            LLMFullResponseStartFrame(),
            LLMTextFrame(text="Hi"),
            TTSTextFrame(text="Hi", aggregated_by=AggregationType.SENTENCE),
            LLMFullResponseEndFrame(),
        ):
            await relay.process_frame(frame, FrameDirection.DOWNSTREAM)
        self.assertFalse(any(isinstance(f, OutputTransportMessageUrgentFrame) for f in nxt.frames))
        self.assertTrue(any("'Hi' (persona, audio)" in line for line in trace.lines))
        self.assertEqual(
            _untimed(relay._stream.events_seen), [{"type": "persona_turn", "text": "Hi"}]
        )

    async def test_a_response_without_words_is_not_a_turn(self):
        # An end_call-only response, or one cut off before any word went out.
        relay, _, _ = self._relay()
        for frame in (LLMFullResponseStartFrame(), LLMFullResponseEndFrame()):
            await relay.process_frame(frame, FrameDirection.DOWNSTREAM)
        for frame in (
            LLMFullResponseStartFrame(),
            InterruptionFrame(),
            LLMFullResponseEndFrame(),
        ):
            await relay.process_frame(frame, FrameDirection.DOWNSTREAM)
        self.assertEqual(relay._stream.events_seen, [])

    async def test_an_interrupted_response_is_not_a_turn(self):
        relay, _, _ = self._relay()
        for frame in (
            LLMFullResponseStartFrame(),
            LLMTextFrame(text="Could you"),
            InterruptionFrame(),
        ):
            await relay.process_frame(frame, FrameDirection.DOWNSTREAM)
        self.assertEqual(relay._stream.events_seen, [])


class TestAudioSender(unittest.IsolatedAsyncioTestCase):
    """User audio is spoken by pushing a TTSSpeakFrame into the pipeline at the sink."""

    async def test_send_user_audio_injects_tts_speak_frame(self):
        s = _client(bot_audio=True)
        injected = _capture_injected(s)
        await s.say("hello world")

        self.assertEqual(len(injected), 1)
        self.assertIsInstance(injected[0], TTSSpeakFrame)
        self.assertEqual(injected[0].text, "hello world")


class TestAudioFileSender(unittest.IsolatedAsyncioTestCase):
    """A turn's ``audio:`` file is played to the bot instead of being synthesized."""

    @staticmethod
    def _write_tone(path, sample_rate=16000, seconds=2.0, channels=1):
        import numpy as np
        import soundfile as sf

        t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
        tone = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        data = tone if channels == 1 else np.column_stack([tone] * channels)
        sf.write(str(path), data, sample_rate)
        return tone

    async def test_file_is_spoken_as_one_tts_utterance_at_its_own_rate(self):
        from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame

        d = Path(tempfile.mkdtemp())
        tone = self._write_tone(d / "hi.wav", sample_rate=16000, seconds=2.0)

        s = _client(bot_audio=True)
        injected = _capture_injected(s)
        await s.play(str(d / "hi.wav"))

        # Bracketed like the user TTS's output, so the output transport flushes
        # the utterance's final partial chunk on the stop frame.
        self.assertEqual(
            [type(f) for f in injected], [TTSStartedFrame, TTSAudioRawFrame, TTSStoppedFrame]
        )
        self.assertEqual(injected[1].sample_rate, 16000)
        self.assertEqual(injected[1].num_channels, 1)
        self.assertEqual(injected[1].audio, tone.tobytes())

    async def test_non_native_rate_is_preserved(self):
        # The frame carries the file's rate (the output transport resamples it),
        # so a recording need not match the bot.
        d = Path(tempfile.mkdtemp())
        self._write_tone(d / "hi.wav", sample_rate=44100, seconds=0.5)

        s = _client(bot_audio=True)
        injected = _capture_injected(s)
        await s.play(str(d / "hi.wav"))

        self.assertEqual(len(injected), 3)
        self.assertEqual(injected[1].sample_rate, 44100)

    async def test_stereo_is_downmixed_to_mono(self):
        d = Path(tempfile.mkdtemp())
        tone = self._write_tone(d / "s.wav", sample_rate=16000, seconds=0.5, channels=2)

        pcm, rate = await load_user_audio(str(d / "s.wav"))
        self.assertEqual(rate, 16000)
        self.assertEqual(pcm, tone.tobytes())

    async def test_unreadable_file_reports_the_path(self):
        d = Path(tempfile.mkdtemp())
        (d / "x.txt").write_text("not audio", encoding="utf-8")
        with self.assertRaises(ValueError) as cm:
            await load_user_audio(str(d / "x.txt"))
        self.assertIn("x.txt", str(cm.exception))


class TestAudioFileEnablesUserAudio(unittest.TestCase):
    def test_audio_turn_enables_user_audio_output(self):
        # The harness streams user audio only when audio output is enabled, so
        # a turn that plays a file must enable it even though the scenario
        # names no TTS.
        scenario = EvalScriptScenario(
            name="t",
            bot_audio=True,
            user_audio=True,
            turns=[EvalScriptTurn(user="hi", audio="hi.wav", expect=[])],
        )
        self.assertTrue(_client(scenario).sends_user_audio)

    def test_text_turns_do_not(self):
        scenario = EvalScriptScenario(
            name="t", bot_audio=True, turns=[EvalScriptTurn(user="hi", expect=[])]
        )
        self.assertFalse(_client(scenario).sends_user_audio)


class TestDTMFSender(unittest.IsolatedAsyncioTestCase):
    """A dtmf turn sends one RTVI ``dtmf`` message with all keys."""

    async def test_send_user_dtmf_single_message_with_all_keys(self):
        s = _client()
        sent: list[RTVI.Message] = []

        async def fake_send(message):
            sent.append(message)

        s.send = fake_send
        await s.send_dtmf("12#")

        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].type, "dtmf")
        self.assertEqual(sent[0].data["buttons"], ["1", "2", "#"])

    async def test_send_user_dtmf_single_key(self):
        s = _client()
        sent: list[RTVI.Message] = []

        async def fake_send(message):
            sent.append(message)

        s.send = fake_send
        await s.send_dtmf("1")

        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].type, "dtmf")
        self.assertEqual(sent[0].data["buttons"], ["1"])


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


class _FakeRTVIServer:
    """A minimal RTVI WebSocket server with scripted replies.

    Auto-replies ``bot-ready`` to ``client-ready``. For each ``send-text``, it
    sends the list of RTVI server messages registered for that content string.
    Records every message it receives for assertions.
    """

    def __init__(self, port: int):
        self.port = port
        self.received: list[dict] = []
        self.script: dict[str, list[str]] = {}
        # Sent right after bot-ready, like a bot that greets on connect.
        self.greeting: list[str] = []
        # After replying to this send-text content, close the connection, like a
        # bot that ends the call.
        self.close_after: str | None = None
        self._server: websockets.WebSocketServer | None = None

    def on_text(self, content: str, *messages: str):
        self.script[content] = list(messages)

    async def _handler(self, ws):
        speaking = False
        async for raw in ws:
            msg = json.loads(raw)
            self.received.append(msg)
            match msg.get("type"):
                case "client-ready":
                    await ws.send(_rtvi("bot-ready", {"version": RTVI.PROTOCOL_VERSION}))
                    for out in self.greeting:
                        await ws.send(out)
                case "send-text":
                    for out in self.script.get(msg["data"]["content"], []):
                        await ws.send(out)
                    if msg["data"]["content"] == self.close_after:
                        await ws.close()
                        return
                case "raw-audio":
                    # The harness streams the user's side continuously (silence
                    # around each utterance). A real bot transcribes the audio and
                    # answers once the user stops speaking; the fake one answers the
                    # single scripted reply on the first silent frame after speech.
                    if any(base64.b64decode(msg["data"]["base64Audio"])):
                        speaking = True
                    elif speaking:
                        speaking = False
                        for out in next(iter(self.script.values()), []):
                            await ws.send(out)

    async def start(self):
        self._server = await websockets.serve(self._handler, "localhost", self.port)

    async def stop(self):
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    @property
    def url(self) -> str:
        return f"ws://localhost:{self.port}"


def _capture_deadlines(session: EvalScriptSession) -> list[float]:
    """Record the deadline every expectation in a run is matched against.

    A deadline is ``anchor + within_ms``, so expectations anchored together
    produce one identical float rather than a float per expectation.
    """
    deadlines: list[float] = []
    match = session._driver._matcher.match

    async def capture(expectation, anchor, budget_ms, turn_idx, exp_idx):
        deadlines.append(anchor + budget_ms / 1000.0)
        return await match(expectation, anchor, budget_ms, turn_idx, exp_idx)

    session._driver._matcher.match = capture
    return deadlines


class TestEvalsHarnessIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.server = _FakeRTVIServer(_free_port())
        await self.server.start()

    async def asyncTearDown(self):
        await self.server.stop()

    async def test_one_event_pass(self):
        self.server.on_text(
            "what is the capital of France?",
            _rtvi("bot-llm-started"),
            _rtvi("bot-llm-text", {"text": "Paris"}),
            _rtvi("bot-llm-stopped"),
        )
        scenario = EvalScriptScenario(
            name="capital",
            bot_audio=False,
            turns=[
                EvalScriptTurn(
                    user="what is the capital of France?",
                    expect=[
                        EvalExpectation(event="llm_started", within_ms=2000),
                        EvalExpectation(
                            event="llm_response", within_ms=2000, text_contains="Paris"
                        ),
                    ],
                )
            ],
        )
        result = await EvalScriptSession.from_scenario(scenario, self.server.url).run()
        self.assertTrue(result.passed, f"failures: {[str(f) for f in result.failures]}")

    async def test_one_event_aggregates_past_filler(self):
        # First bot response is filler; the answer arrives in a second segment.
        # text_contains aggregates across both and matches.
        self.server.on_text(
            "weather?",
            _rtvi("bot-llm-started"),
            _rtvi("bot-llm-text", {"text": "Let me check on that. "}),
            _rtvi("bot-llm-stopped"),
            _rtvi("bot-llm-started"),
            _rtvi("bot-llm-text", {"text": "It is sunny in Paris."}),
            _rtvi("bot-llm-stopped"),
        )
        scenario = EvalScriptScenario(
            name="filler",
            bot_audio=False,
            turns=[
                EvalScriptTurn(
                    user="weather?",
                    expect=[
                        EvalExpectation(event="llm_response", within_ms=2000, text_contains="Paris")
                    ],
                )
            ],
        )
        result = await EvalScriptSession.from_scenario(scenario, self.server.url).run()
        self.assertTrue(result.passed, f"failures: {[str(f) for f in result.failures]}")

    async def test_user_transcription_aggregates_pieces(self):
        # An STT that finalizes an utterance in pieces emits one user-transcription
        # per piece; text_contains accumulates the finals (skipping interims), so a
        # phrase spanning pieces matches however each piece is spaced.
        self.server.on_text(
            "what is the capital of Germany?",
            _rtvi("user-transcription", {"text": " What is", "final": False}),
            _rtvi("user-transcription", {"text": " What", "final": True}),
            _rtvi("user-transcription", {"text": " is the capital", "final": True}),
            _rtvi("user-transcription", {"text": " of", "final": True}),
            _rtvi("user-transcription", {"text": " Germany?", "final": True}),
            _rtvi("bot-llm-started"),
            _rtvi("bot-llm-text", {"text": "Berlin."}),
            _rtvi("bot-llm-stopped"),
        )
        scenario = EvalScriptScenario(
            name="pieces",
            turns=[
                EvalScriptTurn(
                    user="what is the capital of Germany?",
                    expect=[
                        EvalExpectation(
                            event="user_transcription",
                            within_ms=2000,
                            text_contains="capital of Germany",
                        ),
                        EvalExpectation(
                            event="llm_response", within_ms=2000, text_contains="Berlin"
                        ),
                    ],
                )
            ],
        )
        result = await EvalScriptSession.from_scenario(scenario, self.server.url).run()
        self.assertTrue(result.passed, f"failures: {[str(f) for f in result.failures]}")

    async def test_function_call_pass(self):
        self.server.on_text(
            "weather in Paris?",
            _rtvi(
                "llm-function-call-in-progress",
                {
                    "function_name": "get_weather",
                    "arguments": {"city": "Paris"},
                    "tool_call_id": "1",
                },
            ),
        )
        scenario = EvalScriptScenario(
            name="tool",
            turns=[
                EvalScriptTurn(
                    user="weather in Paris?",
                    expect=[
                        EvalExpectation(
                            event="function_call",
                            within_ms=2000,
                            calls=[EvalFunctionCall(name="get_weather", args={"city": "Paris"})],
                        ),
                    ],
                )
            ],
        )
        result = await EvalScriptSession.from_scenario(scenario, self.server.url).run()
        self.assertTrue(result.passed, f"failures: {[str(f) for f in result.failures]}")

    def _cancel_scenario(self, expected_id: str) -> EvalScriptScenario:
        """One turn asserting a cancel call named a particular id."""
        return EvalScriptScenario(
            name="cancel",
            turns=[
                EvalScriptTurn(
                    user="never mind that one",
                    expect=[
                        EvalExpectation(
                            event="function_call",
                            within_ms=2000,
                            calls=[
                                EvalFunctionCall(
                                    name="cancel_write_report",
                                    args={"tool_call_id": expected_id},
                                )
                            ],
                        ),
                    ],
                ),
            ],
        )

    async def test_corrected_call_satisfies_the_turn(self):
        # A model that mistypes an id, is refused, and repeats the call with the
        # right one has made the call the turn asks for.
        self.server.on_text(
            "never mind that one",
            _rtvi(
                "llm-function-call-in-progress",
                {
                    "function_name": "cancel_write_report",
                    "arguments": {"tool_call_id": "cull_turtles"},
                    "tool_call_id": "call_cancel_1",
                },
            ),
            _rtvi(
                "llm-function-call-in-progress",
                {
                    "function_name": "cancel_write_report",
                    "arguments": {"tool_call_id": "call_turtles"},
                    "tool_call_id": "call_cancel_2",
                },
            ),
        )
        result = await EvalScriptSession.from_scenario(
            self._cancel_scenario("call_turtles"), self.server.url
        ).run()
        self.assertTrue(result.passed, f"failures: {[str(f) for f in result.failures]}")

    async def test_wrong_args_reports_what_the_bot_sent(self):
        # No call carried the expected id, so the failure names the ones that did
        # arrive rather than claiming the call was never made.
        self.server.on_text(
            "never mind that one",
            _rtvi(
                "llm-function-call-in-progress",
                {
                    "function_name": "cancel_write_report",
                    "arguments": {"tool_call_id": "call_volcanoes"},
                    "tool_call_id": "call_cancel",
                },
            ),
        )
        result = await EvalScriptSession.from_scenario(
            self._cancel_scenario("call_turtles"), self.server.url
        ).run()
        self.assertFalse(result.passed)
        self.assertEqual(result.failures[0].kind, "function_args_mismatch")
        self.assertIn("call_volcanoes", str(result.failures[0]))

    def _stopped_scenario(self, cancelled: bool) -> EvalScriptScenario:
        """One turn asserting a call stopped, and how it ended."""
        return EvalScriptScenario(
            name="stopped",
            turns=[
                EvalScriptTurn(
                    user="never mind that one",
                    expect=[
                        EvalExpectation(
                            event="function_call_stopped",
                            within_ms=2000,
                            calls=[
                                EvalFunctionCall(
                                    name="write_report",
                                    args={"tool_call_id": "call_turtles", "cancelled": cancelled},
                                )
                            ],
                        ),
                    ],
                ),
            ],
        )

    async def _run_stopped(self, reported_cancelled: bool, expect_cancelled: bool):
        self.server.on_text(
            "never mind that one",
            _rtvi(
                "llm-function-call-stopped",
                {
                    "function_name": "write_report",
                    "tool_call_id": "call_turtles",
                    "cancelled": reported_cancelled,
                },
            ),
        )
        return await EvalScriptSession.from_scenario(
            self._stopped_scenario(expect_cancelled), self.server.url
        ).run()

    async def test_cancelled_call_matches(self):
        result = await self._run_stopped(reported_cancelled=True, expect_cancelled=True)
        self.assertTrue(result.passed, f"failures: {[str(f) for f in result.failures]}")

    async def test_call_that_ran_to_completion_is_not_a_cancellation(self):
        # The distinction the event exists for: work that finished on its own
        # stops too, and a scenario asserting cancellation must not accept it.
        result = await self._run_stopped(reported_cancelled=False, expect_cancelled=True)
        self.assertFalse(result.passed)
        self.assertEqual(result.failures[0].kind, "function_args_mismatch")

    async def test_started_and_stopped_events_do_not_claim_each_other(self):
        # Both carry a name and a tool_call_id, so an expectation for one must
        # not be satisfied by the other.
        self.server.on_text(
            "report on sea turtles",
            _rtvi(
                "llm-function-call-in-progress",
                {
                    "function_name": "write_report",
                    "arguments": {"topic": "sea turtles"},
                    "tool_call_id": "call_turtles",
                },
            ),
        )
        scenario = EvalScriptScenario(
            name="no_crosstalk",
            turns=[
                EvalScriptTurn(
                    user="report on sea turtles",
                    expect=[
                        EvalExpectation(
                            event="function_call_stopped",
                            within_ms=1500,
                            calls=[EvalFunctionCall(name="write_report")],
                        ),
                    ],
                )
            ],
        )
        result = await EvalScriptSession.from_scenario(scenario, self.server.url).run()
        self.assertFalse(result.passed)
        self.assertEqual(result.failures[0].kind, "missing_function_call")

    async def test_text_mismatch_fails_clearly(self):
        self.server.on_text(
            "hi",
            _rtvi("bot-llm-started"),
            _rtvi("bot-llm-text", {"text": "Paris"}),
            _rtvi("bot-llm-stopped"),
        )
        scenario = EvalScriptScenario(
            name="mismatch",
            bot_audio=False,
            turns=[
                EvalScriptTurn(
                    user="hi",
                    expect=[
                        EvalExpectation(
                            event="llm_response", within_ms=2000, text_contains="London"
                        )
                    ],
                )
            ],
        )
        result = await EvalScriptSession.from_scenario(scenario, self.server.url).run()
        self.assertFalse(result.passed)
        self.assertEqual(len(result.failures), 1)
        self.assertIn("does not contain", result.failures[0].reason)

    def _two_turn_first_fails(self, *, stop_on_failure: bool) -> EvalScriptScenario:
        """A scenario whose first turn fails on content and whose second passes."""
        self.server.on_text(
            "first",
            _rtvi("bot-llm-started"),
            _rtvi("bot-llm-text", {"text": "Paris"}),
            _rtvi("bot-llm-stopped"),
        )
        self.server.on_text(
            "second",
            _rtvi("bot-llm-started"),
            _rtvi("bot-llm-text", {"text": "Berlin"}),
            _rtvi("bot-llm-stopped"),
        )
        return EvalScriptScenario(
            name="stop",
            bot_audio=False,
            stop_on_failure=stop_on_failure,
            turns=[
                EvalScriptTurn(
                    user="first",
                    expect=[
                        EvalExpectation(event="llm_response", within_ms=300, text_contains="London")
                    ],
                ),
                EvalScriptTurn(
                    user="second",
                    expect=[
                        EvalExpectation(
                            event="llm_response", within_ms=2000, text_contains="Berlin"
                        )
                    ],
                ),
            ],
        )

    def _sent_texts(self) -> list[str]:
        return [m["data"]["content"] for m in self.server.received if m.get("type") == "send-text"]

    async def test_failed_turn_stops_scenario_by_default(self):
        scenario = self._two_turn_first_fails(stop_on_failure=True)
        result = await EvalScriptSession.from_scenario(scenario, self.server.url).run()
        self.assertFalse(result.passed)
        # Only turn 0 is reported, and turn 1 is never sent.
        self.assertEqual([f.turn_index for f in result.failures], [0])
        self.assertEqual(self._sent_texts(), ["first"])
        # The turn the run stopped short of is not_run, not a pass.
        self.assertEqual([t.status for t in result.turns], ["failed", "not_run"])

    async def test_stop_on_failure_false_drives_remaining_turns(self):
        scenario = self._two_turn_first_fails(stop_on_failure=False)
        result = await EvalScriptSession.from_scenario(scenario, self.server.url).run()
        # The run still fails, but every turn was driven and the passing turn
        # after the failure adds no failure of its own.
        self.assertFalse(result.passed)
        self.assertEqual([f.turn_index for f in result.failures], [0])
        self.assertEqual(self._sent_texts(), ["first", "second"])
        self.assertEqual([t.status for t in result.turns], ["failed", "passed"])

    async def test_turn_results_carry_their_own_failures(self):
        scenario = self._two_turn_first_fails(stop_on_failure=False)
        result = await EvalScriptSession.from_scenario(scenario, self.server.url).run()
        failed, passed = result.turns
        self.assertEqual([f.kind for f in failed.failures], ["text_mismatch"])
        self.assertEqual(passed.failures, [])
        # Every turn's failures, in order, are the flat list on the result.
        self.assertEqual([f for t in result.turns for f in t.failures], result.failures)

    async def test_turn_results_are_timed(self):
        scenario = self._two_turn_first_fails(stop_on_failure=False)
        result = await EvalScriptSession.from_scenario(scenario, self.server.url).run()
        # A driven turn is timed; one that never ran has no duration to report.
        self.assertGreater(result.turns[0].duration_ms, 0)
        self.assertGreater(result.turns[1].duration_ms, 0)

        stopping = await EvalScriptSession.from_scenario(
            self._two_turn_first_fails(stop_on_failure=True), self.server.url
        ).run()
        self.assertEqual(stopping.turns[1].duration_ms, 0)

    async def test_missing_event_times_out(self):
        scenario = EvalScriptScenario(
            name="never",
            turns=[
                EvalScriptTurn(
                    user="hi", expect=[EvalExpectation(event="llm_response", within_ms=200)]
                )
            ],
        )
        result = await EvalScriptSession.from_scenario(scenario, self.server.url).run()
        self.assertFalse(result.passed)
        self.assertEqual(len(result.failures), 1)
        self.assertIn("arrived within", result.failures[0].reason)
        self.assertIn("200ms", result.failures[0].reason)

    async def test_subsequent_assertions_skipped_after_timeout(self):
        scenario = EvalScriptScenario(
            name="cascading",
            turns=[
                EvalScriptTurn(
                    user="hi",
                    expect=[
                        EvalExpectation(event="llm_started", within_ms=100),
                        EvalExpectation(event="llm_response", within_ms=100),
                        EvalExpectation(event="function_call", within_ms=100),
                    ],
                )
            ],
        )
        result = await EvalScriptSession.from_scenario(scenario, self.server.url).run()
        self.assertFalse(result.passed)
        self.assertEqual(len(result.failures), 1, "only the first failed expectation should report")

    async def test_turn_shares_one_deadline_across_expectations(self):
        # A turn that expects a function call AND a response but gets neither must
        # fail within a single within_ms budget. The function_call timeout returns a
        # failure (not a raise) so the loop continues to the response; both share the
        # anchor, so the turn spends one budget, not budget-per-expectation.
        scenario = EvalScriptScenario(
            name="shared_deadline",
            turns=[
                EvalScriptTurn(
                    user="weather?",  # no scripted reply -> nothing arrives
                    expect=[
                        EvalExpectation(
                            event="function_call",
                            within_ms=400,
                            calls=[EvalFunctionCall(name="get_weather")],
                        ),
                        EvalExpectation(event="llm_response", within_ms=400),
                    ],
                )
            ],
        )
        session = EvalScriptSession.from_scenario(scenario, self.server.url)
        deadlines = _capture_deadlines(session)
        result = await session.run()
        self.assertFalse(result.passed)
        # Both expectations wait against one 400ms deadline, not 400ms each.
        self.assertEqual(len(deadlines), 2)
        self.assertEqual(len(set(deadlines)), 1)

    async def test_send_after_delays_run(self):
        self.server.on_text("first", _rtvi("bot-llm-started"), _rtvi("bot-llm-stopped"))
        self.server.on_text(
            "second",
            _rtvi("bot-llm-started"),
            _rtvi("bot-llm-text", {"text": "ok"}),
            _rtvi("bot-llm-stopped"),
        )
        scenario = EvalScriptScenario(
            name="send_after",
            bot_audio=False,
            turns=[
                EvalScriptTurn(
                    user="first", expect=[EvalExpectation(event="llm_started", within_ms=2000)]
                ),
                EvalScriptTurn(
                    user="second",
                    expect=[EvalExpectation(event="llm_response", within_ms=2000)],
                    send_after=EvalSendAfter(event="llm_started", delay_ms=200),
                ),
            ],
        )
        result = await EvalScriptSession.from_scenario(scenario, self.server.url).run()
        self.assertTrue(result.passed, f"failures: {[str(f) for f in result.failures]}")
        self.assertGreaterEqual(result.duration_ms, 200)

    async def test_connect_failure_reported_cleanly(self):
        scenario = EvalScriptScenario(
            name="no_bot",
            turns=[EvalScriptTurn(user="x", expect=[EvalExpectation(event="llm_started")])],
        )
        result = await EvalScriptSession.from_scenario(
            scenario,
            f"ws://localhost:{_free_port()}",
            params=EvalSessionParams(connect_timeout_s=0.5),
        ).run()
        self.assertFalse(result.passed)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].event_name, "<connect>")
        self.assertIn("failed to connect", result.failures[0].reason)
        # A run that never reached the bot scored nothing.
        self.assertEqual([t.status for t in result.turns], ["not_run"])

    async def test_unexpected_error_surfaced_not_swallowed(self):
        # An unexpected error mid-run (here a judge raising) must be reported as a
        # structured failure with its traceback, not propagate out raw and get
        # swallowed as a bare "error:" with no eval.log.
        class _BoomJudge:
            def add_user_message(self, text):
                raise RuntimeError("judge boom")

            def add_assistant_message(self, text):
                pass

            async def evaluate(self, criterion):
                raise AssertionError("unreachable")

        scenario = EvalScriptScenario(
            name="boom",
            turns=[EvalScriptTurn(user="hi", expect=[EvalExpectation(event="llm_started")])],
        )
        result = await EvalScriptSession.from_scenario(
            scenario, self.server.url, judge=_BoomJudge()
        ).run()
        self.assertFalse(result.passed)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].event_name, "<error>")
        self.assertIn("RuntimeError: judge boom", result.failures[0].reason)
        # The full traceback is preserved in the debug trace (saved to <bot>.eval.log).
        self.assertTrue(any("judge boom" in line for line in result.debug_log))
        self.assertTrue(any("Traceback" in line for line in result.debug_log))
        # The raise came from inside the turn, so it is scored as that turn's failure.
        self.assertEqual([t.status for t in result.turns], ["failed"])
        self.assertEqual(result.turns[0].failures, [result.failures[0]])

    async def test_audio_turn_sends_the_file_not_synthesized_text(self):
        import numpy as np
        import soundfile as sf

        d = Path(tempfile.mkdtemp())
        sr = 16000
        t = np.linspace(0, 0.5, sr // 2, endpoint=False)
        tone = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        sf.write(str(d / "hi.wav"), tone, sr)

        self.server.on_text(
            "hello there",
            _rtvi("bot-llm-started"),
            _rtvi("bot-llm-text", {"text": "Hi!"}),
            _rtvi("bot-llm-stopped"),
        )
        scenario = EvalScriptScenario(
            name="audio-turn",
            bot_audio=False,
            user_audio=True,
            turns=[
                EvalScriptTurn(
                    user="hello there",
                    audio=str(d / "hi.wav"),
                    expect=[EvalExpectation(event="llm_started", within_ms=2000)],
                )
            ],
        )

        # No user_tts= override: the file is played, so nothing has to synthesize it.
        result = await EvalScriptSession.from_scenario(scenario, self.server.url).run()

        self.assertTrue(result.passed, f"failures: {[str(f) for f in result.failures]}")
        audio_msgs = [m for m in self.server.received if m.get("type") == "raw-audio"]
        self.assertTrue(audio_msgs, "the turn's audio file never reached the bot")
        self.assertEqual(
            [m for m in self.server.received if m.get("type") == "send-text"],
            [],
            "an audio turn must not also be sent as text",
        )
        # The user's side is streamed continuously, with silence before and after
        # the turn, so the recording is one contiguous run inside that stream.
        got = b"".join(base64.b64decode(m["data"]["base64Audio"]) for m in audio_msgs)
        self.assertIn(tone.tobytes(), got)
        self.assertTrue(all(m["data"]["sampleRate"] == sr for m in audio_msgs))

    async def test_context_sends_eval_context_message(self):
        self.server.on_text("hi", _rtvi("bot-llm-started"), _rtvi("bot-llm-stopped"))
        scenario = EvalScriptScenario(
            name="context",
            turns=[
                EvalScriptTurn(
                    user="hi", expect=[EvalExpectation(event="llm_started", within_ms=2000)]
                )
            ],
            context=[{"role": "system", "content": "be terse"}],
        )
        result = await EvalScriptSession.from_scenario(scenario, self.server.url).run()
        self.assertTrue(result.passed, f"failures: {[str(f) for f in result.failures]}")
        context_messages = [
            m
            for m in self.server.received
            if m.get("type") == "client-message" and m["data"].get("t") == "eval-context"
        ]
        self.assertEqual(len(context_messages), 1)
        self.assertEqual(
            context_messages[0]["data"]["d"]["messages"],
            [{"role": "system", "content": "be terse"}],
        )

    async def test_no_eval_context_message_when_empty(self):
        self.server.on_text("hi", _rtvi("bot-llm-started"), _rtvi("bot-llm-stopped"))
        scenario = EvalScriptScenario(
            name="nocontext",
            turns=[
                EvalScriptTurn(
                    user="hi", expect=[EvalExpectation(event="llm_started", within_ms=2000)]
                )
            ],
        )
        await EvalScriptSession.from_scenario(scenario, self.server.url).run()
        context_messages = [
            m
            for m in self.server.received
            if m.get("type") == "client-message" and m["data"].get("t") == "eval-context"
        ]
        self.assertEqual(context_messages, [])


class TestProgressEvent(unittest.IsolatedAsyncioTestCase):
    """``on_progress`` handlers see every turn and expectation, in order."""

    def _scenario(self) -> EvalScriptScenario:
        return EvalScriptScenario(
            name="progress",
            turns=[
                EvalScriptTurn(
                    user="hi",
                    expect=[
                        EvalExpectation(event="llm_started", within_ms=2000),
                        EvalExpectation(event="llm_response", within_ms=2000),
                    ],
                )
            ],
        )

    async def asyncSetUp(self):
        self.server = _FakeRTVIServer(_free_port())
        await self.server.start()
        self.server.on_text(
            "hi",
            _rtvi("bot-llm-started"),
            _rtvi("bot-llm-text", {"text": "hello"}),
            _rtvi("bot-llm-stopped"),
        )

    async def asyncTearDown(self):
        await self.server.stop()

    async def test_event_handler_receives_records(self):
        session = EvalScriptSession.from_scenario(self._scenario(), self.server.url)
        seen = []

        @session.event_handler("on_progress")
        async def on_progress(source, progress):
            seen.append((source, progress))

        await session.run()

        self.assertTrue(all(source is session for source, _ in seen))
        self.assertEqual(
            [(p.event_name, p.status) for _, p in seen],
            [("hi", "turn"), ("llm_started", "matched"), ("llm_response", "matched")],
        )

    async def test_records_are_delivered_before_run_returns(self):
        """Handlers dispatch as tasks, so run() waits them out before it returns."""
        session = EvalScriptSession.from_scenario(self._scenario(), self.server.url)
        finished = []

        @session.event_handler("on_progress")
        async def on_progress(source, progress):
            await asyncio.sleep(0.01)
            finished.append(progress.event_name)

        await session.run()

        self.assertEqual(finished, ["hi", "llm_started", "llm_response"])

    async def test_callback_is_deprecated_and_still_called(self):
        seen = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            session = EvalScriptSession.from_scenario(
                self._scenario(), self.server.url, on_progress=seen.append
            )
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, DeprecationWarning)

        await session.run()

        # The callback takes only the record, not the session an event handler gets.
        self.assertEqual(
            [(p.event_name, p.status) for p in seen],
            [("hi", "turn"), ("llm_started", "matched"), ("llm_response", "matched")],
        )


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# A simulation end to end: the persona LLM in the pipeline talks to the fake bot.
# ---------------------------------------------------------------------------

from pipecat.evals.judge import JudgeVerdict, RunVerdicts  # noqa: E402
from pipecat.evals.scenario import EvalSimulationMetric, EvalSimulationScenario  # noqa: E402
from pipecat.evals.session import EvalSession, EvalSessionParams  # noqa: E402
from pipecat.evals.simulation_session import EvalSimulationSession  # noqa: E402
from pipecat.frames.frames import FunctionCallFromLLM  # noqa: E402
from pipecat.services.llm_service import LLMService  # noqa: E402
from pipecat.services.settings import LLMSettings  # noqa: E402


class _ScriptedPersonaLLM(LLMService):
    """A persona LLM that answers the bot's last message from a script.

    A script entry is the persona's reply text, or ``("end_call", arguments)``
    to call the tool the way a real model would.
    """

    def __init__(self, script: dict):
        super().__init__(settings=LLMSettings(model="scripted"))
        self._script = script
        self.seen: list[str] = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return
        last = [
            m
            for m in frame.context.get_messages()
            if isinstance(m, dict) and m.get("role") == "user"
        ][-1]["content"]
        self.seen.append(last)
        reply = self._script[last]
        if isinstance(reply, tuple):
            _, arguments = reply
            await self.run_function_calls(
                [
                    FunctionCallFromLLM(
                        function_name="end_call",
                        tool_call_id="c1",
                        arguments=arguments,
                        context=frame.context,
                    )
                ]
            )
            return
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame(text=reply))
        await self.push_frame(LLMFullResponseEndFrame())


class _YesJudge:
    def __init__(self):
        self.transcript: list[dict] = []
        self.criteria: list[str] = []
        self.run_criteria: dict[str, str] = {}

    async def evaluate_run(self, transcript, criteria, success):
        self.transcript = list(transcript)
        self.criteria.append(success)
        self.run_criteria = dict(criteria)
        turns = sum(1 for e in transcript if e["role"] == "assistant")
        yes = JudgeVerdict(verdict="yes", reason="", raw_response="")
        return RunVerdicts(goal=yes, turns={name: [yes] * turns for name in criteria})


class TestSimulationIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.server = _FakeRTVIServer(_free_port())
        await self.server.start()

    async def asyncTearDown(self):
        await self.server.stop()

    async def test_text_simulation_runs_to_end_call(self):
        self.server.greeting = [
            _rtvi("bot-llm-started"),
            _rtvi("bot-llm-text", {"text": "Hello! How can I help?"}),
            _rtvi("bot-llm-stopped"),
        ]
        self.server.on_text(
            "What is the capital of Germany?",
            _rtvi("bot-llm-started"),
            _rtvi("bot-llm-text", {"text": "The capital of Germany is Berlin."}),
            _rtvi("bot-llm-stopped"),
        )
        persona = _ScriptedPersonaLLM(
            {
                "Hello! How can I help?": "What is the capital of Germany?",
                "The capital of Germany is Berlin.": (
                    "end_call",
                    {"success": True, "reason": "I learned it"},
                ),
            }
        )
        simulation = EvalSimulationScenario(
            name="capital",
            persona="A curious traveler.",
            goal="Learn the capital of Germany.",
            simulator={"service": "scripted"},
            success="the bot named Berlin",
            metrics=[EvalSimulationMetric("politeness", "stayed polite")],
            max_turns=5,
            max_duration_s=10.0,
        )
        judge = _YesJudge()

        result = await EvalSimulationSession(
            simulation, self.server.url, persona_llm=persona, judge=judge
        ).run()

        self.assertIsNone(result.error, result.debug_log)
        self.assertTrue(result.succeeded)
        self.assertEqual(result.ended_by, "end_call")
        self.assertEqual(result.end_call, {"success": True, "reason": "I learned it"})
        self.assertEqual(result.turns, 1)  # the question; end_call is not a turn
        # The persona's question went to the bot as one text turn, not spoken.
        sent = [m for m in self.server.received if m.get("type") == "send-text"]
        self.assertEqual([m["data"]["content"] for m in sent], ["What is the capital of Germany?"])
        self.assertFalse(sent[0]["data"]["options"]["audio_response"])
        # The judge saw the whole conversation, persona as the user.
        self.assertEqual(
            result.messages,
            [
                {"role": "assistant", "content": "Hello! How can I help?"},
                {"role": "user", "content": "What is the capital of Germany?"},
                {"role": "assistant", "content": "The capital of Germany is Berlin."},
            ],
        )
        self.assertEqual(judge.criteria, ["the bot named Berlin"])
        self.assertEqual(judge.run_criteria, {"politeness": "stayed polite"})
        # Two bot turns, the greeting and the answer, reached the judge.
        self.assertEqual(sum(1 for e in judge.transcript if e["role"] == "assistant"), 2)

    async def test_the_bot_hanging_up_ends_the_run(self):
        self.server.greeting = [
            _rtvi("bot-llm-started"),
            _rtvi("bot-llm-text", {"text": "How many in your party?"}),
            _rtvi("bot-llm-stopped"),
        ]
        self.server.on_text(
            "Two, please.",
            _rtvi("bot-llm-started"),
            _rtvi("bot-llm-text", {"text": "Booked. Goodbye!"}),
            _rtvi("bot-llm-stopped"),
        )
        self.server.close_after = "Two, please."
        persona = _ScriptedPersonaLLM(
            {"How many in your party?": "Two, please.", "Booked. Goodbye!": "Thanks!"}
        )
        simulation = EvalSimulationScenario(
            name="table",
            persona="A diner.",
            goal="Book a table for two.",
            simulator={"service": "scripted"},
            success="a table was booked",
            max_turns=5,
            max_duration_s=10.0,
        )

        result = await EvalSimulationSession(
            simulation, self.server.url, persona_llm=persona, judge=_YesJudge()
        ).run()

        self.assertIsNone(result.error, result.debug_log)
        self.assertEqual(result.ended_by, "bot")
        self.assertTrue(result.succeeded)


class TestSessionFromScenario(unittest.TestCase):
    """``EvalSession.from_scenario`` builds the session of the scenario's kind."""

    def _script(self) -> EvalScriptScenario:
        return EvalScriptScenario(
            name="capital",
            bot_audio=False,
            turns=[EvalScriptTurn(user="hi", expect=[EvalExpectation(event="llm_started")])],
        )

    def _simulation(self) -> EvalSimulationScenario:
        return EvalSimulationScenario(
            name="capital",
            persona="A curious traveler.",
            goal="Learn the capital of Germany.",
            simulator={"service": "scripted"},
            success="the bot named Berlin",
            bot_audio=False,
        )

    def test_script_scenario_gets_a_script_session(self):
        session = EvalSession.from_scenario(self._script(), "ws://localhost:0")
        self.assertIsInstance(session, EvalScriptSession)

    def test_simulation_gets_a_simulation_session(self):
        session = EvalSession.from_scenario(
            self._simulation(),
            "ws://localhost:0",
            persona_llm=_ScriptedPersonaLLM({}),
            judge=_YesJudge(),
        )
        self.assertIsInstance(session, EvalSimulationSession)

    def test_params_reach_the_driver_and_the_client(self):
        params = EvalSessionParams(default_timeout_ms=1234, trigger_disconnect=True)
        session = EvalSession.from_scenario(self._script(), "ws://localhost:0", params=params)
        self.assertEqual(session._driver._default_timeout_ms, 1234)
        self.assertIn("trigger_disconnect=true", session._client._connect_url())

    def test_deprecated_knobs_fold_into_params(self):
        # A knob passed by its old name wins over the params field of the same
        # name, and the others keep the params object's values.
        for build in (EvalSession.from_scenario, EvalScriptSession.from_scenario):
            with self.subTest(build=build.__qualname__):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    session = build(
                        self._script(),
                        "ws://localhost:0",
                        params=EvalSessionParams(stop_bot=False, use_cache=False),
                        stop_bot=True,
                        default_timeout_ms=1234,
                    )
                self.assertEqual([w.category for w in caught], [DeprecationWarning])
                self.assertIn("`default_timeout_ms`, `stop_bot` of", str(caught[0].message))
                self.assertTrue(session._params.stop_bot)
                self.assertEqual(session._params.default_timeout_ms, 1234)
                self.assertFalse(session._params.use_cache)

    def test_deprecated_progress_callback_reaches_a_script_session(self):
        seen = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            session = EvalSession.from_scenario(
                self._script(), "ws://localhost:0", on_progress=seen.append
            )
        self.assertEqual([w.category for w in caught], [DeprecationWarning])
        self.assertIsInstance(session, EvalScriptSession)
        with self.assertRaises(ValueError):
            EvalSession.from_scenario(
                self._simulation(),
                "ws://localhost:0",
                persona_llm=_ScriptedPersonaLLM({}),
                judge=_YesJudge(),
                on_progress=seen.append,
            )

    def test_persona_llm_is_rejected_for_a_script(self):
        with self.assertRaises(ValueError):
            EvalSession.from_scenario(
                self._script(), "ws://localhost:0", persona_llm=_ScriptedPersonaLLM({})
            )

    def test_other_objects_are_rejected(self):
        with self.assertRaises(TypeError):
            EvalSession.from_scenario({"name": "x"}, "ws://localhost:0")  # type: ignore[arg-type]
