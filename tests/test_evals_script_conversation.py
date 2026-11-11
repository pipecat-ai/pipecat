#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Scripted conversation provenance, partial output, and result compatibility."""

import asyncio
import time
import unittest
from collections import deque
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pipecat.evals._script_conversation import _ScriptConversation
from pipecat.evals.client import (
    EvalClient,
    EvalClientParams,
    _BotSegmentTranscriptionFrame,
    _BotSpeechGate,
)
from pipecat.evals.events import EvalEventStream, EvalHarnessError
from pipecat.evals.results import EvalScriptResult, EvalTrace
from pipecat.evals.script import (
    EvalExpectation,
    EvalFunctionCall,
    EvalScriptScenario,
    EvalScriptTurn,
    EvalSendAfter,
)
from pipecat.evals.script_session import EvalScriptSession
from pipecat.evals.serializer import EvalFunctionCallCancelFrame
from pipecat.evals.session import EvalSessionParams
from pipecat.evals.suite import _result_from_dict
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    ErrorFrame,
    FunctionCallInProgressFrame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TextType,
    TranscriptionFrame,
    TTSTextFrame,
)
from pipecat.services.stt_service import SegmentedSTTService, STTService
from pipecat.tests.utils import run_test
from tests.test_evals_session import _FakeRTVIServer, _free_port, _rtvi


def _reply(conversation: _ScriptConversation, *parts: str) -> None:
    conversation.observe(LLMFullResponseStartFrame())
    for part in parts:
        conversation.observe(LLMTextFrame(text=part))
    conversation.observe(LLMFullResponseEndFrame())


class TestScriptConversation(unittest.TestCase):
    def test_greeting_multiturn_and_exact_token_concatenation(self):
        conversation = _ScriptConversation(bot_audio=False)
        _reply(conversation, "Hello", ", welcome!")
        with conversation.submitting("Book a table"):
            _reply(conversation, "For how ", "many?")
        with conversation.submitting("Two"):
            _reply(conversation, "Booked.")
        self.assertEqual(
            conversation.snapshot(),
            [
                {"role": "assistant", "content": "Hello, welcome!"},
                {"role": "user", "content": "Book a table"},
                {"role": "assistant", "content": "For how many?"},
                {"role": "user", "content": "Two"},
                {"role": "assistant", "content": "Booked."},
            ],
        )

    def test_interrupted_response_retains_ownership_across_late_tokens(self):
        conversation = _ScriptConversation(bot_audio=False)
        with conversation.submitting("First question"):
            pass
        conversation.observe(LLMFullResponseStartFrame())
        conversation.observe(LLMTextFrame(text="Old "))
        with conversation.submitting("Interrupting question"):
            pass
        conversation.observe(LLMTextFrame(text="tail."))
        conversation.observe(LLMFullResponseEndFrame())
        _reply(conversation, "New answer.")
        self.assertEqual(
            conversation.snapshot(),
            [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "Old tail."},
                {"role": "user", "content": "Interrupting question"},
                {"role": "assistant", "content": "New answer."},
            ],
        )

    def test_error_preserves_partial_text_without_a_response_end(self):
        conversation = _ScriptConversation(bot_audio=False)
        with conversation.submitting("Question"):
            pass
        conversation.observe(LLMFullResponseStartFrame())
        conversation.observe(LLMTextFrame(text="A partial"))
        self.assertEqual(
            conversation.snapshot(),
            [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "A partial"},
            ],
        )

    def test_failed_and_cancelled_submissions_roll_back_only_the_input(self):
        for error in (RuntimeError("enqueue failed"), asyncio.CancelledError()):
            with self.subTest(error=type(error).__name__):
                conversation = _ScriptConversation(bot_audio=False)
                with self.assertRaises(type(error)):
                    with conversation.submitting("Not submitted"):
                        _reply(conversation, "Already observed greeting")
                        raise error
                self.assertEqual(
                    conversation.snapshot(),
                    [{"role": "assistant", "content": "Already observed greeting"}],
                )

    def test_audio_uses_recognition_and_its_source_order(self):
        conversation = _ScriptConversation(bot_audio=True)
        greeting_at = time.monotonic()
        with conversation.submitting("User's question"):
            pass
        _reply(conversation, "Unspoken LLM text")
        conversation.observe(TTSTextFrame(text="Unspoken TTS text", text_type=TextType.SENTENCE))
        conversation.speech("Spoken answer")
        conversation.speech("Spoken greeting", started_at=greeting_at)
        self.assertEqual(
            conversation.snapshot(),
            [
                {"role": "assistant", "content": "Spoken greeting"},
                {"role": "user", "content": "User's question"},
                {"role": "assistant", "content": "Spoken answer"},
            ],
        )

    def test_duplicate_tool_names_cancel_by_id_and_freeze_nested_arguments(self):
        conversation = _ScriptConversation(bot_audio=False)
        arguments = {"items": [{"name": "first"}]}
        conversation.observe(
            FunctionCallInProgressFrame(
                function_name="order", tool_call_id="first", arguments=arguments
            )
        )
        arguments["items"][0]["name"] = "mutated"
        conversation.observe(
            FunctionCallInProgressFrame(
                function_name="order", tool_call_id="second", arguments={"second": True}
            )
        )
        before = conversation.snapshot()
        conversation.observe(
            EvalFunctionCallCancelFrame(function_name="order", tool_call_id="first")
        )
        _reply(conversation, "The second order is confirmed.")
        after = conversation.snapshot()
        self.assertEqual(after[0]["arguments"], {"items": [{"name": "first"}]})
        self.assertEqual([item["cancelled"] for item in after[:2]], [True, False])
        self.assertEqual([item["cancelled"] for item in before], [False, False])
        after[0]["arguments"]["items"][0]["name"] = "snapshot mutation"
        self.assertEqual(conversation.snapshot()[0]["arguments"], {"items": [{"name": "first"}]})
        self.assertEqual(
            after[-1], {"role": "assistant", "content": "The second order is confirmed."}
        )

    def test_unknown_tool_id_does_not_cancel_a_different_same_named_call(self):
        conversation = _ScriptConversation(bot_audio=False)
        conversation.observe(
            FunctionCallInProgressFrame(function_name="order", tool_call_id="known", arguments={})
        )
        conversation.observe(
            EvalFunctionCallCancelFrame(function_name="order", tool_call_id="other")
        )
        self.assertFalse(conversation.snapshot()[0]["cancelled"])

    def test_script_worker_json_reads_old_and_new_results(self):
        old = {"scenario_name": "old", "passed": True}
        self.assertEqual(_result_from_dict(old).conversation, [])
        result = EvalScriptResult(
            "new", True, conversation=[{"role": "user", "content": "Submitted"}]
        )
        self.assertEqual(_result_from_dict(asdict(result)), result)


class TestCaptureServiceAdmission(unittest.TestCase):
    def test_only_audio_capture_adds_stt_and_all_capture_requests_full_tools(self):
        for user_audio in (False, True):
            for bot_audio in (False, True):
                for capture in (False, True):
                    with self.subTest(user=user_audio, bot=bot_audio, capture=capture):
                        scenario = EvalScriptScenario(
                            name="tool-only",
                            user_audio=user_audio,
                            bot_audio=bot_audio,
                            turns=[
                                EvalScriptTurn(
                                    user="Question", expect=[EvalExpectation(event="function_call")]
                                )
                            ],
                        )
                        with patch(
                            "pipecat.evals.script_session.stt_service_from_config"
                        ) as build_stt:
                            session = EvalScriptSession.from_scenario(
                                scenario,
                                "ws://localhost:0",
                                params=EvalSessionParams(capture_conversation=capture),
                            )
                        self.assertEqual(build_stt.call_count, int(bot_audio and capture))
                        self.assertEqual(
                            session._client._params.capture_bot_audio, bot_audio and capture
                        )
                        self.assertEqual(
                            session._client._params.report_level, "full" if capture else None
                        )

    def test_direct_audio_capture_requires_an_injected_stt(self):
        with self.assertRaisesRegex(ValueError, "requires a bot STT"):
            EvalScriptSession(
                EvalScriptScenario(name="audio", bot_audio=True, turns=[]),
                "ws://localhost:0",
                params=EvalSessionParams(capture_conversation=True),
            )


class TestCaptureClient(unittest.IsolatedAsyncioTestCase):
    def _client(self, *, bot_audio=False, user_tts=None):
        return EvalClient(
            "ws://localhost:0",
            params=EvalClientParams(bot_audio=bot_audio),
            session_params=EvalSessionParams(capture_conversation=True),
            stream=EvalEventStream(bot_audio=bot_audio, trace=EvalTrace()),
            trace=EvalTrace(),
            user_tts=user_tts,
        )

    async def test_slow_synthesis_precedes_input_but_fast_reply_follows_it(self):
        client = self._client(user_tts=SimpleNamespace(sample_rate=16000))

        async def synthesize(text):
            _reply(client._conversation, "Greeting during preparation")
            await asyncio.sleep(0)
            return b"\0\0"

        async def inject(frame):
            if isinstance(frame, EndFrame):
                return
            if not client.conversation[-1].get("content") == "Question":
                return
            _reply(client._conversation, "Fast reply")

        client._user_tts.synthesize = synthesize
        client._sink = SimpleNamespace(inject=AsyncMock(side_effect=inject))
        await client.say_with_pauses(["Question", 10], transcript_text="Question")
        self.assertEqual(
            client.conversation,
            [
                {"role": "assistant", "content": "Greeting during preparation"},
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Fast reply"},
            ],
        )

    async def test_file_loading_precedes_submitted_wording(self):
        client = self._client()

        async def load(path):
            _reply(client._conversation, "Greeting while loading")
            return b"\0\0", 16000

        client._sink = SimpleNamespace(inject=AsyncMock())
        with patch("pipecat.evals.client.load_user_audio", side_effect=load):
            await client.play("recording.wav", transcript_text="Recorded question")
        self.assertEqual(
            client.conversation,
            [
                {"role": "assistant", "content": "Greeting while loading"},
                {"role": "user", "content": "Recorded question"},
            ],
        )

    async def test_failed_text_enqueue_does_not_create_a_user_message(self):
        client = self._client()
        client.send = AsyncMock(side_effect=RuntimeError("enqueue failed"))
        with self.assertRaisesRegex(RuntimeError, "enqueue failed"):
            await client.send_text("Not submitted")
        self.assertEqual(client.conversation, [])

    async def test_capture_refuses_next_input_when_previous_speech_cannot_drain(self):
        for capture in (True, False):
            for quiet in (True, False):
                for explicit_send in (True, False):
                    for segmented in (True, False):
                        with self.subTest(
                            capture=capture,
                            quiet=quiet,
                            explicit=explicit_send,
                            segmented=segmented,
                        ):
                            await self._undrained_speech(capture, quiet, explicit_send, segmented)

    async def test_every_audio_submission_checks_the_final_boundary(self):
        for method, argument in (
            ("send_text", "Question"),
            ("send_dtmf", "1"),
            ("say", "Question"),
            ("say_with_pauses", ["Question", 10]),
            ("play", "recording.wav"),
        ):
            with self.subTest(method=method):
                tts = SimpleNamespace(sample_rate=16000, synthesize=AsyncMock(return_value=b"\0\0"))
                client = self._client(bot_audio=True, user_tts=tts)
                client._sink = SimpleNamespace(inject=AsyncMock())
                client.send = AsyncMock()
                client._stream.bot_turn_started()
                with patch("pipecat.evals.client.load_user_audio", return_value=(b"\0\0", 16000)):
                    with self.assertRaisesRegex(EvalHarnessError, "must be fully transcribed"):
                        await getattr(client, method)(argument)
                client.send.assert_not_awaited()
                client._sink.inject.assert_not_awaited()
                self.assertEqual(client.conversation, [])

    async def test_text_capture_does_not_require_audio_drain(self):
        client = self._client(bot_audio=False)
        client._stream.bot_turn_started()
        client.send = AsyncMock()
        await client.send_text("Question")
        self.assertEqual(client.conversation, [{"role": "user", "content": "Question"}])
        client.send.assert_awaited_once()

    async def _undrained_speech(self, capture, quiet, explicit_send, segmented):
        scenario = EvalScriptScenario(
            name="undrained-speech",
            bot_audio=True,
            turns=[
                EvalScriptTurn(
                    user="First", expect=[EvalExpectation(event="llm_started", within_ms=100)]
                ),
                EvalScriptTurn(
                    user="Second",
                    send_after=EvalSendAfter(event=None, delay_ms=0) if explicit_send else None,
                    expect=[EvalExpectation(event="llm_started", within_ms=100)],
                ),
            ],
        )
        session = EvalScriptSession(
            scenario,
            "ws://localhost:0",
            params=EvalSessionParams(capture_conversation=capture),
            bot_stt=_SegmentedControlledSTT() if segmented else _FinalDrainSTT(),
        )
        client, stream = session._client, session._stream
        client.wait_for_bot = AsyncMock()
        client.start = AsyncMock()
        client.handshake = AsyncMock()
        client.stop = AsyncMock()
        client.send = AsyncMock()

        async def progress(record):
            if record.status != "turn":
                return
            await stream.append(stream.frame_to_event(LLMFullResponseStartFrame()))
            if record.turn_index == 0:
                if client._conversation is not None:
                    client._conversation.speech("Already observed prefix")
                stream.frame_to_event(BotStartedSpeakingFrame())
                stream.bot_turn_started()
                if quiet:
                    stream.frame_to_event(BotStoppedSpeakingFrame())

        session._driver._progress = progress
        with patch("pipecat.evals.script_driver.BOT_QUIET_MAX_WAIT_S", 0.01):
            result = await session.run()
        refused = capture and not segmented
        self.assertEqual(client.send.await_count, 1 if refused else 2)
        if refused:
            self.assertFalse(result.passed)
            self.assertEqual([failure.kind for failure in result.failures], ["error"])
            self.assertEqual([turn.status for turn in result.turns], ["passed", "failed"])
            self.assertEqual(
                result.conversation,
                [
                    {"role": "user", "content": "First"},
                    {"role": "assistant", "content": "Already observed prefix"},
                ],
            )
        else:
            self.assertTrue(result.passed, result.failures)
            if capture:
                self.assertEqual(
                    [
                        message["content"]
                        for message in result.conversation
                        if message["role"] == "user"
                    ],
                    ["First", "Second"],
                )
            else:
                self.assertEqual(result.conversation, [])

    async def test_gate_keeps_tagged_old_speech_only_in_conversation(self):
        stream = EvalEventStream(bot_audio=True, trace=EvalTrace())
        conversation = _ScriptConversation(bot_audio=True)
        old_at = time.monotonic()
        stream.frame_to_event(BotStartedSpeakingFrame())
        with conversation.submitting("Interrupt"):
            pass
        stream.input_sent()
        stream.frame_to_event(BotStoppedSpeakingFrame())
        stream.frame_to_event(BotStartedSpeakingFrame())
        gate = _BotSpeechGate(stream, EvalTrace(), deque(), conversation=conversation)
        down, _ = await run_test(
            gate,
            frames_to_send=[
                _BotSegmentTranscriptionFrame(
                    text="Old spoken prefix", user_id="", timestamp="", segment_started_at=old_at
                ),
                _BotSegmentTranscriptionFrame(
                    text="Actual reply",
                    user_id="",
                    timestamp="",
                    segment_started_at=time.monotonic(),
                ),
            ],
        )
        self.assertEqual(
            [frame.text for frame in down if isinstance(frame, TranscriptionFrame)],
            ["Actual reply"],
        )
        self.assertEqual(
            conversation.snapshot(),
            [
                {"role": "assistant", "content": "Old spoken prefix"},
                {"role": "user", "content": "Interrupt"},
                {"role": "assistant", "content": "Actual reply"},
            ],
        )

    async def test_gate_keeps_stable_chunks_but_never_interim_or_ambiguous_text(self):
        stream = EvalEventStream(bot_audio=True, trace=EvalTrace())
        conversation = _ScriptConversation(bot_audio=True)
        gate = _BotSpeechGate(stream, EvalTrace(), None, conversation=conversation)
        await run_test(
            gate,
            frames_to_send=[
                InterimTranscriptionFrame(text="draft", user_id="", timestamp=""),
                TranscriptionFrame(text="Stable chunk", user_id="", timestamp="", finalized=False),
                TranscriptionFrame(text="Final chunk", user_id="", timestamp="", finalized=True),
            ],
        )
        self.assertEqual(
            conversation.snapshot(), [{"role": "assistant", "content": "Stable chunk Final chunk"}]
        )
        stream.frame_to_event(BotStartedSpeakingFrame())
        stream.input_sent()
        gate = _BotSpeechGate(stream, EvalTrace(), None, conversation=conversation)
        _, up = await run_test(
            gate,
            frames_to_send=[TranscriptionFrame(text="Ambiguous", user_id="", timestamp="")],
        )
        self.assertTrue(any(isinstance(frame, ErrorFrame) for frame in up))
        self.assertEqual(
            conversation.snapshot(), [{"role": "assistant", "content": "Stable chunk Final chunk"}]
        )


class _FinalDrainSTT(STTService):
    """A controlled recognizer that supplies stable output only during graceful end."""

    async def run_stt(self, audio):
        yield None

    async def stop(self, frame):
        await self.push_frame(
            TranscriptionFrame(text="Recognized speech", user_id="", timestamp="")
        )
        await super().stop(frame)


class _SegmentedControlledSTT(SegmentedSTTService):
    async def run_stt(self, audio):
        yield None


class _FailedDrainSTT(_FinalDrainSTT):
    async def stop(self, frame):
        await self.push_error("capture recognition failed")
        await super().stop(frame)


class TestCapturedScriptWebSocket(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.server = _FakeRTVIServer(_free_port())
        await self.server.start()

    async def asyncTearDown(self):
        await self.server.stop()

    async def test_text_greeting_multiturn_tools_and_partial_failure(self):
        self.server.greeting = [
            _rtvi("bot-llm-started"),
            _rtvi("bot-llm-text", {"text": "Welcome."}),
            _rtvi("bot-llm-stopped"),
        ]
        self.server.on_text(
            "Question",
            _rtvi("bot-llm-started"),
            _rtvi("bot-llm-text", {"text": "Checking "}),
            _rtvi(
                "llm-function-call-in-progress",
                {"function_name": "lookup", "tool_call_id": "call", "arguments": {"item": 1}},
            ),
            _rtvi("bot-llm-text", {"text": "now."}),
            _rtvi("bot-llm-stopped"),
        )
        self.server.on_text(
            "Next", _rtvi("bot-llm-started"), _rtvi("bot-llm-text", {"text": "Unfinished"})
        )
        scenario = EvalScriptScenario(
            name="conversation",
            bot_audio=False,
            turns=[
                EvalScriptTurn(
                    user=None, expect=[EvalExpectation(event="llm_response", within_ms=2000)]
                ),
                EvalScriptTurn(
                    user="Question", expect=[EvalExpectation(event="llm_response", within_ms=2000)]
                ),
                EvalScriptTurn(
                    user="Next", expect=[EvalExpectation(event="llm_response", within_ms=200)]
                ),
            ],
        )
        result = await EvalScriptSession.from_scenario(
            scenario, self.server.url, params=EvalSessionParams(capture_conversation=True)
        ).run()
        self.assertFalse(result.passed)
        self.assertEqual(result.failures[0].kind, "timeout")
        self.assertEqual(
            result.conversation,
            [
                {"role": "assistant", "content": "Welcome."},
                {"role": "user", "content": "Question"},
                {
                    "role": "tool_call",
                    "name": "lookup",
                    "arguments": {"item": 1},
                    "cancelled": False,
                },
                {"role": "assistant", "content": "Checking now."},
                {"role": "user", "content": "Next"},
                {"role": "assistant", "content": "Unfinished"},
            ],
        )
        self.assertTrue(any("full" in str(message) for message in self.server.received))

    async def test_audio_captures_final_drain_without_response_assertion_or_tts_substitution(self):
        self.server.on_text(
            "Question",
            _rtvi("bot-llm-started"),
            _rtvi("bot-llm-text", {"text": "Unspoken LLM text"}),
            _rtvi("bot-tts-text", {"text": "Unspoken TTS text"}),
            _rtvi("bot-llm-stopped"),
        )
        scenario = EvalScriptScenario(
            name="capture-without-response",
            bot_audio=True,
            turns=[
                EvalScriptTurn(
                    user="Question", expect=[EvalExpectation(event="llm_response", within_ms=2000)]
                )
            ],
        )
        result = await EvalScriptSession.from_scenario(
            scenario,
            self.server.url,
            params=EvalSessionParams(capture_conversation=True),
            bot_stt=_FinalDrainSTT(sample_rate=16000),
        ).run()
        self.assertTrue(result.passed, result.failures)
        self.assertEqual(
            result.conversation,
            [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Recognized speech"},
            ],
        )

    async def test_capture_failure_during_drain_invalidates_a_nominal_pass(self):
        self.server.on_text("Question", _rtvi("bot-llm-started"))
        scenario = EvalScriptScenario(
            name="capture-drain-error",
            bot_audio=True,
            turns=[
                EvalScriptTurn(
                    user="Question", expect=[EvalExpectation(event="llm_started", within_ms=2000)]
                )
            ],
        )
        result = await EvalScriptSession.from_scenario(
            scenario,
            self.server.url,
            params=EvalSessionParams(capture_conversation=True),
            bot_stt=_FailedDrainSTT(sample_rate=16000),
        ).run()
        self.assertFalse(result.passed)
        self.assertTrue(any(failure.kind == "error" for failure in result.failures))
        self.assertEqual(result.failures[0].turn_index, -1)
        self.assertEqual(result.conversation[-1]["content"], "Recognized speech")

    async def test_capture_disabled_preserves_completed_assertions_after_late_error(self):
        self.server.on_text("Question", _rtvi("bot-llm-started"))
        scenario = EvalScriptScenario(
            name="capture-off",
            bot_audio=True,
            turns=[
                EvalScriptTurn(
                    user="Question", expect=[EvalExpectation(event="llm_started", within_ms=2000)]
                )
            ],
        )
        result = await EvalScriptSession.from_scenario(
            scenario,
            self.server.url,
            bot_stt=_FailedDrainSTT(sample_rate=16000),
        ).run()
        self.assertTrue(result.passed, result.failures)
        self.assertEqual(result.conversation, [])

    async def test_decided_assertion_failure_keeps_its_cause_when_capture_also_fails(self):
        self.server.on_text(
            "Question",
            _rtvi(
                "llm-function-call-in-progress",
                {"function_name": "order", "tool_call_id": "call", "arguments": {"count": 2}},
            ),
        )
        scenario = EvalScriptScenario(
            name="assertion-and-capture-error",
            bot_audio=True,
            turns=[
                EvalScriptTurn(
                    user="Question",
                    expect=[
                        EvalExpectation(
                            event="function_call",
                            calls=[EvalFunctionCall(name="order", args={"count": 1})],
                            within_ms=200,
                        )
                    ],
                )
            ],
        )
        result = await EvalScriptSession.from_scenario(
            scenario,
            self.server.url,
            params=EvalSessionParams(capture_conversation=True),
            bot_stt=_FailedDrainSTT(sample_rate=16000),
        ).run()
        self.assertFalse(result.passed)
        self.assertEqual([failure.kind for failure in result.failures], ["function_args_mismatch"])
