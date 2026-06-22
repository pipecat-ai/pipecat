#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the eval harness.

Two layers:

- :class:`TestTranslate` unit-tests the RTVI-server-message → friendly-event
  translation in isolation (pure, fast).
- :class:`TestEvalsHarnessIntegration` runs scenarios via :meth:`EvalSession.from_scenario` against a fake
  RTVI WebSocket server that replies to ``client-ready``/``send-text`` with
  scripted RTVI server messages — exercising the handshake, send/receive, event
  matching, and context paths without a real bot pipeline.
"""

import json
import socket
import unittest

import websockets

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.evals.harness import EvalSession
from pipecat.evals.scenario import (
    EvalExpectation,
    EvalFunctionCall,
    EvalScenario,
    EvalSendAfter,
    EvalTurn,
)


def _rtvi(msg_type: str, data: dict | None = None) -> str:
    return json.dumps({"label": RTVI.MESSAGE_LABEL, "type": msg_type, "data": data})


def _session(bot_audio: bool = False) -> EvalSession:
    return EvalSession(EvalScenario(name="t", turns=[], bot_audio=bot_audio), "ws://localhost:0")


class TestTranslate(unittest.TestCase):
    def test_user_speaking_events(self):
        s = _session()
        self.assertEqual(
            s._translate({"type": "user-started-speaking"}),
            [{"type": "user_started_speaking"}],
        )
        self.assertEqual(
            s._translate({"type": "user-stopped-speaking"}),
            [{"type": "user_stopped_speaking"}],
        )

    def test_user_started_speaking_discards_interrupted_output(self):
        # An interruption: the bot's in-flight output (buffers + queued events)
        # is discarded so it can't be matched against this turn.
        s = _session(bot_audio=True)
        s._text_buffer = ["greeting"]
        s._tts_audio = bytearray(b"\x01\x02\x03\x04")
        s._queue.put_nowait({"type": "llm_response", "text": "greeting"})
        self.assertEqual(
            s._translate({"type": "user-started-speaking"}),
            [{"type": "user_started_speaking"}],
        )
        self.assertEqual(s._text_buffer, [])
        self.assertEqual(len(s._tts_audio), 0)
        self.assertTrue(s._queue.empty())

    def test_discard_preserves_user_transcription(self):
        # A DTMF keypress emits its user_transcription right before the turn-start
        # interruption. The discard must keep it (it's the turn's input) while
        # still dropping the bot's interrupted output.
        s = _session(bot_audio=True)
        s._queue.put_nowait({"type": "llm_response", "text": "greeting"})
        s._queue.put_nowait({"type": "user_transcription", "transcript": "DTMF: 1#"})
        self.assertEqual(
            s._translate({"type": "user-started-speaking"}),
            [{"type": "user_started_speaking"}],
        )
        # The bot output is gone; the user transcription survives, still queued.
        self.assertEqual(
            s._queue.get_nowait(), {"type": "user_transcription", "transcript": "DTMF: 1#"}
        )
        self.assertTrue(s._queue.empty())

    def test_user_transcription_final_only(self):
        s = _session()
        interim = {"type": "user-transcription", "data": {"text": "he", "final": False}}
        final = {"type": "user-transcription", "data": {"text": "hello", "final": True}}
        self.assertEqual(s._translate(interim), [])
        self.assertEqual(
            s._translate(final),
            [{"type": "user_transcription", "transcript": "hello"}],
        )

    def _one_event(self, result: list[dict]) -> dict:
        self.assertEqual(len(result), 1)
        return result[0]

    def test_text_mode_accumulates_bot_llm_text(self):
        # Text mode (bot_audio=False): llm_response comes from bot-llm-text.
        s = _session(bot_audio=False)
        self.assertEqual(s._translate({"type": "bot-llm-started"}), [{"type": "llm_started"}])
        self.assertEqual(s._translate({"type": "bot-llm-text", "data": {"text": "Hello "}}), [])
        self.assertEqual(s._translate({"type": "bot-llm-text", "data": {"text": "world"}}), [])
        # bot-tts-text is ignored in text mode.
        self.assertEqual(s._translate({"type": "bot-tts-text", "data": {"text": "ignored"}}), [])
        self.assertEqual(
            self._one_event(s._translate({"type": "bot-llm-stopped"})),
            {"type": "llm_response", "text": "Hello world"},
        )

    def test_interruption_suppresses_straggler_llm_response(self):
        # After an interruption, the interrupted response can still flush a trailing
        # token. It must be dropped (not attributed to the new turn); the new
        # response begins at the next bot-llm-started.
        s = _session(bot_audio=False)
        s._translate({"type": "bot-llm-started"})
        s._translate({"type": "bot-llm-text", "data": {"text": "Tell me about Paris"}})
        # User barges in: the bot is interrupted.
        self.assertEqual(s._translate({"type": "bot-interrupted"}), [{"type": "bot_interrupted"}])
        # Straggler from the interrupted response arrives just after the interrupt.
        self.assertEqual(
            s._translate({"type": "bot-llm-text", "data": {"text": " what would"}}), []
        )
        self.assertEqual(s._translate({"type": "bot-llm-stopped"}), [])  # straggler dropped
        # The genuinely new response.
        self.assertEqual(s._translate({"type": "bot-llm-started"}), [{"type": "llm_started"}])
        self.assertEqual(s._translate({"type": "bot-llm-text", "data": {"text": "Tokyo"}}), [])
        self.assertEqual(
            self._one_event(s._translate({"type": "bot-llm-stopped"})),
            {"type": "llm_response", "text": "Tokyo"},
        )

    def test_audio_mode_llm_text_and_tts_text(self):
        # Audio mode (bot_audio=True): bot-llm-text -> llm_response (the LLM text),
        # bot-tts-text -> tts_response (the TTS's spoken text, one per segment).
        # The Whisper-transcription `response` event is exercised separately.
        s = _session(bot_audio=True)
        self.assertEqual(s._translate({"type": "bot-llm-started"}), [{"type": "llm_started"}])
        self.assertEqual(s._translate({"type": "bot-llm-text", "data": {"text": "Hello "}}), [])
        self.assertEqual(
            self._one_event(s._translate({"type": "bot-llm-stopped"})),
            {"type": "llm_response", "text": "Hello "},
        )
        self.assertEqual(
            self._one_event(s._translate({"type": "bot-tts-text", "data": {"text": "spoken "}})),
            {"type": "tts_response", "text": "spoken "},
        )
        self.assertEqual(
            self._one_event(s._translate({"type": "bot-tts-text", "data": {"text": "words"}})),
            {"type": "tts_response", "text": "words"},
        )
        self.assertEqual(s._translate({"type": "bot-tts-stopped"}), [])

    def test_empty_response_still_emitted(self):
        # An interrupted response (no text) emits an empty llm_response — the
        # matcher's aggregation decides whether that should pass or fail.
        s = _session(bot_audio=False)
        self.assertEqual(s._translate({"type": "bot-llm-started"}), [{"type": "llm_started"}])
        self.assertEqual(
            self._one_event(s._translate({"type": "bot-llm-stopped"})),
            {"type": "llm_response", "text": ""},
        )

    def test_function_call(self):
        s = _session()
        msg = {
            "type": "llm-function-call-in-progress",
            "data": {"function_name": "get_weather", "arguments": {"city": "Paris"}},
        }
        self.assertEqual(
            s._translate(msg),
            [{"type": "function_call", "name": "get_weather", "args": {"city": "Paris"}}],
        )

    def test_unmapped_message_ignored(self):
        self.assertEqual(_session()._translate({"type": "metrics", "data": {}}), [])


class _FakeJudge:
    """Returns queued verdicts without calling a real LLM."""

    def __init__(self, verdicts: list[str]):
        self._verdicts = list(verdicts)
        self.calls: list[str] = []
        self.segments: list[str] = []

    def add_user_message(self, text):
        pass

    def add_assistant_message(self, text):
        self.segments.append(text)

    async def evaluate(self, criterion: str):
        from pipecat.evals.judge import JudgeVerdict

        self.calls.append(criterion)
        v = self._verdicts.pop(0)
        return JudgeVerdict(verdict=v, reason=f"({v})", raw_response="")


class TestEvaluateAggregate(unittest.IsolatedAsyncioTestCase):
    """The pass/fail/continue decision over accumulated response text."""

    async def test_text_contains_present_passes(self):
        s = _session(bot_audio=False)
        exp = EvalExpectation(event="llm_response", text_contains="Paris")
        self.assertEqual(await s._evaluate_aggregate("The capital is Paris.", exp), ("pass", ""))

    async def test_text_contains_absent_continues(self):
        s = _session()
        exp = EvalExpectation(event="llm_response", text_contains="Paris")
        status, _ = await s._evaluate_aggregate("Let me check on that.", exp)
        self.assertEqual(status, "continue")

    async def test_eval_yes_passes(self):
        s = _session()
        s._judge = _FakeJudge(["yes"])
        exp = EvalExpectation(event="llm_response", eval="describes the weather")
        status, reason = await s._evaluate_aggregate("It's 75 and sunny.", exp)
        self.assertEqual(status, "pass")
        self.assertIn("judge said yes", reason)

    async def test_eval_no_fails(self):
        s = _session()
        s._judge = _FakeJudge(["no"])
        exp = EvalExpectation(event="llm_response", eval="describes the weather")
        status, reason = await s._evaluate_aggregate("I like turtles.", exp)
        self.assertEqual(status, "fail")
        self.assertIn("judge said no", reason)

    async def test_eval_continue_waits_for_more(self):
        s = _session()
        s._judge = _FakeJudge(["continue"])
        exp = EvalExpectation(event="llm_response", eval="describes the weather")
        status, _ = await s._evaluate_aggregate("Let me check on that.", exp)
        self.assertEqual(status, "continue")

    async def test_eval_empty_aggregate_skips_judge(self):
        s = _session()
        judge = _FakeJudge([])  # would IndexError if the judge were called
        s._judge = judge
        exp = EvalExpectation(event="llm_response", eval="describes the weather")
        status, _ = await s._evaluate_aggregate("   ", exp)
        self.assertEqual(status, "continue")
        self.assertEqual(judge.calls, [])


class TestRequiredReportLevel(unittest.TestCase):
    """The minimal function-call report level the harness asks the bot for."""

    def _level(self, *expects) -> str | None:
        scenario = EvalScenario(
            name="t", bot_audio=False, turns=[EvalTurn(user="x", expect=list(expects))]
        )
        return EvalSession(scenario, "ws://localhost:0")._required_report_level()

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


class TestNeedsVadEvents(unittest.TestCase):
    """The harness enables raw VAD events only when a scenario references them."""

    def _needs(self, turn: EvalTurn) -> bool:
        scenario = EvalScenario(name="t", turns=[turn])
        return EvalSession(scenario, "ws://localhost:0")._needs_vad_events()

    def test_false_without_vad_events(self):
        self.assertFalse(
            self._needs(EvalTurn(user="x", expect=[EvalExpectation(event="response")]))
        )

    def test_true_when_expected(self):
        self.assertTrue(
            self._needs(
                EvalTurn(user="x", expect=[EvalExpectation(event="vad_user_started_speaking")])
            )
        )

    def test_true_when_used_as_send_after_anchor(self):
        self.assertTrue(
            self._needs(
                EvalTurn(
                    user="x",
                    expect=[EvalExpectation(event="response")],
                    send_after=EvalSendAfter(event="vad_user_stopped_speaking", delay_ms=2000),
                )
            )
        )


class TestConnectURL(unittest.TestCase):
    """The harness signals skip-TTS via the connect URL in text mode."""

    def _url(self, bot_audio: bool, base: str = "ws://localhost:7860") -> str:
        scenario = EvalScenario(name="t", turns=[], bot_audio=bot_audio)
        return EvalSession(scenario, base)._connect_url()

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
        scenario = EvalScenario(
            name="t",
            bot_audio=True,
            turns=[EvalTurn(user="x", expect=[EvalExpectation(event="response", eval="ok")])],
        )
        url = EvalSession(scenario, "ws://localhost:7860")._connect_url()
        self.assertIn("capture_bot_audio=true", url)
        self.assertNotIn("skip_tts", url)  # audio mode, so no skip

    def test_speech_adds_user_audio(self):
        # Audio-mode user turns enable the transport's virtual mic; text-mode
        # scenarios must not (the mic would feed silence into the bot's STT).
        scenario = EvalScenario(name="t", turns=[], bot_audio=True)
        session = EvalSession(scenario, "ws://localhost:7860", speech=object())
        self.assertIn("user_audio=true", session._connect_url())
        no_speech = EvalSession(scenario, "ws://localhost:7860")
        self.assertNotIn("user_audio", no_speech._connect_url())


class TestResponseTranscriptionSkip(unittest.IsolatedAsyncioTestCase):
    async def test_skipped_without_audio_mode(self):
        # The `response` transcription needs the bot's audio; without audio mode,
        # skip (don't run a guaranteed failure).
        scenario = EvalScenario(
            name="t",
            bot_audio=False,
            turns=[EvalTurn(user="x", expect=[EvalExpectation(event="response", eval="ok")])],
        )
        result = await EvalSession(scenario, "ws://localhost:0").run()
        self.assertIsNotNone(result.skipped)
        self.assertFalse(result.passed)
        self.assertIn("response", result.skipped)


class TestTextContainsResolution(unittest.TestCase):
    """text_contains resolves against whichever event carries the text."""

    def _check(self, event: dict, exp: EvalExpectation):
        return EvalSession._check_payload(event, exp, 0, 0)

    def test_on_one_event_text(self):
        exp = EvalExpectation(event="llm_response", text_contains="Paris")
        self.assertIsNone(self._check({"type": "llm_response", "text": "It's Paris."}, exp))
        self.assertIsNotNone(self._check({"type": "llm_response", "text": "London."}, exp))

    def test_on_user_transcription_transcript(self):
        exp = EvalExpectation(event="user_transcription", text_contains="hello")
        ok = {"type": "user_transcription", "transcript": "hello world"}
        self.assertIsNone(self._check(ok, exp))
        failure = self._check({"type": "user_transcription", "transcript": "bye"}, exp)
        self.assertIsNotNone(failure)
        self.assertIn("does not contain", failure.reason)


class TestAudioSender(unittest.IsolatedAsyncioTestCase):
    """User audio goes out whole; the eval transport's virtual mic paces it bot-side."""

    async def test_send_user_audio_sends_whole_utterance(self):
        s = _session(bot_audio=True)
        sent: list[tuple[bytes, int]] = []

        class _FakeSpeech:
            sample_rate = 16000

            async def generate(self, text):
                return b"\x01\x02" * 16000 * 2, 16000  # 2s of 16kHz mono

        async def fake_send_raw(chunk, sample_rate):
            sent.append((chunk, sample_rate))

        s._speech = _FakeSpeech()
        s._send_raw_audio = fake_send_raw
        await s._send_user_audio("hello")

        self.assertEqual(len(sent), 2)  # 2s -> two ~1s slices
        self.assertEqual(b"".join(chunk for chunk, _ in sent), b"\x01\x02" * 16000 * 2)
        self.assertTrue(all(rate == 16000 for _, rate in sent))


class TestDTMFSender(unittest.IsolatedAsyncioTestCase):
    """A dtmf turn sends one RTVI ``dtmf`` message per key."""

    async def test_send_user_dtmf_one_message_per_key(self):
        s = _session()
        sent: list[RTVI.Message] = []

        async def fake_send(message):
            sent.append(message)

        s._send = fake_send
        await s._send_user_dtmf("12#")

        self.assertEqual(len(sent), 3)
        self.assertTrue(all(m.type == "dtmf" for m in sent))
        self.assertEqual([m.data["button"] for m in sent], ["1", "2", "#"])


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
        self._server: websockets.WebSocketServer | None = None

    def on_text(self, content: str, *messages: str):
        self.script[content] = list(messages)

    async def _handler(self, ws):
        async for raw in ws:
            msg = json.loads(raw)
            self.received.append(msg)
            match msg.get("type"):
                case "client-ready":
                    await ws.send(_rtvi("bot-ready", {"version": RTVI.PROTOCOL_VERSION}))
                case "send-text":
                    for out in self.script.get(msg["data"]["content"], []):
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
        scenario = EvalScenario(
            name="capital",
            bot_audio=False,
            turns=[
                EvalTurn(
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
        result = await EvalSession.from_scenario(scenario, self.server.url).run()
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
        scenario = EvalScenario(
            name="filler",
            bot_audio=False,
            turns=[
                EvalTurn(
                    user="weather?",
                    expect=[
                        EvalExpectation(event="llm_response", within_ms=2000, text_contains="Paris")
                    ],
                )
            ],
        )
        result = await EvalSession.from_scenario(scenario, self.server.url).run()
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
        scenario = EvalScenario(
            name="tool",
            turns=[
                EvalTurn(
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
        result = await EvalSession.from_scenario(scenario, self.server.url).run()
        self.assertTrue(result.passed, f"failures: {[str(f) for f in result.failures]}")

    async def test_text_mismatch_fails_clearly(self):
        self.server.on_text(
            "hi",
            _rtvi("bot-llm-started"),
            _rtvi("bot-llm-text", {"text": "Paris"}),
            _rtvi("bot-llm-stopped"),
        )
        scenario = EvalScenario(
            name="mismatch",
            bot_audio=False,
            turns=[
                EvalTurn(
                    user="hi",
                    expect=[
                        EvalExpectation(
                            event="llm_response", within_ms=2000, text_contains="London"
                        )
                    ],
                )
            ],
        )
        result = await EvalSession.from_scenario(scenario, self.server.url).run()
        self.assertFalse(result.passed)
        self.assertEqual(len(result.failures), 1)
        self.assertIn("does not contain", result.failures[0].reason)

    async def test_missing_event_times_out(self):
        scenario = EvalScenario(
            name="never",
            turns=[
                EvalTurn(user="hi", expect=[EvalExpectation(event="llm_response", within_ms=200)])
            ],
        )
        result = await EvalSession.from_scenario(scenario, self.server.url).run()
        self.assertFalse(result.passed)
        self.assertEqual(len(result.failures), 1)
        self.assertIn("arrived within", result.failures[0].reason)
        self.assertIn("200ms", result.failures[0].reason)

    async def test_subsequent_assertions_skipped_after_timeout(self):
        scenario = EvalScenario(
            name="cascading",
            turns=[
                EvalTurn(
                    user="hi",
                    expect=[
                        EvalExpectation(event="llm_started", within_ms=100),
                        EvalExpectation(event="llm_response", within_ms=100),
                        EvalExpectation(event="function_call", within_ms=100),
                    ],
                )
            ],
        )
        result = await EvalSession.from_scenario(scenario, self.server.url).run()
        self.assertFalse(result.passed)
        self.assertEqual(len(result.failures), 1, "only the first failed expectation should report")

    async def test_turn_shares_one_deadline_across_expectations(self):
        # A turn that expects a function call AND a response but gets neither must
        # fail within a single within_ms budget. The function_call timeout returns a
        # failure (not a raise) so the loop continues to the response; both share the
        # anchor, so the run takes ~one budget, not budget-per-expectation.
        scenario = EvalScenario(
            name="shared_deadline",
            turns=[
                EvalTurn(
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
        result = await EvalSession.from_scenario(scenario, self.server.url).run()
        self.assertFalse(result.passed)
        # One shared 400ms budget, not 400ms + 400ms.
        self.assertLess(result.duration_ms, 700)

    async def test_send_after_delays_run(self):
        self.server.on_text("first", _rtvi("bot-llm-started"), _rtvi("bot-llm-stopped"))
        self.server.on_text(
            "second",
            _rtvi("bot-llm-started"),
            _rtvi("bot-llm-text", {"text": "ok"}),
            _rtvi("bot-llm-stopped"),
        )
        scenario = EvalScenario(
            name="send_after",
            bot_audio=False,
            turns=[
                EvalTurn(
                    user="first", expect=[EvalExpectation(event="llm_started", within_ms=2000)]
                ),
                EvalTurn(
                    user="second",
                    expect=[EvalExpectation(event="llm_response", within_ms=2000)],
                    send_after=EvalSendAfter(event="llm_started", delay_ms=200),
                ),
            ],
        )
        result = await EvalSession.from_scenario(scenario, self.server.url).run()
        self.assertTrue(result.passed, f"failures: {[str(f) for f in result.failures]}")
        self.assertGreaterEqual(result.duration_ms, 200)

    async def test_connect_failure_reported_cleanly(self):
        scenario = EvalScenario(
            name="no_bot",
            turns=[EvalTurn(user="x", expect=[EvalExpectation(event="llm_started")])],
        )
        result = await EvalSession.from_scenario(
            scenario, f"ws://localhost:{_free_port()}", connect_timeout_s=0.5
        ).run()
        self.assertFalse(result.passed)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].event_name, "<connect>")
        self.assertIn("failed to connect", result.failures[0].reason)

    async def test_unexpected_error_surfaced_not_swallowed(self):
        # A sub-pipeline that fails to start (e.g. a local model thrashing under
        # load) must be reported as a structured failure with its traceback, not
        # propagate out raw and get swallowed as a bare "error:" with no eval.log.
        class _BoomSpeech:
            sample_rate = 16000

            async def start(self):
                raise RuntimeError("kokoro boom")

            async def aclose(self):
                pass

        scenario = EvalScenario(
            name="boom",
            turns=[EvalTurn(user="hi", expect=[EvalExpectation(event="llm_started")])],
        )
        result = await EvalSession.from_scenario(
            scenario, self.server.url, speech=_BoomSpeech()
        ).run()
        self.assertFalse(result.passed)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].event_name, "<error>")
        self.assertIn("RuntimeError: kokoro boom", result.failures[0].reason)
        # The full traceback is preserved in the debug trace (saved to <bot>.eval.log).
        self.assertTrue(any("kokoro boom" in line for line in result.debug_log))
        self.assertTrue(any("Traceback" in line for line in result.debug_log))

    async def test_context_sends_eval_context_message(self):
        self.server.on_text("hi", _rtvi("bot-llm-started"), _rtvi("bot-llm-stopped"))
        scenario = EvalScenario(
            name="context",
            turns=[
                EvalTurn(user="hi", expect=[EvalExpectation(event="llm_started", within_ms=2000)])
            ],
            context=[{"role": "system", "content": "be terse"}],
        )
        result = await EvalSession.from_scenario(scenario, self.server.url).run()
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
        scenario = EvalScenario(
            name="nocontext",
            turns=[
                EvalTurn(user="hi", expect=[EvalExpectation(event="llm_started", within_ms=2000)])
            ],
        )
        await EvalSession.from_scenario(scenario, self.server.url).run()
        context_messages = [
            m
            for m in self.server.received
            if m.get("type") == "client-message" and m["data"].get("t") == "eval-context"
        ]
        self.assertEqual(context_messages, [])


if __name__ == "__main__":
    unittest.main()
