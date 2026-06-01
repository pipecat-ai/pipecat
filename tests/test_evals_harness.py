#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the eval harness.

Two layers:

- :class:`TestTranslate` unit-tests the RTVI-server-message → friendly-event
  translation in isolation (pure, fast).
- :class:`TestEvalsHarnessIntegration` runs :func:`run_scenario` against a fake
  RTVI WebSocket server that replies to ``client-ready``/``send-text`` with
  scripted RTVI server messages — exercising the handshake, send/receive, event
  matching, and reset paths without a real bot pipeline.
"""

import asyncio
import json
import socket
import unittest

import websockets

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.evals.harness import _RunContext, _translate, run_scenario
from pipecat.evals.scenario import Expectation, Scenario, SendAfter, Turn


def _rtvi(msg_type: str, data: dict | None = None) -> str:
    return json.dumps({"label": RTVI.MESSAGE_LABEL, "type": msg_type, "data": data})


def _ctx() -> _RunContext:
    return _RunContext(ws=None, queue=asyncio.Queue(), latest_event_times={}, events_seen=[])


class TestTranslate(unittest.TestCase):
    def test_user_speaking_events(self):
        ctx = _ctx()
        self.assertEqual(
            _translate({"type": "user-started-speaking"}, ctx),
            [{"type": "user_started_speaking"}],
        )
        self.assertEqual(
            _translate({"type": "user-stopped-speaking"}, ctx),
            [{"type": "user_stopped_speaking"}],
        )

    def test_user_transcription_final_only(self):
        ctx = _ctx()
        interim = {"type": "user-transcription", "data": {"text": "he", "final": False}}
        final = {"type": "user-transcription", "data": {"text": "hello", "final": True}}
        self.assertEqual(_translate(interim, ctx), [])
        self.assertEqual(
            _translate(final, ctx),
            [{"type": "user_transcription", "transcript": "hello"}],
        )

    def test_llm_response_accumulates_text(self):
        ctx = _ctx()
        self.assertEqual(_translate({"type": "bot-llm-started"}, ctx), [{"type": "llm_started"}])
        self.assertEqual(_translate({"type": "bot-llm-text", "data": {"text": "Hello "}}, ctx), [])
        self.assertEqual(_translate({"type": "bot-llm-text", "data": {"text": "world"}}, ctx), [])
        self.assertEqual(
            _translate({"type": "bot-llm-stopped"}, ctx),
            [{"type": "llm_response", "text": "Hello world"}],
        )

    def test_tool_call(self):
        ctx = _ctx()
        msg = {
            "type": "llm-function-call-in-progress",
            "data": {"function_name": "get_weather", "arguments": {"city": "Paris"}},
        }
        self.assertEqual(
            _translate(msg, ctx),
            [{"type": "tool_call", "name": "get_weather", "args": {"city": "Paris"}}],
        )

    def test_unmapped_message_ignored(self):
        self.assertEqual(_translate({"type": "metrics", "data": {}}, _ctx()), [])


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

    async def test_llm_response_pass(self):
        self.server.on_text(
            "what is the capital of France?",
            _rtvi("bot-llm-started"),
            _rtvi("bot-llm-text", {"text": "Paris"}),
            _rtvi("bot-llm-stopped"),
        )
        scenario = Scenario(
            name="capital",
            turns=[
                Turn(
                    user="what is the capital of France?",
                    expect=[
                        Expectation(event="llm_started", within_ms=2000),
                        Expectation(event="llm_response", within_ms=2000, text_contains="Paris"),
                    ],
                )
            ],
        )
        result = await run_scenario(scenario, self.server.url)
        self.assertTrue(result.passed, f"failures: {[str(f) for f in result.failures]}")

    async def test_tool_call_pass(self):
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
        scenario = Scenario(
            name="tool",
            turns=[
                Turn(
                    user="weather in Paris?",
                    expect=[
                        Expectation(
                            event="tool_call",
                            within_ms=2000,
                            name="get_weather",
                            args={"city": "Paris"},
                        ),
                    ],
                )
            ],
        )
        result = await run_scenario(scenario, self.server.url)
        self.assertTrue(result.passed, f"failures: {[str(f) for f in result.failures]}")

    async def test_text_mismatch_fails_clearly(self):
        self.server.on_text(
            "hi",
            _rtvi("bot-llm-started"),
            _rtvi("bot-llm-text", {"text": "Paris"}),
            _rtvi("bot-llm-stopped"),
        )
        scenario = Scenario(
            name="mismatch",
            turns=[
                Turn(
                    user="hi",
                    expect=[
                        Expectation(event="llm_response", within_ms=2000, text_contains="London")
                    ],
                )
            ],
        )
        result = await run_scenario(scenario, self.server.url)
        self.assertFalse(result.passed)
        self.assertEqual(len(result.failures), 1)
        self.assertIn("does not contain", result.failures[0].reason)

    async def test_missing_event_times_out(self):
        scenario = Scenario(
            name="never",
            turns=[Turn(user="hi", expect=[Expectation(event="llm_response", within_ms=200)])],
        )
        result = await run_scenario(scenario, self.server.url)
        self.assertFalse(result.passed)
        self.assertEqual(len(result.failures), 1)
        self.assertIn("arrived within", result.failures[0].reason)
        self.assertIn("200ms", result.failures[0].reason)

    async def test_subsequent_assertions_skipped_after_timeout(self):
        scenario = Scenario(
            name="cascading",
            turns=[
                Turn(
                    user="hi",
                    expect=[
                        Expectation(event="llm_started", within_ms=100),
                        Expectation(event="llm_response", within_ms=100),
                        Expectation(event="tool_call", within_ms=100),
                    ],
                )
            ],
        )
        result = await run_scenario(scenario, self.server.url)
        self.assertFalse(result.passed)
        self.assertEqual(len(result.failures), 1, "only the first failed expectation should report")

    async def test_send_after_delays_run(self):
        self.server.on_text("first", _rtvi("bot-llm-started"), _rtvi("bot-llm-stopped"))
        self.server.on_text("second", _rtvi("bot-llm-started"), _rtvi("bot-llm-stopped"))
        scenario = Scenario(
            name="send_after",
            turns=[
                Turn(user="first", expect=[Expectation(event="llm_started", within_ms=2000)]),
                Turn(
                    user="second",
                    expect=[Expectation(event="llm_response", within_ms=2000)],
                    send_after=SendAfter(event="llm_started", delay_ms=200),
                ),
            ],
        )
        result = await run_scenario(scenario, self.server.url)
        self.assertTrue(result.passed, f"failures: {[str(f) for f in result.failures]}")
        self.assertGreaterEqual(result.duration_ms, 200)

    async def test_connect_failure_reported_cleanly(self):
        scenario = Scenario(
            name="no_bot",
            turns=[Turn(user="x", expect=[Expectation(event="llm_started")])],
        )
        result = await run_scenario(
            scenario, f"ws://localhost:{_free_port()}", connect_timeout_s=0.5
        )
        self.assertFalse(result.passed)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].event_name, "<connect>")
        self.assertIn("failed to connect", result.failures[0].reason)

    async def test_reset_sends_eval_reset_message(self):
        self.server.on_text("hi", _rtvi("bot-llm-started"), _rtvi("bot-llm-stopped"))
        scenario = Scenario(
            name="reset",
            turns=[Turn(user="hi", expect=[Expectation(event="llm_started", within_ms=2000)])],
            reset=[{"role": "system", "content": "be terse"}],
        )
        result = await run_scenario(scenario, self.server.url)
        self.assertTrue(result.passed, f"failures: {[str(f) for f in result.failures]}")
        resets = [
            m
            for m in self.server.received
            if m.get("type") == "client-message" and m["data"].get("t") == "eval-reset"
        ]
        self.assertEqual(len(resets), 1)
        self.assertEqual(
            resets[0]["data"]["d"]["messages"], [{"role": "system", "content": "be terse"}]
        )

    async def test_no_reset_when_not_requested(self):
        self.server.on_text("hi", _rtvi("bot-llm-started"), _rtvi("bot-llm-stopped"))
        scenario = Scenario(
            name="noreset",
            turns=[Turn(user="hi", expect=[Expectation(event="llm_started", within_ms=2000)])],
        )
        await run_scenario(scenario, self.server.url)
        resets = [m for m in self.server.received if m.get("type") == "client-message"]
        self.assertEqual(resets, [])


if __name__ == "__main__":
    unittest.main()
