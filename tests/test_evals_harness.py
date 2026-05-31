#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import socket
import unittest

import websockets

from pipecat.evals.harness import run_scenario
from pipecat.evals.scenario import Expectation, Scenario, SendAfter, Turn
from pipecat.frames.frames import Frame, LLMMessagesUpdateFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.eval.transport import EvalTransport, EvalTransportParams
from pipecat.workers.runner import WorkerRunner


class _FrameRecorder(FrameProcessor):
    """Pipeline processor that records every frame flowing through it."""

    def __init__(self):
        super().__init__()
        self.frames: list[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        self.frames.append(frame)
        await self.push_frame(frame, direction)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


class _BotFixture:
    """Spins up a passthrough EvalTransport-backed bot in a background task."""

    def __init__(self, port: int):
        self.port = port
        self.transport = EvalTransport(params=EvalTransportParams(), port=port)
        pipeline = Pipeline([self.transport.input(), self.transport.output()])
        self.worker = PipelineWorker(pipeline, params=PipelineParams())
        self.runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
        self._task: asyncio.Task | None = None

    async def start(self):
        await self.runner.add_workers(self.worker)
        self._task = asyncio.create_task(self.runner.run())
        for _ in range(50):
            try:
                async with asyncio.timeout(0.1):
                    ws = await websockets.connect(f"ws://localhost:{self.port}")
                await ws.close()
                return
            except (OSError, TimeoutError):
                await asyncio.sleep(0.05)
        raise RuntimeError(f"EvalTransport never started listening on port {self.port}")

    async def stop(self):
        await self.worker.cancel()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=3.0)
            except (TimeoutError, asyncio.CancelledError, Exception):
                pass

    @property
    def url(self) -> str:
        return f"ws://localhost:{self.port}"


class TestEvalsHarnessIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bot = _BotFixture(_free_port())
        await self.bot.start()

    async def asyncTearDown(self):
        await self.bot.stop()

    async def test_basic_pass(self):
        scenario = Scenario(
            name="basic",
            turns=[
                Turn(
                    user="hello world",
                    expect=[
                        Expectation(event="user_started_speaking"),
                        Expectation(event="user_stopped_speaking", transcript_contains="hello"),
                    ],
                )
            ],
        )
        result = await run_scenario(scenario, self.bot.url)
        self.assertTrue(result.passed, f"failures: {[str(f) for f in result.failures]}")

    async def test_transcript_mismatch_fails_with_clear_reason(self):
        scenario = Scenario(
            name="mismatch",
            turns=[
                Turn(
                    user="hello world",
                    expect=[
                        Expectation(event="user_stopped_speaking", transcript_contains="goodbye"),
                    ],
                )
            ],
        )
        result = await run_scenario(scenario, self.bot.url)
        self.assertFalse(result.passed)
        self.assertEqual(len(result.failures), 1)
        self.assertIn("does not contain", result.failures[0].reason)
        self.assertIn("goodbye", result.failures[0].reason)

    async def test_missing_event_times_out(self):
        """An event the passthrough bot never emits should fail with a timeout reason."""
        scenario = Scenario(
            name="never_arrives",
            turns=[
                Turn(
                    user="hello",
                    expect=[
                        Expectation(event="bot_started_speaking", within_ms=200),
                    ],
                )
            ],
        )
        result = await run_scenario(scenario, self.bot.url)
        self.assertFalse(result.passed)
        self.assertEqual(len(result.failures), 1)
        self.assertIn("arrived within", result.failures[0].reason)
        self.assertIn("200ms", result.failures[0].reason)

    async def test_subsequent_assertions_skipped_after_timeout(self):
        """When an expectation times out, the rest of that turn is skipped (not chained timeouts)."""
        scenario = Scenario(
            name="cascading",
            turns=[
                Turn(
                    user="hello",
                    expect=[
                        Expectation(event="bot_started_speaking", within_ms=100),
                        Expectation(event="bot_stopped_speaking", within_ms=100),
                        Expectation(event="tool_call", within_ms=100),
                    ],
                )
            ],
        )
        result = await run_scenario(scenario, self.bot.url)
        self.assertFalse(result.passed)
        self.assertEqual(len(result.failures), 1, "only the first failed expectation should report")

    async def test_send_after_schedules_with_expected_delay(self):
        scenario = Scenario(
            name="send_after",
            turns=[
                Turn(
                    user="first",
                    expect=[
                        Expectation(event="user_stopped_speaking", transcript_contains="first"),
                    ],
                ),
                Turn(
                    user="second",
                    expect=[
                        Expectation(event="user_stopped_speaking", transcript_contains="second"),
                    ],
                    send_after=SendAfter(event="user_stopped_speaking", delay_ms=200),
                ),
            ],
        )
        result = await run_scenario(scenario, self.bot.url)
        self.assertTrue(result.passed, f"failures: {[str(f) for f in result.failures]}")
        # The send_after delay should make the total run >= 200ms
        self.assertGreaterEqual(result.duration_ms, 200)

    async def test_connect_failure_reported_cleanly(self):
        """Pointing at a port nothing is listening on should fail with a connect-error structure."""
        scenario = Scenario(
            name="no_bot",
            turns=[Turn(user="x", expect=[Expectation(event="user_started_speaking")])],
        )
        result = await run_scenario(
            scenario, f"ws://localhost:{_free_port()}", connect_timeout_s=0.5
        )
        self.assertFalse(result.passed)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].event_name, "<connect>")
        self.assertIn("failed to connect", result.failures[0].reason)

    async def test_reset_pushes_llm_messages_update_frame(self):
        """Default reset (empty list) and explicit reset should both push a
        LLMMessagesUpdateFrame downstream after the harness handshakes."""
        # Rebuild the bot with a frame recorder in the pipeline.
        await self.bot.stop()
        port = _free_port()
        transport = EvalTransport(params=EvalTransportParams(), port=port)
        recorder = _FrameRecorder()
        pipeline = Pipeline([transport.input(), recorder, transport.output()])
        worker = PipelineWorker(pipeline, params=PipelineParams())
        runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
        await runner.add_workers(worker)
        bot_task = asyncio.create_task(runner.run())

        try:
            # Wait for the WS server to come up
            for _ in range(50):
                try:
                    async with asyncio.timeout(0.1):
                        ws = await websockets.connect(f"ws://localhost:{port}")
                    await ws.close()
                    break
                except (OSError, TimeoutError):
                    await asyncio.sleep(0.05)

            scenario = Scenario(
                name="explicit_reset",
                turns=[
                    Turn(
                        user="hi",
                        expect=[Expectation(event="user_stopped_speaking")],
                    )
                ],
                reset=[{"role": "system", "content": "You are a helpful test bot."}],
            )
            result = await run_scenario(scenario, f"ws://localhost:{port}")
            self.assertTrue(result.passed, f"failures: {[str(f) for f in result.failures]}")

            update_frames = [f for f in recorder.frames if isinstance(f, LLMMessagesUpdateFrame)]
            self.assertEqual(
                len(update_frames), 1, "exactly one LLMMessagesUpdateFrame should appear"
            )
            self.assertEqual(
                update_frames[0].messages,
                [{"role": "system", "content": "You are a helpful test bot."}],
            )
        finally:
            await worker.cancel()
            try:
                await asyncio.wait_for(bot_task, timeout=3.0)
            except (TimeoutError, asyncio.CancelledError, Exception):
                pass

    async def test_events_seen_captures_full_stream(self):
        scenario = Scenario(
            name="capture",
            turns=[
                Turn(
                    user="hi",
                    expect=[Expectation(event="user_started_speaking")],
                )
            ],
        )
        result = await run_scenario(scenario, self.bot.url)
        self.assertTrue(result.passed)
        # The passthrough bot emits both user_started and user_stopped — even though
        # the scenario only asserted on user_started, both should be in events_seen.
        types = [e["type"] for e in result.events_seen]
        self.assertIn("user_started_speaking", types)
        self.assertIn("user_stopped_speaking", types)


if __name__ == "__main__":
    unittest.main()
