#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import json
import socket
import unittest

import websockets

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.transports.eval.transport import EvalTransport, EvalTransportParams
from pipecat.workers.runner import WorkerRunner


def _free_port() -> int:
    """Bind to port 0 to let the kernel pick a free port, then return it."""
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


class _BotFixture:
    """Spins up an EvalTransport-backed bot in a background task."""

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
        # Wait until the WS server is accepting connections
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


class TestEvalTransportWSRoundTrip(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bot = _BotFixture(_free_port())
        await self.bot.start()

    async def asyncTearDown(self):
        await self.bot.stop()

    async def test_user_input_produces_started_and_stopped_events(self):
        async with websockets.connect(f"ws://localhost:{self.bot.port}") as ws:
            await ws.send(json.dumps({"type": "ready"}))
            await ws.send(json.dumps({"type": "user_input", "text": "hello world"}))

            events = await self._collect_events(ws, count=2, timeout_s=2.0)

        types = [e["type"] for e in events]
        self.assertIn("user_started_speaking", types)
        self.assertIn("user_stopped_speaking", types)

        stopped = next(e for e in events if e["type"] == "user_stopped_speaking")
        self.assertEqual(stopped["transcript"], "hello world")

    async def test_transcript_preserved_with_special_chars(self):
        text = 'Hello! "world" & — friends 🙂'
        async with websockets.connect(f"ws://localhost:{self.bot.port}") as ws:
            await ws.send(json.dumps({"type": "ready"}))
            await ws.send(json.dumps({"type": "user_input", "text": text}))
            events = await self._collect_events(ws, count=2, timeout_s=2.0)
        stopped = next(e for e in events if e["type"] == "user_stopped_speaking")
        self.assertEqual(stopped["transcript"], text)

    async def test_multiple_user_inputs_in_one_connection(self):
        async with websockets.connect(f"ws://localhost:{self.bot.port}") as ws:
            await ws.send(json.dumps({"type": "ready"}))
            await ws.send(json.dumps({"type": "user_input", "text": "first"}))
            await ws.send(json.dumps({"type": "user_input", "text": "second"}))
            events = await self._collect_events(ws, count=4, timeout_s=2.0)

        transcripts = [e["transcript"] for e in events if e["type"] == "user_stopped_speaking"]
        self.assertEqual(transcripts, ["first", "second"])

    async def test_event_has_timestamp_field(self):
        async with websockets.connect(f"ws://localhost:{self.bot.port}") as ws:
            await ws.send(json.dumps({"type": "ready"}))
            await ws.send(json.dumps({"type": "user_input", "text": "x"}))
            events = await self._collect_events(ws, count=2, timeout_s=2.0)
        for e in events:
            self.assertIn("t", e)
            self.assertIsInstance(e["t"], int)
            self.assertGreaterEqual(e["t"], 0)

    async def _collect_events(self, ws, count: int, timeout_s: float) -> list[dict]:
        events: list[dict] = []
        try:
            async with asyncio.timeout(timeout_s):
                while len(events) < count:
                    raw = await ws.recv()
                    events.append(json.loads(raw))
        except TimeoutError:
            pass
        return events


class TestEvalTransportKeepAlive(unittest.IsolatedAsyncioTestCase):
    """Verify keep_alive controls whether on_client_(dis)connected handlers fire."""

    async def _build_bot(self, keep_alive: bool):
        port = _free_port()
        transport = EvalTransport(params=EvalTransportParams(keep_alive=keep_alive), port=port)
        pipeline = Pipeline([transport.input(), transport.output()])
        worker = PipelineWorker(pipeline, params=PipelineParams())
        runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
        await runner.add_workers(worker)
        task = asyncio.create_task(runner.run())
        for _ in range(50):
            try:
                async with asyncio.timeout(0.1):
                    ws = await websockets.connect(f"ws://localhost:{port}")
                await ws.close()
                break
            except (OSError, TimeoutError):
                await asyncio.sleep(0.05)
        return transport, worker, task, port

    async def test_keep_alive_true_fires_connect_suppresses_disconnect(self):
        """keep_alive=True still fires on_client_connected (bots use it to
        kick off conversations per-eval) but suppresses on_client_disconnected
        so bots that tear down on disconnect stay alive across evals."""
        transport, worker, bot_task, port = await self._build_bot(keep_alive=True)
        try:
            connected = asyncio.Event()
            disconnected = asyncio.Event()

            @transport.event_handler("on_client_connected")
            async def _on_conn(_t, _c):
                connected.set()

            @transport.event_handler("on_client_disconnected")
            async def _on_disc(_t, _c):
                disconnected.set()

            async with websockets.connect(f"ws://localhost:{port}"):
                await asyncio.wait_for(connected.wait(), timeout=1.0)

            await asyncio.sleep(0.3)
            self.assertTrue(connected.is_set(), "on_client_connected should fire")
            self.assertFalse(disconnected.is_set(), "on_client_disconnected should be suppressed")
        finally:
            await worker.cancel()
            try:
                await asyncio.wait_for(bot_task, timeout=3.0)
            except (TimeoutError, asyncio.CancelledError, Exception):
                pass

    async def test_keep_alive_false_fires_handlers(self):
        transport, worker, bot_task, port = await self._build_bot(keep_alive=False)
        try:
            connected = asyncio.Event()
            disconnected = asyncio.Event()

            @transport.event_handler("on_client_connected")
            async def _on_conn(_t, _c):
                connected.set()

            @transport.event_handler("on_client_disconnected")
            async def _on_disc(_t, _c):
                disconnected.set()

            async with websockets.connect(f"ws://localhost:{port}"):
                await asyncio.wait_for(connected.wait(), timeout=1.0)

            await asyncio.wait_for(disconnected.wait(), timeout=1.0)
            self.assertTrue(connected.is_set())
            self.assertTrue(disconnected.is_set())
        finally:
            await worker.cancel()
            try:
                await asyncio.wait_for(bot_task, timeout=3.0)
            except (TimeoutError, asyncio.CancelledError, Exception):
                pass

    async def test_bot_survives_multiple_connections_with_keep_alive(self):
        """Default keep_alive=True should let two consecutive harness connections succeed."""
        _, worker, bot_task, port = await self._build_bot(keep_alive=True)
        try:
            for attempt in range(2):
                async with websockets.connect(f"ws://localhost:{port}") as ws:
                    await ws.send(json.dumps({"type": "user_input", "text": f"msg{attempt}"}))
                    try:
                        async with asyncio.timeout(1.0):
                            while True:
                                raw = await ws.recv()
                                evt = json.loads(raw)
                                if evt.get("type") == "user_stopped_speaking":
                                    self.assertEqual(evt["transcript"], f"msg{attempt}")
                                    break
                    except TimeoutError:
                        self.fail(f"connection {attempt} never produced user_stopped_speaking")
        finally:
            await worker.cancel()
            try:
                await asyncio.wait_for(bot_task, timeout=3.0)
            except (TimeoutError, asyncio.CancelledError, Exception):
                pass


if __name__ == "__main__":
    unittest.main()
