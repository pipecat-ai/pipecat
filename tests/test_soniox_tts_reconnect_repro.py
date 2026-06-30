#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Reproduction test for SonioxTTSService losing the per-stream config on reconnect.

Bug: ``SonioxTTSService`` only ever sends the per-stream config (the Soniox
"start message") from the eager pre-open path ``on_turn_context_created ->
_send_config``.  ``run_tts`` assumes the stream is already configured and sends
only text.

When the WebSocket reconnects between pre-open and synthesis — e.g. Soniox (or
an intermediate proxy) closes the idle socket with ``1001 going away`` — the
reconnect runs ``_disconnect_websocket``, which clears ``_configured_contexts``.
Nothing re-sends the config on the new connection, so the next ``run_tts`` call
streams text for a ``stream_id`` Soniox never opened.  Soniox replies::

    Soniox TTS error 400 (stream <id>): Stream <id> not found. Send a start message first.

Every sentence in that turn fails the same way and the caller hears dead air.

This test drives the real ``SonioxTTSService`` through a pipecat ``Pipeline``.
It first proves synthesis works on the initial connection, then injects a
server-side close *after* the next turn's stream was pre-opened (mirroring the
production idle-timeout race), waits for the framework to reconnect, and asserts
that the turn still produces audio instead of a "stream not found" error.

Against current ``main`` this test FAILS (no audio, 400 error) — reproducing the
bug.  With ``run_tts`` ensuring the config is (re)sent before text, it PASSES.
"""

import asyncio
import base64
import json
import unittest
from typing import Any

import websockets
from websockets.asyncio.server import serve as websocket_serve

from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    StartFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.soniox.tts import SonioxTTSService

# ---------------------------------------------------------------------------
# Mock Soniox TTS WebSocket server
# ---------------------------------------------------------------------------

_FAKE_AUDIO = base64.b64encode(b"\x00\x01" * 160).decode("ascii")


class MockSonioxTTSServer:
    """Mock Soniox real-time TTS server.

    Tracks which ``stream_id``s have been configured *per connection* — Soniox
    streams are connection-scoped, so a stream opened on one socket does not
    exist on a reconnected socket.  Text for an unconfigured stream is rejected
    with the real Soniox 400 "stream not found" error.
    """

    def __init__(self) -> None:
        self.connection_count = 0
        self.peak_connections = 0
        self.not_found_errors: list[str] = []
        self._active_ws: Any = None
        self._server: Any = None
        self.port: int = 0

    async def drop_active_connection(self) -> None:
        """Close the live socket with 1001, as an idle Soniox/proxy timeout would."""
        ws = self._active_ws
        if ws is not None:
            await ws.close(code=1001, reason="Timeout")

    async def handler(self, websocket: Any) -> None:
        self.connection_count += 1
        self.peak_connections = max(self.peak_connections, self.connection_count)
        self._active_ws = websocket
        configured: set[str] = set()  # stream_ids configured on THIS connection
        try:
            async for message in websocket:
                data = json.loads(message)

                # Keepalive / cancel are no-ops for this mock.
                if data.get("keep_alive") or data.get("cancel"):
                    continue

                # Per-stream config ("start message") — identified by api_key.
                if "api_key" in data:
                    stream_id = data.get("stream_id")
                    if stream_id:
                        configured.add(stream_id)
                    continue

                # Text chunk (or end-of-stream flush).
                if "text" in data:
                    stream_id = data.get("stream_id")
                    if data.get("text_end"):
                        await self._safe_send(
                            websocket, {"stream_id": stream_id, "terminated": True}
                        )
                        continue

                    text = data.get("text", "")
                    if not text.strip() or not stream_id:
                        continue

                    if stream_id in configured:
                        await self._safe_send(
                            websocket, {"stream_id": stream_id, "audio": _FAKE_AUDIO}
                        )
                    else:
                        # Real Soniox response when text arrives for a stream that
                        # never received a config message on this connection.
                        msg = f"Stream {stream_id} not found. Send a start message first."
                        self.not_found_errors.append(msg)
                        await self._safe_send(
                            websocket,
                            {
                                "stream_id": stream_id,
                                "error_code": 400,
                                "error_message": msg,
                            },
                        )
        except websockets.exceptions.ConnectionClosed:
            pass

    @staticmethod
    async def _safe_send(websocket: Any, payload: dict) -> None:
        try:
            await websocket.send(json.dumps(payload))
        except websockets.exceptions.ConnectionClosed:
            pass

    async def start(self) -> None:
        self._server = await websocket_serve(self.handler, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()


# ---------------------------------------------------------------------------
# Soniox subclass wired to the mock server
# ---------------------------------------------------------------------------


class MockedSonioxTTSService(SonioxTTSService):
    """SonioxTTSService pointed at a local mock WebSocket server."""

    def __init__(self, server_port: int, **kwargs: Any) -> None:
        super().__init__(
            api_key="test-key",
            url=f"ws://127.0.0.1:{server_port}",
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Frame collector
# ---------------------------------------------------------------------------


class FrameCollector(FrameProcessor):
    """Collects TTS and error frames for test assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.frames: list[Frame] = []
        self.errors: list[ErrorFrame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, ErrorFrame):
            self.errors.append(frame)
        if isinstance(frame, (TTSStartedFrame, TTSAudioRawFrame, TTSStoppedFrame, ErrorFrame)):
            self.frames.append(frame)
        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class TestSonioxTTSReconnectRepro(unittest.IsolatedAsyncioTestCase):
    """Reproduce the "stream not found" bug after a mid-turn reconnect."""

    async def _wait_for(self, predicate, timeout: float = 5.0) -> bool:
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if predicate():
                return True
            await asyncio.sleep(0.02)
        return False

    async def test_reconnect_between_preopen_and_synthesis_loses_stream(self) -> None:
        """A reconnect after pre-open must not strand the stream config.

        Turn 1 synthesizes normally on the initial connection (proves the
        service + mock work).  Then the server drops the socket right after
        turn 2's stream is pre-opened; the framework reconnects.  Turn 2's text
        must still be synthesized — with the bug it instead hits Soniox's 400
        "stream not found" because ``run_tts`` never re-sent the config.
        """
        server = MockSonioxTTSServer()
        await server.start()

        tts = MockedSonioxTTSService(server.port)
        collector = FrameCollector()
        pipeline = Pipeline([tts, collector])
        task = PipelineTask(
            pipeline,
            params=PipelineParams(enable_metrics=False),
            cancel_on_idle_timeout=False,
            enable_rtvi=False,
        )
        runner = PipelineRunner()

        result: dict = {}

        def audio_count() -> int:
            return sum(1 for f in collector.frames if isinstance(f, TTSAudioRawFrame))

        async def drive() -> None:
            await task.queue_frame(StartFrame())
            connected = await self._wait_for(
                lambda: tts._websocket is not None and server.connection_count >= 1
            )
            self.assertTrue(connected, "Service never connected to the mock server.")

            # --- Turn 1: normal synthesis on the initial connection ---
            collector.frames.clear()
            await task.queue_frame(LLMFullResponseStartFrame())
            await task.queue_frame(TextFrame("Sveiki!"))
            await task.queue_frame(LLMFullResponseEndFrame())
            await self._wait_for(lambda: audio_count() > 0, timeout=3.0)
            result["audio_turn1"] = audio_count()

            # --- Turn 2: pre-open, then lose the socket before synthesis ---
            collector.frames.clear()
            collector.errors.clear()

            # LLMFullResponseStartFrame pre-opens the stream (sends config) on
            # the current connection via on_turn_context_created.
            await task.queue_frame(LLMFullResponseStartFrame())
            await asyncio.sleep(0.1)  # let the config reach the server

            # Idle-timeout: Soniox/proxy closes the socket with 1001 going away.
            first_conn = server.connection_count
            await server.drop_active_connection()

            # The framework's receive loop reconnects (clearing _configured_contexts).
            reconnected = await self._wait_for(
                lambda: server.connection_count > first_conn and tts._websocket is not None
            )
            self.assertTrue(reconnected, "Service did not reconnect after the socket drop.")
            await asyncio.sleep(0.1)  # let the new connection settle

            # Now the turn's text is synthesized — on the reconnected socket.
            await task.queue_frame(TextFrame("Vai varat šo summu samaksāt tuvākajā laikā?"))
            await task.queue_frame(LLMFullResponseEndFrame())
            await self._wait_for(
                lambda: audio_count() > 0 or len(collector.errors) > 0, timeout=3.0
            )
            await asyncio.sleep(0.1)
            result["audio_turn2"] = audio_count()
            result["errors_turn2"] = [str(e.error) for e in collector.errors]

            await task.queue_frame(EndFrame())

        try:
            async with asyncio.timeout(20):
                await asyncio.gather(runner.run(task), drive())
        finally:
            await server.stop()

        # Sanity: the service reconnected and synthesis worked before the drop.
        self.assertGreaterEqual(
            server.peak_connections, 2, "Expected a reconnect (>= 2 server connections)."
        )
        self.assertGreater(
            result.get("audio_turn1", 0), 0, "Turn 1 should synthesize on the initial connection."
        )

        not_found = [e for e in result.get("errors_turn2", []) if "not found" in e]

        # The bug: text was streamed to a stream the reconnected socket never
        # opened, so Soniox rejected it and the turn produced no audio.
        self.assertEqual(
            not_found,
            [],
            f"run_tts streamed text without (re)sending the stream config after "
            f"reconnect; Soniox rejected it: {not_found}",
        )
        self.assertGreater(
            result.get("audio_turn2", 0),
            0,
            f"Turn 2 produced no audio after the reconnect. Errors: {result.get('errors_turn2')}.",
        )


if __name__ == "__main__":
    unittest.main()
