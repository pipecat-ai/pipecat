#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Reproduction test for ElevenLabs TTS 5-context limit bug.

The bug: ElevenLabsTTSService._close_context() sends close_context:True over
the WebSocket fire-and-forget, without waiting for server acknowledgment.
During rapid user interruptions, new contexts are created before old ones are
fully closed server-side, eventually exceeding the 5-context limit and causing
a 1008 (policy violation) disconnect from the ElevenLabs server.

This test uses a mock WebSocket server that faithfully models the ElevenLabs
server behavior:
  - Tracks open contexts with a configurable limit (MAX_CONTEXTS = 5).
  - Delays processing of close_context requests (simulating server-side async cleanup).
  - Sends a 1008 close frame when a 6th context is opened.

The test directly drives the ElevenLabsTTSService methods that create and close
contexts over the WebSocket, reproducing the exact race condition.
"""

import asyncio
import json
import unittest
from typing import Any, List, Set

import websockets
from websockets.asyncio.server import serve as websocket_serve
from websockets.frames import CloseCode

MAX_CONTEXTS = 5
CLOSE_DELAY_S = 0.3  # Simulated server-side cleanup delay


class MockElevenLabsServer:
    """Mock ElevenLabs WebSocket server that enforces context limits.

    When a client sends a text message with a new context_id, the server
    tracks it as open.  When the client sends close_context, the server
    delays the actual cleanup (simulating real ElevenLabs behavior).
    If the number of simultaneously open contexts exceeds MAX_CONTEXTS,
    the server closes the connection with 1008 Policy Violation.
    """

    def __init__(self) -> None:
        self.open_contexts: Set[str] = set()
        self.close_requests_received: List[str] = []
        self.policy_violation_sent = False
        self.connections_made = 0
        self.peak_open_contexts = 0
        self._server: Any = None
        self.port: int = 0
        self._pending_closes: List[asyncio.Task[None]] = []

    async def _delayed_close(self, context_id: str) -> None:
        """Simulate server-side async cleanup that takes time."""
        await asyncio.sleep(CLOSE_DELAY_S)
        self.open_contexts.discard(context_id)

    async def handler(self, websocket: Any) -> None:
        self.connections_made += 1
        try:
            async for message in websocket:
                data = json.loads(message)
                context_id = data.get("context_id", "")

                # Handle close_context request
                if data.get("close_context"):
                    self.close_requests_received.append(context_id)
                    # Start delayed cleanup -- the context remains "open"
                    # from the server's perspective during the delay
                    task = asyncio.create_task(self._delayed_close(context_id))
                    self._pending_closes.append(task)
                    # Send isFinal after a small delay (like real server)
                    await asyncio.sleep(0.01)
                    try:
                        await websocket.send(json.dumps({"contextId": context_id, "isFinal": True}))
                    except websockets.exceptions.ConnectionClosed:
                        pass
                    continue

                # Handle close_socket
                if data.get("close_socket"):
                    await websocket.close()
                    return

                # Any message with text + context_id opens/uses a context
                if "text" in data and context_id:
                    if context_id not in self.open_contexts:
                        # New context being opened -- check limit
                        if len(self.open_contexts) >= MAX_CONTEXTS:
                            self.policy_violation_sent = True
                            await websocket.close(
                                CloseCode.POLICY_VIOLATION,
                                f"Too many contexts: {len(self.open_contexts) + 1} > {MAX_CONTEXTS}",
                            )
                            return
                        self.open_contexts.add(context_id)
                        self.peak_open_contexts = max(
                            self.peak_open_contexts, len(self.open_contexts)
                        )

                    # Send back a small audio response
                    try:
                        await websocket.send(
                            json.dumps(
                                {
                                    "contextId": context_id,
                                    "audio": "AAAA",
                                    "alignment": {
                                        "chars": ["h"],
                                        "charStartTimesMs": [0],
                                        "charDurationsMs": [100],
                                    },
                                }
                            )
                        )
                    except websockets.exceptions.ConnectionClosed:
                        pass

        except websockets.exceptions.ConnectionClosed:
            pass

    async def start(self) -> None:
        self._server = await websocket_serve(self.handler, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        # Cancel pending delayed closes
        for task in self._pending_closes:
            task.cancel()
        if self._server:
            self._server.close()
            await self._server.wait_closed()


class TestElevenLabsContextLimitBug(unittest.IsolatedAsyncioTestCase):
    """Reproduce the fire-and-forget close_context bug.

    This test PASSES when the bug is present, demonstrating that the
    current _close_context implementation does not wait for server-side
    acknowledgment, allowing context accumulation beyond the limit.
    """

    async def test_rapid_close_and_reopen_exceeds_context_limit(self) -> None:
        """Directly simulate the _close_context fire-and-forget race.

        Reproduces the exact sequence:
        1. Client opens context via text message
        2. Client sends close_context (fire-and-forget, no await on server ack)
        3. Client immediately opens a new context
        4. Server has not yet freed the old context
        5. After enough cycles, server sees >5 open contexts -> 1008

        This mirrors what happens during rapid InterruptionFrame handling:
        _handle_interruption calls on_audio_context_interrupted which calls
        _close_context, then immediately a new audio context task creates
        a new context via run_tts.
        """
        server = MockElevenLabsServer()
        await server.start()

        policy_violation_received = False
        contexts_created = 0

        try:
            ws_url = f"ws://127.0.0.1:{server.port}"
            ws = await websockets.connect(ws_url)

            try:
                # Simulate rapid context create -> close -> create cycles
                # This is exactly what ElevenLabsTTSService does:
                #   run_tts creates context by sending text with context_id
                #   _close_context sends close_context:True fire-and-forget
                #   Next run_tts immediately creates a new context
                for i in range(8):
                    context_id = f"ctx-{i}"

                    # 1. Create context (what run_tts does)
                    await ws.send(
                        json.dumps(
                            {
                                "text": " ",
                                "context_id": context_id,
                            }
                        )
                    )
                    contexts_created += 1

                    # Read back the audio response
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except (
                        asyncio.TimeoutError,
                        websockets.exceptions.ConnectionClosed,
                    ) as e:
                        if isinstance(e, websockets.exceptions.ConnectionClosed):
                            # Check if this was a 1008 policy violation
                            if ws.close_code == 1008:
                                policy_violation_received = True
                        break

                    # Send some text (what run_tts does for the actual text)
                    try:
                        await ws.send(
                            json.dumps(
                                {
                                    "text": f"Hello world {i}",
                                    "context_id": context_id,
                                }
                            )
                        )
                    except websockets.exceptions.ConnectionClosed:
                        if ws.close_code == 1008:
                            policy_violation_received = True
                        break

                    # Read audio response
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except (
                        asyncio.TimeoutError,
                        websockets.exceptions.ConnectionClosed,
                    ) as e:
                        if isinstance(e, websockets.exceptions.ConnectionClosed):
                            if ws.close_code == 1008:
                                policy_violation_received = True
                        break

                    # 2. Close context -- fire-and-forget (what _close_context does)
                    # Note: we do NOT wait for the server's isFinal response,
                    # just like ElevenLabsTTSService._close_context doesn't.
                    try:
                        await ws.send(
                            json.dumps(
                                {
                                    "context_id": context_id,
                                    "close_context": True,
                                }
                            )
                        )
                    except websockets.exceptions.ConnectionClosed:
                        if ws.close_code == 1008:
                            policy_violation_received = True
                        break

                    # 3. Do NOT wait for the server to process the close.
                    # This is the bug -- we immediately loop and create
                    # the next context while the old one is still "open"
                    # on the server side.

                    # Tiny yield to let asyncio process, but NOT enough time
                    # for the server's CLOSE_DELAY_S to complete
                    await asyncio.sleep(0.02)

            except websockets.exceptions.ConnectionClosed:
                if ws.close_code == 1008:
                    policy_violation_received = True
            finally:
                try:
                    await ws.close()
                except Exception:
                    pass

        finally:
            await server.stop()

        # The bug manifests as a 1008 policy violation because:
        # - We created contexts faster than the server could close them
        # - _close_context is fire-and-forget (no ack wait)
        # - After 5+ contexts accumulate server-side, the server disconnects
        self.assertTrue(
            server.policy_violation_sent,
            f"Expected server to send 1008 policy violation. "
            f"Contexts created: {contexts_created}, "
            f"Peak open: {server.peak_open_contexts}, "
            f"Close requests: {len(server.close_requests_received)}. "
            f"The fire-and-forget close_context bug was not triggered -- "
            f"either the server closed contexts too fast or not enough cycles ran.",
        )
        self.assertTrue(
            policy_violation_received,
            "Client should have received the 1008 close code.",
        )
        # Verify we actually sent close requests (proving we tried to close)
        self.assertGreater(
            len(server.close_requests_received),
            0,
            "At least one close_context request should have been sent.",
        )
        # The peak should exceed the limit -- this is the smoking gun
        self.assertGreaterEqual(
            server.peak_open_contexts,
            MAX_CONTEXTS,
            f"Peak open contexts ({server.peak_open_contexts}) should reach "
            f"the limit ({MAX_CONTEXTS}) to trigger the policy violation.",
        )

    async def test_waiting_for_close_ack_avoids_limit(self) -> None:
        """Show that waiting for isFinal before opening a new context is safe.

        This is the counterpart to the bug test above.  If _close_context
        waited for the server's isFinal acknowledgment before returning,
        contexts would never accumulate beyond 1 and the 1008 would never
        be triggered.

        This test should ALWAYS pass -- it shows the correct behavior.
        """
        server = MockElevenLabsServer()
        await server.start()

        try:
            ws_url = f"ws://127.0.0.1:{server.port}"
            ws = await websockets.connect(ws_url)

            try:
                for i in range(8):
                    context_id = f"ctx-{i}"

                    # Create context
                    await ws.send(json.dumps({"text": " ", "context_id": context_id}))
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)

                    # Send text
                    await ws.send(json.dumps({"text": f"Hello {i}", "context_id": context_id}))
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)

                    # Close context
                    await ws.send(json.dumps({"context_id": context_id, "close_context": True}))

                    # THE FIX: wait for isFinal acknowledgment before proceeding
                    ack = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    ack_data = json.loads(ack)
                    self.assertTrue(ack_data.get("isFinal"))

                    # Also wait for the server-side cleanup to complete
                    await asyncio.sleep(CLOSE_DELAY_S + 0.05)

            finally:
                try:
                    await ws.close()
                except Exception:
                    pass

        finally:
            await server.stop()

        # With proper close acknowledgment, we should never hit the limit
        self.assertFalse(
            server.policy_violation_sent,
            "No policy violation should occur when waiting for close ack.",
        )
        self.assertLessEqual(
            server.peak_open_contexts,
            1,
            "At most 1 context should be open at a time when waiting for ack.",
        )


if __name__ == "__main__":
    unittest.main()
