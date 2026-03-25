#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Reproduction test for ElevenLabs TTS 5-context-per-connection limit bug.

Bug: ElevenLabsTTSService._close_context() sends close_context:True over the
WebSocket fire-and-forget.  During rapid user interruptions the bot creates new
contexts faster than ElevenLabs releases old ones, eventually exceeding the
5-context limit and receiving a 1008 (policy violation) disconnect.  After that
the bot goes silent — no more TTS audio is produced.

This test drives the real ElevenLabsTTSService through a pipecat Pipeline,
alternating between TTS generation and InterruptionFrame, against a mock
WebSocket server that enforces the 5-context limit with realistic cleanup
delay.
"""

import asyncio
import base64
import json
import unittest
from typing import Any, List, Set

import websockets
from websockets.asyncio.server import serve as websocket_serve
from websockets.frames import CloseCode

from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
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
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService, ElevenLabsTTSSettings

# ---------------------------------------------------------------------------
# Mock ElevenLabs WebSocket server
# ---------------------------------------------------------------------------

MAX_CONTEXTS = 5
CLOSE_DELAY_S = 1.0  # Simulated server-side cleanup latency (real ElevenLabs can be slow)
_FAKE_AUDIO = base64.b64encode(b"\x00\x01" * 160).decode("ascii")


class MockElevenLabsServer:
    """Mock server that enforces the ElevenLabs 5-context limit.

    Accepts text messages (opens context), close_context messages (delayed
    cleanup), and disconnects with 1008 when more than MAX_CONTEXTS are
    simultaneously open — matching real ElevenLabs behavior.
    """

    def __init__(self) -> None:
        self.open_contexts: Set[str] = set()
        self.close_requests: List[str] = []
        self.policy_violation_sent = False
        self.peak_open_contexts = 0
        self._server: Any = None
        self.port: int = 0
        self._pending_closes: List[asyncio.Task[None]] = []

    async def _delayed_close(self, context_id: str) -> None:
        await asyncio.sleep(CLOSE_DELAY_S)
        self.open_contexts.discard(context_id)

    async def handler(self, websocket: Any) -> None:
        try:
            async for message in websocket:
                data = json.loads(message)
                context_id = data.get("context_id", "")

                if data.get("close_context"):
                    self.close_requests.append(context_id)
                    task = asyncio.create_task(self._delayed_close(context_id))
                    self._pending_closes.append(task)
                    await asyncio.sleep(0.005)
                    try:
                        await websocket.send(json.dumps({"contextId": context_id, "isFinal": True}))
                    except websockets.exceptions.ConnectionClosed:
                        pass
                    continue

                if data.get("close_socket"):
                    await websocket.close()
                    return

                if data.get("flush"):
                    continue

                text = data.get("text", "")
                if not text.strip() or not context_id:
                    continue

                # New context — enforce the limit
                if context_id not in self.open_contexts:
                    if len(self.open_contexts) >= MAX_CONTEXTS:
                        self.policy_violation_sent = True
                        await websocket.close(
                            CloseCode.POLICY_VIOLATION,
                            f"Max contexts exceeded: {len(self.open_contexts) + 1} > {MAX_CONTEXTS}",
                        )
                        return
                    self.open_contexts.add(context_id)
                    self.peak_open_contexts = max(self.peak_open_contexts, len(self.open_contexts))

                # Respond with audio + alignment
                try:
                    await websocket.send(
                        json.dumps(
                            {
                                "contextId": context_id,
                                "audio": _FAKE_AUDIO,
                                "alignment": {
                                    "chars": list(text[:5]),
                                    "charStartTimesMs": list(range(0, len(text[:5]) * 20, 20)),
                                    "charDurationsMs": [20] * len(text[:5]),
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
        for task in self._pending_closes:
            task.cancel()
        if self._server:
            self._server.close()
            await self._server.wait_closed()


# ---------------------------------------------------------------------------
# ElevenLabs subclass wired to mock server
# ---------------------------------------------------------------------------


class MockedElevenLabsTTSService(ElevenLabsTTSService):
    """ElevenLabsTTSService pointed at a local mock WebSocket server."""

    def __init__(self, server_port: int, **kwargs: Any) -> None:
        super().__init__(
            api_key="test-key",
            url=f"ws://127.0.0.1:{server_port}",
            settings=ElevenLabsTTSSettings(
                voice="test-voice",
                model="eleven_flash_v2_5",
            ),
            **kwargs,
        )
        self._pause_frame_processing = False
        self._stop_frame_timeout_s = 0.1


# ---------------------------------------------------------------------------
# Frame collector
# ---------------------------------------------------------------------------


class FrameCollector(FrameProcessor):
    """Collects TTS and error frames for test assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.frames: List[Frame] = []
        self.errors: List[ErrorFrame] = []

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


class TestElevenLabsContextLimitInPipeline(unittest.IsolatedAsyncioTestCase):
    """Reproduce the 5-context limit bug using the real ElevenLabsTTSService
    in a pipecat pipeline with InterruptionFrames.
    """

    async def test_rapid_interruptions_trigger_1008_and_silence(self) -> None:
        """Rapid interruptions during TTS exhaust the context limit.

        Simulates a fast conversation: each turn the LLM produces text (via
        LLMFullResponseStartFrame → TextFrame → LLMFullResponseEndFrame),
        then the user interrupts (InterruptionFrame).  _close_context sends
        close_context fire-and-forget, but the mock server delays cleanup.
        After enough turns, contexts accumulate past 5 → 1008 disconnect
        → bot goes silent (no more TTSAudioRawFrames).
        """
        server = MockElevenLabsServer()
        await server.start()

        tts = MockedElevenLabsTTSService(server.port)
        collector = FrameCollector()
        pipeline = Pipeline([tts, collector])
        task = PipelineTask(
            pipeline,
            params=PipelineParams(enable_metrics=False),
            cancel_on_idle_timeout=False,
            enable_rtvi=False,
        )
        runner = PipelineRunner()

        audio_per_turn: List[int] = []

        async def drive() -> None:
            await task.queue_frame(StartFrame())
            await asyncio.sleep(0.1)

            for turn in range(10):
                collector.frames.clear()

                # -- LLM produces a response --
                await task.queue_frame(LLMFullResponseStartFrame())
                await task.queue_frame(TextFrame(f"Turn {turn} response."))
                await task.queue_frame(LLMFullResponseEndFrame())
                await asyncio.sleep(0.15)

                # Count audio frames produced this turn
                audio_count = sum(1 for f in collector.frames if isinstance(f, TTSAudioRawFrame))
                audio_per_turn.append(audio_count)

                # -- User interrupts mid-speech --
                await tts.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
                # Brief pause — not enough for server-side cleanup
                await asyncio.sleep(0.03)

            await task.queue_frame(EndFrame())

        try:
            async with asyncio.timeout(15):
                await asyncio.gather(runner.run(task), drive())
        finally:
            await server.stop()

        # 1. Server hit the 5-context limit
        self.assertTrue(
            server.policy_violation_sent,
            f"Expected 1008 policy violation. "
            f"Peak open: {server.peak_open_contexts}, "
            f"Close requests: {len(server.close_requests)}.",
        )

        # 2. _close_context did send close requests (it tried)
        self.assertGreater(
            len(server.close_requests),
            0,
            "close_context messages should have been sent.",
        )

        # 3. Some turns produced audio, proving TTS worked initially
        turns_with_audio = sum(1 for c in audio_per_turn if c > 0)
        self.assertGreater(
            turns_with_audio,
            0,
            "TTS should produce audio for at least some turns.",
        )

        # 4. Some turns produced NO audio — the user-facing bug.
        #    The 1008 disconnect causes the bot to drop audio for one
        #    or more turns until the reconnect succeeds.
        silent_turns = [i for i, c in enumerate(audio_per_turn) if c == 0]
        self.assertGreater(
            len(silent_turns),
            0,
            f"Expected some turns with no audio (bot goes silent after 1008). "
            f"Audio per turn: {audio_per_turn}.",
        )


if __name__ == "__main__":
    unittest.main()
